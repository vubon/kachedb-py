"""
Synchronous KacheDB client.

Provides a high-level, Redis-like API for communicating with a KacheDB
daemon over TCP using the RESP2 wire protocol.

Usage::

    from kachedb import KacheClient

    with KacheClient() as client:
        client.set("user:1", "alice", ex=3600)
        print(client.get("user:1"))  # b"alice"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .pipeline import Pipeline
from .pool import ConnectionPool

if TYPE_CHECKING:
    from .connection import Connection
    from .resp import RespValue


class KacheClient:
    """High-level synchronous KacheDB client.

    Supports both direct connection mode and connection pool mode.

    Parameters
    ----------
    host : str
        Server hostname or IP address.
    port : int
        Server TCP port.
    socket_timeout : float | None
        Socket timeout in seconds.
    decode_responses : bool
        If ``True``, decode byte responses to UTF-8 strings.
    max_connections : int
        Maximum pool size.  Set to ``1`` to disable pooling.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        *,
        socket_timeout: float | None = 5.0,
        decode_responses: bool = False,
        max_connections: int = 10,
    ) -> None:
        self.host = host
        self.port = port
        self._pool = ConnectionPool(
            host=host,
            port=port,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            decode_responses=decode_responses,
        )
        self._conn: Connection | None = None

    def connect(self) -> KacheClient:
        """Establish a dedicated connection (used with context manager)."""
        self._conn = self._pool.get_connection()
        return self

    def close(self) -> None:
        """Release the dedicated connection back to the pool."""
        if self._conn is not None:
            self._pool.release_connection(self._conn)
            self._conn = None

    def disconnect_all(self) -> None:
        """Close all connections in the pool."""
        self._pool.disconnect_all()

    def __enter__(self) -> KacheClient:
        return self.connect()

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _get_conn(self) -> Connection:
        """Get the active connection or checkout from pool."""
        if self._conn is not None:
            return self._conn
        # For non-context-manager usage: get a connection per-call.
        return self._pool.get_connection()

    def _release_conn(self, conn: Connection) -> None:
        """Release if it was a per-call checkout."""
        if self._conn is None:
            self._pool.release_connection(conn)

    def _execute(self, *args: str | bytes) -> RespValue:
        """Execute a single command and return the response."""
        conn = self._get_conn()
        try:
            conn.send_command(*args)
            return conn.read_response()
        except Exception:
            # On error, discard the connection.
            conn.disconnect()
            raise
        finally:
            self._release_conn(conn)

    # ── Redis-Compatible Commands ─────────────────────────────────────────

    def ping(self, message: str | None = None) -> str:
        """Send ``PING`` and return ``PONG`` or the echoed message.

        Parameters
        ----------
        message : str | None
            Optional message to echo back.
        """
        args: list[str | bytes] = ["PING"]
        if message is not None:
            args.append(message)
        result = self._execute(*args)
        return str(result) if result is not None else "PONG"

    def get(self, key: str | bytes) -> bytes | str | None:
        """Retrieve the value for *key*.

        Returns ``None`` if the key does not exist or has expired.
        """
        return self._execute("GET", key)  # type: ignore[return-value]

    def set(
        self,
        key: str | bytes,
        value: str | bytes,
        *,
        ex: int | None = None,
        px: int | None = None,
    ) -> bool:
        """Store *value* under *key* with an optional TTL.

        Parameters
        ----------
        key : str | bytes
            The cache key.
        value : str | bytes
            The value to store (binary-safe, up to 2 MB).
        ex : int | None
            Expiration time in **seconds**.
        px : int | None
            Expiration time in **milliseconds**.

        Returns
        -------
        bool
            ``True`` if the server acknowledged with ``OK``.
        """
        args: list[str | bytes] = ["SET", key, value]
        if ex is not None:
            args.extend(["EX", str(ex)])
        elif px is not None:
            args.extend(["PX", str(px)])
        result = self._execute(*args)
        return result == "OK"

    def mget(self, *keys: str | bytes) -> list[bytes | str | None]:
        """Batch-retrieve values for multiple keys in a single round-trip.

        Returns a list of values (or ``None`` for missing/expired keys)
        in the same order as the input keys.
        """
        if not keys:
            return []
        result = self._execute("MGET", *keys)
        return result if isinstance(result, list) else []  # type: ignore[return-value]

    def delete(self, *keys: str | bytes) -> int:
        """Delete one or more keys.

        Returns the number of keys that were actually removed.
        """
        if not keys:
            return 0
        result = self._execute("DEL", *keys)
        return int(result) if isinstance(result, int) else 0

    def exists(self, *keys: str | bytes) -> int:
        """Check existence of one or more keys.

        Returns the count of keys that exist and have not expired.
        """
        if not keys:
            return 0
        result = self._execute("EXISTS", *keys)
        return int(result) if isinstance(result, int) else 0

    # ── Pipeline ──────────────────────────────────────────────────────────

    def pipeline(self) -> Pipeline:
        """Create a pipeline for batching multiple commands.

        Usage::

            pipe = client.pipeline()
            pipe.set("a", "1")
            pipe.get("a")
            results = pipe.execute()
        """
        conn = self._get_conn()
        return Pipeline(conn)
