"""
Connection pooling for KacheDB clients.

Provides thread-safe synchronous and asyncio-based connection pools that
reuse TCP connections across multiple operations.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import TYPE_CHECKING

from .connection import Connection
from .exceptions import ConnectionError, PoolExhaustedError

if TYPE_CHECKING:
    from .resp import AsyncRespReader


class ConnectionPool:
    """Thread-safe synchronous connection pool.

    Maintains a pool of reusable :class:`~kachedb.connection.Connection`
    objects.  Connections are lazily created on first checkout and health-checked
    before reuse.

    Parameters
    ----------
    host : str
        Server hostname or IP address.
    port : int
        Server TCP port.
    max_connections : int
        Maximum number of simultaneous connections.
    socket_timeout : float | None
        Per-connection socket timeout in seconds.
    decode_responses : bool
        If ``True``, decode byte responses to UTF-8 strings.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        *,
        max_connections: int = 10,
        socket_timeout: float | None = 5.0,
        decode_responses: bool = False,
        password: str | None = None,
        ssl: bool = False,
        ssl_keyfile: str | None = None,
        ssl_certfile: str | None = None,
        ssl_ca_certs: str | None = None,
        ssl_check_hostname: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.decode_responses = decode_responses
        self.password = password
        self.ssl = ssl
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile
        self.ssl_ca_certs = ssl_ca_certs
        self.ssl_check_hostname = ssl_check_hostname

        self._pool: queue.Queue[Connection] = queue.Queue(maxsize=max_connections)
        self._active_count = 0
        self._lock = threading.Lock()

    def get_connection(self) -> Connection:
        """Acquire a connection from the pool.

        Returns a healthy, connected :class:`Connection`.  Creates a new
        connection if the pool is empty and the limit has not been reached.

        Raises
        ------
        PoolExhaustedError
            If all connections are in use and ``max_connections`` is reached.
        """
        # Try to get a connection from the pool (non-blocking).
        try:
            conn = self._pool.get_nowait()
            # Verify it's still alive.
            if conn.is_connected and conn.check_health():
                return conn
            # Stale — discard and create fresh.
            conn.disconnect()
            with self._lock:
                self._active_count -= 1
        except queue.Empty:
            pass

        # Create a new connection if under limit.
        with self._lock:
            if self._active_count >= self.max_connections:
                raise PoolExhaustedError(
                    f"Connection pool exhausted (max_connections={self.max_connections})"
                )
            self._active_count += 1

        conn = Connection(
            host=self.host,
            port=self.port,
            socket_timeout=self.socket_timeout,
            decode_responses=self.decode_responses,
            ssl=self.ssl,
            ssl_keyfile=self.ssl_keyfile,
            ssl_certfile=self.ssl_certfile,
            ssl_ca_certs=self.ssl_ca_certs,
            ssl_check_hostname=self.ssl_check_hostname,
        )
        conn.connect()
        if self.password is not None:
            conn.send_command("AUTH", self.password)
            res = conn.read_response()
            if res != b"OK" and res != "OK":
                conn.disconnect()
                with self._lock:
                    self._active_count -= 1
                raise ConnectionError(f"Authentication failed: {res!r}")
        return conn

    def release_connection(self, conn: Connection) -> None:
        """Return a connection to the pool for reuse."""
        if conn.is_connected:
            try:
                self._pool.put_nowait(conn)
                return
            except queue.Full:
                pass

        # Pool is full or connection is broken — close it.
        conn.disconnect()
        with self._lock:
            self._active_count -= 1

    def disconnect_all(self) -> None:
        """Close all pooled connections."""
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.disconnect()
            except queue.Empty:
                break

        with self._lock:
            self._active_count = 0


class AsyncConnectionPool:
    """Async connection pool using ``asyncio.Queue``.

    Parameters
    ----------
    host : str
        Server hostname or IP address.
    port : int
        Server TCP port.
    max_connections : int
        Maximum number of simultaneous connections.
    decode_responses : bool
        If ``True``, decode byte responses to UTF-8 strings.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        *,
        max_connections: int = 10,
        decode_responses: bool = False,
        password: str | None = None,
        ssl: bool = False,
        ssl_keyfile: str | None = None,
        ssl_certfile: str | None = None,
        ssl_ca_certs: str | None = None,
        ssl_check_hostname: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.decode_responses = decode_responses
        self.password = password
        self.ssl = ssl
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile
        self.ssl_ca_certs = ssl_ca_certs
        self.ssl_check_hostname = ssl_check_hostname

        self._pool: asyncio.Queue[
            tuple[asyncio.StreamReader, asyncio.StreamWriter, AsyncRespReader]
        ] = asyncio.Queue(maxsize=max_connections)
        self._active_count = 0
        self._lock = asyncio.Lock()

    async def get_connection(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, AsyncRespReader]:
        """Acquire an async connection from the pool."""
        from .resp import AsyncRespReader as _AsyncRespReader

        # Try cached connection.
        try:
            reader, writer, resp_reader = self._pool.get_nowait()
            if not writer.is_closing():
                return reader, writer, resp_reader
            # Stale.
            writer.close()
            async with self._lock:
                self._active_count -= 1
        except asyncio.QueueEmpty:
            pass

        async with self._lock:
            if self._active_count >= self.max_connections:
                raise PoolExhaustedError(
                    f"Async connection pool exhausted (max_connections={self.max_connections})"
                )
            self._active_count += 1

        ssl_ctx = None
        if self.ssl:
            import ssl as _ssl

            ssl_ctx = _ssl.create_default_context(cafile=self.ssl_ca_certs)
            if not self.ssl_check_hostname:
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = _ssl.CERT_NONE
            if self.ssl_certfile and self.ssl_keyfile:
                ssl_ctx.load_cert_chain(
                    certfile=self.ssl_certfile, keyfile=self.ssl_keyfile
                )

        try:
            reader, writer = await asyncio.open_connection(
                self.host,
                self.port,
                ssl=ssl_ctx,
                server_hostname=self.host
                if (self.ssl and self.ssl_check_hostname)
                else None,
            )
        except OSError as exc:
            async with self._lock:
                self._active_count -= 1
            raise ConnectionError(
                f"Failed to connect to KacheDB at {self.host}:{self.port}: {exc}"
            ) from exc

        resp_reader = _AsyncRespReader(reader)
        if self.password is not None:
            from .resp import encode_command

            writer.write(encode_command(["AUTH", self.password]))
            await writer.drain()
            res = await resp_reader.read_response()
            if res != b"OK" and res != "OK":
                writer.close()
                async with self._lock:
                    self._active_count -= 1
                raise ConnectionError(f"Authentication failed: {res!r}")

        return reader, writer, resp_reader

    async def release_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        resp_reader: AsyncRespReader,
    ) -> None:
        """Return an async connection to the pool."""
        if not writer.is_closing():
            try:
                self._pool.put_nowait((reader, writer, resp_reader))
                return
            except asyncio.QueueFull:
                pass

        writer.close()
        async with self._lock:
            self._active_count -= 1

    async def disconnect_all(self) -> None:
        """Close all pooled connections."""
        while True:
            try:
                _, writer, _ = self._pool.get_nowait()
                writer.close()
            except asyncio.QueueEmpty:
                break

        async with self._lock:
            self._active_count = 0
