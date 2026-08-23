"""
Async KacheDB client using ``asyncio``.

Provides the same high-level API as :class:`~kachedb.client.KacheClient`
but uses non-blocking ``asyncio`` streams for use in async ML inference
pipelines (vLLM, SGLang, etc.).

Usage::

    import asyncio
    from kachedb import AsyncKacheClient

    async def main():
        async with AsyncKacheClient() as client:
            await client.set("user:1", "alice", ex=3600)
            print(await client.get("user:1"))

    asyncio.run(main())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .exceptions import ConnectionError
from .pipeline import AsyncPipeline
from .pool import AsyncConnectionPool
from .resp import AsyncRespReader, RespValue, encode_command

if TYPE_CHECKING:
    import asyncio


class AsyncKacheClient:
    """High-level async KacheDB client.

    Parameters
    ----------
    host : str
        Server hostname or IP address.
    port : int
        Server TCP port.
    decode_responses : bool
        If ``True``, decode byte responses to UTF-8 strings.
    max_connections : int
        Maximum async connection pool size.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        *,
        decode_responses: bool = False,
        max_connections: int = 10,
    ) -> None:
        self.host = host
        self.port = port
        self.decode_responses = decode_responses
        self._pool = AsyncConnectionPool(
            host=host,
            port=port,
            max_connections=max_connections,
            decode_responses=decode_responses,
        )
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._resp_reader: AsyncRespReader | None = None

    async def connect(self) -> AsyncKacheClient:
        """Establish a dedicated async connection."""
        self._reader, self._writer, self._resp_reader = await self._pool.get_connection()
        return self

    async def close(self) -> None:
        """Release the connection back to the pool."""
        if self._writer is not None and self._reader is not None and self._resp_reader is not None:
            await self._pool.release_connection(self._reader, self._writer, self._resp_reader)
            self._reader = None
            self._writer = None
            self._resp_reader = None

    async def disconnect_all(self) -> None:
        """Close all pooled connections."""
        await self._pool.disconnect_all()

    async def __aenter__(self) -> AsyncKacheClient:
        return await self.connect()

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ── Internal Helpers ──────────────────────────────────────────────────

    async def _execute(self, *args: str | bytes) -> RespValue:
        """Execute a single command and return the response."""
        if self._writer is None or self._resp_reader is None:
            raise ConnectionError("Not connected to KacheDB")

        data = encode_command(list(args))
        self._writer.write(data)
        await self._writer.drain()

        response = await self._resp_reader.read_response()

        if self.decode_responses and isinstance(response, bytes):
            return response.decode("utf-8")

        return response

    # ── Redis-Compatible Commands ─────────────────────────────────────────

    async def ping(self, message: str | None = None) -> str:
        """Send ``PING`` and return ``PONG`` or the echoed message."""
        args: list[str | bytes] = ["PING"]
        if message is not None:
            args.append(message)
        result = await self._execute(*args)
        return str(result) if result is not None else "PONG"

    async def get(self, key: str | bytes) -> bytes | str | None:
        """Retrieve the value for *key*."""
        return await self._execute("GET", key)  # type: ignore[return-value]

    async def set(
        self,
        key: str | bytes,
        value: str | bytes,
        *,
        ex: int | None = None,
        px: int | None = None,
    ) -> bool:
        """Store *value* under *key* with an optional TTL."""
        args: list[str | bytes] = ["SET", key, value]
        if ex is not None:
            args.extend(["EX", str(ex)])
        elif px is not None:
            args.extend(["PX", str(px)])
        result = await self._execute(*args)
        return result == "OK"

    async def mget(self, *keys: str | bytes) -> list[bytes | str | None]:
        """Batch-retrieve values for multiple keys."""
        if not keys:
            return []
        result = await self._execute("MGET", *keys)
        return result if isinstance(result, list) else []  # type: ignore[return-value]

    async def delete(self, *keys: str | bytes) -> int:
        """Delete one or more keys."""
        if not keys:
            return 0
        result = await self._execute("DEL", *keys)
        return int(result) if isinstance(result, int) else 0

    async def exists(self, *keys: str | bytes) -> int:
        """Check existence of one or more keys."""
        if not keys:
            return 0
        result = await self._execute("EXISTS", *keys)
        return int(result) if isinstance(result, int) else 0

    # ── Pipeline ──────────────────────────────────────────────────────────

    def pipeline(self) -> AsyncPipeline:
        """Create an async pipeline for batching commands."""
        if self._writer is None or self._resp_reader is None:
            raise ConnectionError("Not connected to KacheDB")
        return AsyncPipeline(self._writer, self._resp_reader)
