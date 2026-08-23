"""
Pipeline batching for KacheDB commands.

Accumulates multiple commands and sends them in a single ``sendall()`` call,
then reads all responses in sequence.  This amortizes the TCP round-trip
overhead and matches the server's native pipelined RESP frame processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .resp import RespValue, encode_commands

if TYPE_CHECKING:
    import asyncio

    from .connection import Connection
    from .resp import AsyncRespReader


class Pipeline:
    """Synchronous command pipeline.

    Usage::

        pipe = client.pipeline()
        pipe.set("a", "1")
        pipe.set("b", "2")
        pipe.get("a")
        results = pipe.execute()
        # results == [True, True, b"1"]

    Parameters
    ----------
    connection : Connection
        The underlying TCP connection to send commands through.
    """

    def __init__(self, connection: Connection) -> None:
        self._conn = connection
        self._commands: list[list[str | bytes]] = []

    def _queue(self, *args: str | bytes) -> Pipeline:
        """Queue a command for later execution."""
        self._commands.append(list(args))
        return self

    # ── Redis-compatible command methods ───────────────────────────────────

    def ping(self) -> Pipeline:
        """Queue a PING command."""
        return self._queue("PING")

    def get(self, key: str | bytes) -> Pipeline:
        """Queue a GET command."""
        return self._queue("GET", key)

    def set(
        self,
        key: str | bytes,
        value: str | bytes,
        *,
        ex: int | None = None,
        px: int | None = None,
    ) -> Pipeline:
        """Queue a SET command with optional TTL."""
        args: list[str | bytes] = ["SET", key, value]
        if ex is not None:
            args.extend(["EX", str(ex)])
        elif px is not None:
            args.extend(["PX", str(px)])
        self._commands.append(args)
        return self

    def mget(self, *keys: str | bytes) -> Pipeline:
        """Queue an MGET command."""
        return self._queue("MGET", *keys)

    def delete(self, *keys: str | bytes) -> Pipeline:
        """Queue a DEL command."""
        return self._queue("DEL", *keys)

    def exists(self, *keys: str | bytes) -> Pipeline:
        """Queue an EXISTS command."""
        return self._queue("EXISTS", *keys)

    # ── Execution ─────────────────────────────────────────────────────────

    def execute(self) -> list[RespValue]:
        """Send all queued commands and collect responses.

        Returns a list of responses in the same order as the queued commands.
        After execution, the pipeline is reset.
        """
        if not self._commands:
            return []

        packed = encode_commands(self._commands)
        count = len(self._commands)
        self._commands.clear()

        self._conn.send_packed(packed)

        results: list[RespValue] = []
        for _ in range(count):
            results.append(self._conn.read_response())
        return results

    def __len__(self) -> int:
        return len(self._commands)

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *args: Any) -> None:
        self._commands.clear()


class AsyncPipeline:
    """Async command pipeline.

    Usage::

        async with client.pipeline() as pipe:
            pipe.set("a", "1")
            pipe.set("b", "2")
            pipe.get("a")
            results = await pipe.execute()

    Parameters
    ----------
    writer : asyncio.StreamWriter
        The async stream writer.
    resp_reader : AsyncRespReader
        The async RESP response reader.
    """

    def __init__(
        self,
        writer: asyncio.StreamWriter,
        resp_reader: AsyncRespReader,
    ) -> None:
        self._writer = writer
        self._resp_reader = resp_reader
        self._commands: list[list[str | bytes]] = []

    def _queue(self, *args: str | bytes) -> AsyncPipeline:
        self._commands.append(list(args))
        return self

    def ping(self) -> AsyncPipeline:
        """Queue a PING command."""
        return self._queue("PING")

    def get(self, key: str | bytes) -> AsyncPipeline:
        """Queue a GET command."""
        return self._queue("GET", key)

    def set(
        self,
        key: str | bytes,
        value: str | bytes,
        *,
        ex: int | None = None,
        px: int | None = None,
    ) -> AsyncPipeline:
        """Queue a SET command with optional TTL."""
        args: list[str | bytes] = ["SET", key, value]
        if ex is not None:
            args.extend(["EX", str(ex)])
        elif px is not None:
            args.extend(["PX", str(px)])
        self._commands.append(args)
        return self

    def mget(self, *keys: str | bytes) -> AsyncPipeline:
        """Queue an MGET command."""
        return self._queue("MGET", *keys)

    def delete(self, *keys: str | bytes) -> AsyncPipeline:
        """Queue a DEL command."""
        return self._queue("DEL", *keys)

    def exists(self, *keys: str | bytes) -> AsyncPipeline:
        """Queue an EXISTS command."""
        return self._queue("EXISTS", *keys)

    async def execute(self) -> list[RespValue]:
        """Send all queued commands and collect responses."""
        if not self._commands:
            return []

        packed = encode_commands(self._commands)
        count = len(self._commands)
        self._commands.clear()

        self._writer.write(packed)
        await self._writer.drain()

        results: list[RespValue] = []
        for _ in range(count):
            results.append(await self._resp_reader.read_response())
        return results

    def __len__(self) -> int:
        return len(self._commands)

    async def __aenter__(self) -> AsyncPipeline:
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._commands.clear()
