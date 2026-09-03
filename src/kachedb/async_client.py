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

    async def mset(self, mapping: dict[str | bytes, str | bytes]) -> bool:
        """Set multiple keys to their respective values atomically."""
        if not mapping:
            return True
        args: list[str | bytes] = ["MSET"]
        for k, v in mapping.items():
            args.extend([k, v])
        result = await self._execute(*args)
        return result == "OK"

    async def incr(self, key: str | bytes, amount: int = 1) -> int:
        """Increment the integer value of *key* by *amount* (default 1)."""
        if amount == 1:
            result = await self._execute("INCR", key)
        else:
            result = await self._execute("INCRBY", key, str(amount))
        return int(result) if isinstance(result, int) else 0

    async def incrby(self, key: str | bytes, amount: int) -> int:
        """Increment the integer value of *key* by *amount*."""
        return await self.incr(key, amount)

    async def decr(self, key: str | bytes, amount: int = 1) -> int:
        """Decrement the integer value of *key* by *amount* (default 1)."""
        if amount == 1:
            result = await self._execute("DECR", key)
        else:
            result = await self._execute("DECRBY", key, str(amount))
        return int(result) if isinstance(result, int) else 0

    async def decrby(self, key: str | bytes, amount: int) -> int:
        """Decrement the integer value of *key* by *amount*."""
        return await self.decr(key, amount)

    async def append(self, key: str | bytes, value: str | bytes) -> int:
        """Append *value* to *key*. Returns the new byte length of the value."""
        result = await self._execute("APPEND", key, value)
        return int(result) if isinstance(result, int) else 0

    async def strlen(self, key: str | bytes) -> int:
        """Return the byte length of the value stored at *key*."""
        result = await self._execute("STRLEN", key)
        return int(result) if isinstance(result, int) else 0

    async def expire(self, key: str | bytes, seconds: int) -> bool:
        """Set a timeout on *key* in seconds."""
        result = await self._execute("EXPIRE", key, str(seconds))
        return result == 1

    async def pexpire(self, key: str | bytes, milliseconds: int) -> bool:
        """Set a timeout on *key* in milliseconds."""
        result = await self._execute("PEXPIRE", key, str(milliseconds))
        return result == 1

    async def expireat(self, key: str | bytes, timestamp: int) -> bool:
        """Set an expiration deadline on *key* as a Unix timestamp (seconds)."""
        result = await self._execute("EXPIREAT", key, str(timestamp))
        return result == 1

    async def pexpireat(self, key: str | bytes, timestamp_ms: int) -> bool:
        """Set an expiration deadline on *key* as a Unix timestamp (milliseconds)."""
        result = await self._execute("PEXPIREAT", key, str(timestamp_ms))
        return result == 1

    async def ttl(self, key: str | bytes) -> int:
        """Return remaining time-to-live in seconds (-1 if no TTL, -2 if missing)."""
        result = await self._execute("TTL", key)
        return int(result) if isinstance(result, int) else -2

    async def pttl(self, key: str | bytes) -> int:
        """Return remaining time-to-live in milliseconds (-1 if no TTL, -2 if missing)."""
        result = await self._execute("PTTL", key)
        return int(result) if isinstance(result, int) else -2

    async def persist(self, key: str | bytes) -> bool:
        """Remove the existing timeout on *key*, persisting it indefinitely."""
        result = await self._execute("PERSIST", key)
        return result == 1

    async def info(self, section: str | None = None) -> str:
        """Return server information and runtime statistics."""
        args: list[str | bytes] = ["INFO"]
        if section is not None:
            args.append(section)
        result = await self._execute(*args)
        if isinstance(result, bytes):
            return result.decode("utf-8", errors="replace")
        return str(result) if result is not None else ""

    # ── Pipeline ──────────────────────────────────────────────────────────

    def pipeline(self) -> AsyncPipeline:
        """Create an async pipeline for batching commands."""
        if self._writer is None or self._resp_reader is None:
            raise ConnectionError("Not connected to KacheDB")
        return AsyncPipeline(self._writer, self._resp_reader)

    # ── Vector Search & Semantic Cache Commands ───────────────────────────

    async def vadd(
        self,
        index: str | bytes,
        item_id: str | bytes,
        vector: bytes | list[float] | tuple[float, ...],
        *,
        payload: str | bytes | None = None,
        ex: int | None = None,
    ) -> bool:
        """Store a vector embedding in a named vector index asynchronously."""
        import struct

        if isinstance(vector, (list, tuple)):
            dim = len(vector)
            vector_bytes = struct.pack(f"<{dim}f", *vector)
        elif isinstance(vector, (bytes, bytearray)):
            vector_bytes = bytes(vector)
            dim = len(vector_bytes) // 4
        else:
            raise TypeError(f"Unsupported vector type: {type(vector)}")

        args: list[str | bytes] = ["VADD", index, item_id, str(dim), vector_bytes]
        if payload is not None:
            args.extend(["PAYLOAD", payload])
        if ex is not None:
            args.extend(["EX", str(ex)])

        result = await self._execute(*args)
        return result == 1 or result == "OK"

    async def vsearch(
        self,
        index: str | bytes,
        query_vector: bytes | list[float] | tuple[float, ...],
        *,
        top_k: int = 1,
        threshold: float = 0.0,
    ) -> list[tuple[str | bytes, float, str | bytes | None]]:
        """Search for nearest semantic vectors in a named index asynchronously."""
        import struct

        if isinstance(query_vector, (list, tuple)):
            query_bytes = struct.pack(f"<{len(query_vector)}f", *query_vector)
        elif isinstance(query_vector, (bytes, bytearray)):
            query_bytes = bytes(query_vector)
        else:
            raise TypeError(f"Unsupported vector type: {type(query_vector)}")

        args: list[str | bytes] = [
            "VSEARCH",
            index,
            query_bytes,
            "TOPK",
            str(top_k),
            "THRESHOLD",
            str(threshold),
        ]
        raw_results = await self._execute(*args)
        if not isinstance(raw_results, list):
            return []

        results: list[tuple[str | bytes, float, str | bytes | None]] = []
        for item in raw_results:
            if isinstance(item, list) and len(item) >= 2:
                raw_id = item[0]
                item_id: str | bytes = raw_id if isinstance(raw_id, (str, bytes)) else str(raw_id)
                raw_score = item[1]
                try:
                    if isinstance(raw_score, bytes):
                        score = float(raw_score.decode())
                    elif isinstance(raw_score, (int, float, str)):
                        score = float(raw_score)
                    else:
                        score = 0.0
                except Exception:
                    score = 0.0
                raw_payload = item[2] if len(item) > 2 else None
                payload: str | bytes | None = (
                    raw_payload
                    if isinstance(raw_payload, (str, bytes)) or raw_payload is None
                    else str(raw_payload)
                )
                results.append((item_id, score, payload))
        return results

    async def vdel(self, index: str | bytes, item_id: str | bytes) -> bool:
        """Delete a vector from a named index asynchronously."""
        result = await self._execute("VDEL", index, item_id)
        return result == 1

    async def vadd_batch(
        self,
        index: str | bytes,
        items: list[
            tuple[str | bytes, bytes | list[float] | tuple[float, ...], str | bytes | None]
        ],
        *,
        ex: int | None = None,
    ) -> int:
        """Add multiple vector items in a single batch command asynchronously."""
        if not items:
            return 0

        import struct

        args: list[str | bytes] = ["VADD_BATCH", index]
        for item_id, vector, payload in items:
            if isinstance(vector, (list, tuple)):
                dim = len(vector)
                vector_bytes = struct.pack(f"<{dim}f", *vector)
            elif isinstance(vector, (bytes, bytearray)):
                vector_bytes = bytes(vector)
            else:
                raise TypeError(f"Unsupported vector type: {type(vector)}")

            p_val = payload if payload is not None else "-"
            args.extend([item_id, vector_bytes, p_val])
            if ex is not None:
                args.extend(["EX", str(ex)])

        result = await self._execute(*args)
        return int(result) if isinstance(result, int) else 0

    async def vsearch_batch(
        self,
        index: str | bytes,
        query_vectors: list[bytes | list[float] | tuple[float, ...]],
        *,
        top_k: int = 1,
        threshold: float = 0.0,
    ) -> list[list[tuple[str | bytes, float, str | bytes | None]]]:
        """Search multiple query vectors in a single batch command asynchronously."""
        if not query_vectors:
            return []

        import struct

        args: list[str | bytes] = ["VSEARCH_BATCH", index]
        for q in query_vectors:
            if isinstance(q, (list, tuple)):
                q_bytes = struct.pack(f"<{len(q)}f", *q)
            elif isinstance(q, (bytes, bytearray)):
                q_bytes = bytes(q)
            else:
                raise TypeError(f"Unsupported vector type: {type(q)}")
            args.append(q_bytes)

        args.extend(["TOPK", str(top_k), "THRESHOLD", str(threshold)])
        raw_batch = await self._execute(*args)
        if not isinstance(raw_batch, list):
            return []

        all_results: list[list[tuple[str | bytes, float, str | bytes | None]]] = []
        for raw_results in raw_batch:
            if not isinstance(raw_results, list):
                all_results.append([])
                continue
            query_res: list[tuple[str | bytes, float, str | bytes | None]] = []
            for item in raw_results:
                if isinstance(item, list) and len(item) >= 2:
                    raw_id = item[0]
                    item_id = raw_id if isinstance(raw_id, (str, bytes)) else str(raw_id)
                    raw_score = item[1]
                    try:
                        score = float(
                            raw_score.decode()
                            if isinstance(raw_score, bytes)
                            else float(str(raw_score))
                        )
                    except Exception:
                        score = 0.0
                    raw_payload = item[2] if len(item) > 2 else None
                    payload = (
                        raw_payload
                        if isinstance(raw_payload, (str, bytes)) or raw_payload is None
                        else str(raw_payload)
                    )
                    query_res.append((item_id, score, payload))
            all_results.append(query_res)
        return all_results

    async def vstats(self, index: str | bytes) -> dict[str, Any] | None:
        """Get statistics for a named vector index asynchronously."""
        raw = await self._execute("VSTATS", index)
        if not isinstance(raw, list):
            return None
        stats: dict[str, Any] = {}
        for i in range(0, len(raw) - 1, 2):
            raw_k = raw[i]
            k = raw_k.decode() if isinstance(raw_k, bytes) else str(raw_k)
            v = raw[i + 1]
            stats[k] = v
        return stats
