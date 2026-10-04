"""
Unit and integration tests for Redis Hash commands:
HSET, HGET, HDEL, HEXISTS, HLEN, HGETALL across KacheClient, AsyncKacheClient,
Pipeline, and AsyncPipeline.
"""

from __future__ import annotations

import os

import pytest

from kachedb import AsyncKacheClient, KacheClient, ResponseError
from tests.conftest import (
    MockKacheDBServer,
    resp_array,
    resp_bulk_string,
    resp_integer,
    resp_null,
)


def _detect_port() -> int:
    if "KACHEDB_PORT" in os.environ:
        return int(os.environ["KACHEDB_PORT"])
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        if s.connect_ex(("127.0.0.1", 6380)) == 0:
            return 6380
    return 6379


TEST_PORT = _detect_port()


class TestSyncHashCommands:
    """Unit tests for synchronous Hash operations using MockKacheDBServer."""

    def test_hset_single_and_mapping(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(1), resp_integer(2))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            # Single field-value
            res1 = client.hset("user:100", "name", "Alice")
            assert res1 == 1

            # Mapping
            res2 = client.hset("user:100", mapping={"role": "admin", "active": "1"})
            assert res2 == 2

    def test_hset_invalid_arguments(self) -> None:
        client = KacheClient(port=12345)
        with pytest.raises(ValueError, match="hset\\(\\) requires"):
            client.hset("user:100")
        with pytest.raises(ValueError, match="hset\\(\\) requires"):
            client.hset("user:100", "name")  # missing value

    def test_hget_found_and_nil(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_bulk_string(b"Alice"), resp_null())
        port = mock_server.start()

        with KacheClient(port=port) as client:
            val = client.hget("user:100", "name")
            assert val == b"Alice"

            missing = client.hget("user:100", "ghost")
            assert missing is None

    def test_hdel(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(2))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.hdel("user:100", "role", "active") == 2

            with pytest.raises(ValueError, match="hdel\\(\\) requires"):
                client.hdel("user:100")

    def test_hexists(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(1), resp_integer(0))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.hexists("user:100", "name") is True
            assert client.hexists("user:100", "ghost") is False

    def test_hlen(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(3))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.hlen("user:100") == 3

    def test_hgetall_populated_and_empty(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_array(
                resp_bulk_string(b"name"),
                resp_bulk_string(b"Alice"),
                resp_bulk_string(b"role"),
                resp_bulk_string(b"admin"),
            ),
            resp_array(),  # empty array for missing key
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            res = client.hgetall("user:100")
            assert res == {b"name": b"Alice", b"role": b"admin"}

            empty = client.hgetall("missing")
            assert empty == {}

    def test_hash_decode_responses(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_bulk_string(b"Alice"),
            resp_array(resp_bulk_string(b"name"), resp_bulk_string(b"Alice")),
        )
        port = mock_server.start()

        with KacheClient(port=port, decode_responses=True) as client:
            val = client.hget("user:100", "name")
            assert val == "Alice"

            all_pairs = client.hgetall("user:100")
            assert all_pairs == {"name": "Alice"}


class TestAsyncHashCommands:
    """Unit tests for asynchronous Hash operations using MockKacheDBServer."""

    @pytest.mark.asyncio
    async def test_async_hset_and_hget(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_integer(1),
            resp_integer(2),
            resp_bulk_string(b"Bob"),
            resp_null(),
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            assert await client.hset("user:200", "name", "Bob") == 1
            assert await client.hset("user:200", mapping={"a": "1", "b": "2"}) == 2

            assert await client.hget("user:200", "name") == b"Bob"
            assert await client.hget("user:200", "ghost") is None

    @pytest.mark.asyncio
    async def test_async_hset_invalid_arguments(self) -> None:
        client = AsyncKacheClient(port=12345)
        with pytest.raises(ValueError, match="hset\\(\\) requires"):
            await client.hset("user:200")
        with pytest.raises(ValueError, match="hdel\\(\\) requires"):
            await client.hdel("user:200")

    @pytest.mark.asyncio
    async def test_async_hdel_hexists_hlen(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_integer(2),
            resp_integer(1),
            resp_integer(0),
            resp_integer(5),
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            assert await client.hdel("user:200", "f1", "f2") == 2
            assert await client.hexists("user:200", "f1") is True
            assert await client.hexists("user:200", "ghost") is False
            assert await client.hlen("user:200") == 5

    @pytest.mark.asyncio
    async def test_async_hgetall(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_array(
                resp_bulk_string(b"k1"),
                resp_bulk_string(b"v1"),
            ),
            resp_array(),
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            data = await client.hgetall("user:200")
            assert data == {b"k1": b"v1"}

            empty = await client.hgetall("ghost")
            assert empty == {}


class TestPipelineHashCommands:
    """Unit tests for Hash operations within sync and async pipelines."""

    def test_sync_pipeline_hash_commands(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_integer(2)
            + resp_bulk_string(b"val1")
            + resp_integer(1)
            + resp_integer(2)
            + resp_array(resp_bulk_string(b"f1"), resp_bulk_string(b"val1"))
            + resp_integer(1)
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            pipe = client.pipeline()
            pipe.hset("hkey", mapping={"f1": "val1", "f2": "val2"})
            pipe.hget("hkey", "f1")
            pipe.hexists("hkey", "f1")
            pipe.hlen("hkey")
            pipe.hgetall("hkey")
            pipe.hdel("hkey", "f2")
            assert len(pipe) == 6

            results = pipe.execute()
            assert len(results) == 6
            assert results[0] == 2
            assert results[1] == b"val1"
            assert results[2] == 1
            assert results[3] == 2
            assert results[4] == [b"f1", b"val1"]
            assert results[5] == 1

    @pytest.mark.asyncio
    async def test_async_pipeline_hash_commands(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(1) + resp_bulk_string(b"v1") + resp_integer(1))
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client, client.pipeline() as pipe:
            pipe.hset("hkey", "f1", "v1")
            pipe.hget("hkey", "f1")
            pipe.hdel("hkey", "f1")
            assert len(pipe) == 3

            results = await pipe.execute()
            assert results == [1, b"v1", 1]


@pytest.mark.integration
class TestLiveHashIntegration:
    """Live integration tests requiring a running KacheDB server with Hash support."""

    def test_live_sync_hash_lifecycle(self) -> None:
        with KacheClient(port=TEST_PORT) as client:
            key = "test:live:hash:sync"
            client.delete(key)

            # 1. HSET single and mapping
            assert client.hset(key, "name", "Alice") == 1
            assert client.hset(key, mapping={"role": "admin", "active": "1"}) == 2

            # 2. HGET
            assert client.hget(key, "name") == b"Alice"
            assert client.hget(key, "role") == b"admin"
            assert client.hget(key, "ghost") is None

            # 3. HEXISTS & HLEN
            assert client.hexists(key, "name") is True
            assert client.hexists(key, "ghost") is False
            assert client.hlen(key) == 3

            # 4. HGETALL
            all_entries = client.hgetall(key)
            assert all_entries[b"name"] == b"Alice"
            assert all_entries[b"role"] == b"admin"
            assert all_entries[b"active"] == b"1"

            # 5. Type guard: string command against hash key raises ResponseError
            with pytest.raises(ResponseError, match="WRONGTYPE"):
                client.get(key)

            # 6. HDEL
            assert client.hdel(key, "role", "active") == 2
            assert client.hlen(key) == 1
            assert client.hexists(key, "role") is False

            # Cleanup
            client.delete(key)

    @pytest.mark.asyncio
    async def test_live_async_hash_lifecycle(self) -> None:
        async with AsyncKacheClient(port=TEST_PORT) as client:
            key = "test:live:hash:async"
            await client.delete(key)

            assert await client.hset(key, "city", "Tokyo") == 1
            assert await client.hget(key, "city") == b"Tokyo"
            assert await client.hexists(key, "city") is True
            assert await client.hlen(key) == 1

            data = await client.hgetall(key)
            assert data == {b"city": b"Tokyo"}

            assert await client.hdel(key, "city") == 1
            assert await client.hlen(key) == 0

            await client.delete(key)
