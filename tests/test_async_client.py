"""Unit tests for the async KacheDB client."""

from __future__ import annotations

import pytest

from kachedb import AsyncKacheClient
from tests.conftest import (
    MockKacheDBServer,
    resp_array,
    resp_bulk_string,
    resp_integer,
    resp_null,
    resp_simple_string,
)


@pytest.mark.asyncio
async def test_async_client_mget_empty() -> None:
    """MGET with no keys should return empty list without connecting."""
    client = AsyncKacheClient(port=12345)
    result = await client.mget()
    assert result == []


@pytest.mark.asyncio
async def test_async_client_delete_empty() -> None:
    """DEL with no keys should return 0 without connecting."""
    client = AsyncKacheClient(port=12345)
    result = await client.delete()
    assert result == 0


@pytest.mark.asyncio
async def test_async_client_exists_empty() -> None:
    """EXISTS with no keys should return 0 without connecting."""
    client = AsyncKacheClient(port=12345)
    result = await client.exists()
    assert result == 0


class TestAsyncKacheClientKV:
    @pytest.mark.asyncio
    async def test_ping(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("PONG"), resp_simple_string("hello"))
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            assert await client.ping() == "PONG"
            assert await client.ping("hello") == "hello"

    @pytest.mark.asyncio
    async def test_get_set_delete(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_simple_string("OK"),
            resp_bulk_string(b"myval"),
            resp_integer(1),
            resp_integer(1),
            resp_null(),
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            assert await client.set("key", "myval", ex=60) is True
            assert await client.get("key") == b"myval"
            assert await client.exists("key") == 1
            assert await client.delete("key") == 1
            assert await client.get("key") is None

    @pytest.mark.asyncio
    async def test_set_with_px_and_decode_responses(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("OK"), resp_bulk_string(b"hello_str"))
        port = mock_server.start()

        async with AsyncKacheClient(port=port, decode_responses=True) as client:
            assert await client.set("key", "val", px=1000) is True
            assert await client.get("key") == "hello_str"

    @pytest.mark.asyncio
    async def test_mget_and_mset(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_simple_string("OK"),
            resp_array(resp_bulk_string(b"v1"), resp_bulk_string(b"v2")),
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            assert await client.mset({"k1": "v1", "k2": "v2"}) is True
            assert await client.mget("k1", "k2") == [b"v1", b"v2"]

    @pytest.mark.asyncio
    async def test_numeric_and_string_ops(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_integer(1),
            resp_integer(11),
            resp_integer(10),
            resp_integer(5),
            resp_integer(15),
            resp_integer(15),
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            assert await client.incr("counter") == 1
            assert await client.incrby("counter", 10) == 11
            assert await client.decr("counter") == 10
            assert await client.decrby("counter", 5) == 5
            assert await client.append("msg", "hello") == 15
            assert await client.strlen("msg") == 15


class TestAsyncKacheClientTTLAndAdmin:
    @pytest.mark.asyncio
    async def test_ttl_lifecycle(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_integer(1),
            resp_integer(1),
            resp_integer(1),
            resp_integer(1),
            resp_integer(60),
            resp_integer(60000),
            resp_integer(1),
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            assert await client.expire("k", 60) is True
            assert await client.pexpire("k", 60000) is True
            assert await client.expireat("k", 1893456000) is True
            assert await client.pexpireat("k", 1893456000000) is True
            assert await client.ttl("k") == 60
            assert await client.pttl("k") == 60000
            assert await client.persist("k") is True

    @pytest.mark.asyncio
    async def test_admin_commands(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_bulk_string(b"# Server\r\nkachedb_version:0.1.0\r\n"),
            resp_integer(100),
            resp_simple_string("string"),
            resp_simple_string("OK"),
            resp_simple_string("OK"),
            resp_simple_string("OK"),
            resp_simple_string("Background append only file rewriting started"),
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            assert "kachedb_version:0.1.0" in await client.info()
            assert await client.dbsize() == 100
            assert await client.type("k") == "string"
            assert await client.flushdb() is True
            assert await client.flushall() is True
            assert await client.auth("secret") is True
            assert "Background" in await client.bgrewriteaof()


class TestAsyncKacheClientVectorsAndIndices:
    @pytest.mark.asyncio
    async def test_vector_crud(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_integer(1),
            resp_array(
                resp_array(
                    resp_bulk_string(b"doc1"),
                    resp_bulk_string(b"0.92"),
                    resp_bulk_string(b"Payload 1"),
                )
            ),
            resp_integer(1),
            resp_array(
                resp_bulk_string(b"dimension"),
                resp_integer(128),
            ),
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            ok = await client.vadd("idx", "doc1", [1.0, 0.0], payload="Payload 1", ex=3600)
            assert ok is True
            results = await client.vsearch("idx", [1.0, 0.0], top_k=1, threshold=0.8)
            assert len(results) == 1
            assert results[0][0] == b"doc1"
            assert await client.vdel("idx", "doc1") is True
            stats = await client.vstats("idx")
            assert stats == {"dimension": 128}

    @pytest.mark.asyncio
    async def test_vindex_lifecycle(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_simple_string("OK"),
            resp_array(
                resp_bulk_string(b"dimension"),
                resp_integer(256),
                resp_bulk_string(b"metric"),
                resp_bulk_string(b"cosine"),
            ),
            resp_integer(1),
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            created = await client.vindex_create("vidx", 256, metric="COSINE", quantization="SQ8")
            assert created is True
            info = await client.vindex_info("vidx")
            assert info == {"dimension": 256, "metric": "cosine"}
            assert await client.vindex_drop("vidx") is True

    @pytest.mark.asyncio
    async def test_disconnect_all(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("PONG"))
        port = mock_server.start()

        client = AsyncKacheClient(port=port)
        await client.connect()
        assert await client.ping() == "PONG"
        await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_async_batch_vectors(self, mock_server: MockKacheDBServer) -> None:
        import struct

        mock_server.program_responses(
            resp_integer(2),
            resp_array(
                resp_array(
                    resp_array(
                        resp_bulk_string(b"doc1"),
                        resp_bulk_string(b"0.99"),
                        resp_bulk_string(b"res1"),
                    )
                )
            ),
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            # Empty items
            assert await client.vadd_batch("idx", []) == 0
            assert await client.vsearch_batch("idx", []) == []

            # Invalid vector type in vadd_batch
            with pytest.raises(TypeError, match="Unsupported vector type"):
                await client.vadd_batch("idx", [("id1", 12345, "payload")])  # type: ignore[list-item]

            # Invalid vector type in vsearch_batch
            with pytest.raises(TypeError, match="Unsupported vector type"):
                await client.vsearch_batch("idx", [12345])  # type: ignore[list-item]

            # Valid vadd_batch and vsearch_batch
            v_bytes = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
            items = [
                ("doc1", [0.1, 0.2, 0.3, 0.4], "res1"),
                ("doc2", v_bytes, None),
            ]
            added = await client.vadd_batch("idx", items, ex=60)
            assert added == 2

            search_res = await client.vsearch_batch("idx", [[0.1, 0.2, 0.3, 0.4], v_bytes])
            assert len(search_res) == 1
            assert search_res[0][0][0] == b"doc1"

    @pytest.mark.asyncio
    async def test_async_vector_types_and_errors(self, mock_server: MockKacheDBServer) -> None:
        import struct

        mock_server.program_responses(
            resp_integer(1),
            resp_array(
                resp_array(
                    resp_bulk_string(b"doc1"),
                    resp_bulk_string(b"0.95"),
                    resp_bulk_string(b"p1"),
                )
            ),
            resp_bulk_string(b"ERROR_NOT_A_LIST"),
            resp_bulk_string(b"ERROR_NOT_A_LIST"),
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            # Invalid type in vadd
            with pytest.raises(TypeError, match="Unsupported vector type"):
                await client.vadd("idx", "doc1", "invalid_type")  # type: ignore[arg-type]

            # Invalid type in vsearch
            with pytest.raises(TypeError, match="Unsupported vector type"):
                await client.vsearch("idx", "invalid_type")  # type: ignore[arg-type]

            # vadd with bytes
            raw_v = struct.pack("<2f", 1.0, 2.0)
            assert await client.vadd("idx", "doc1", raw_v) is True

            # vsearch with bytes
            res = await client.vsearch("idx", raw_v)
            assert len(res) == 1
            assert res[0][0] == b"doc1"

            # vstats with non-list response
            assert await client.vstats("idx") is None

            # vindex_info with non-list response
            assert await client.vindex_info("idx") is None

    @pytest.mark.asyncio
    async def test_async_auth_and_admin(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_simple_string("OK"),
            resp_simple_string("OK"),
            resp_bulk_string(b"Background append only file rewriting started"),
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            assert await client.auth("secret") is True
            assert await client.auth("secret", username="user1") is True
            bg = await client.bgrewriteaof()
            assert "rewriting started" in bg
