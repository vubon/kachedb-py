"""Unit tests for pipeline batching."""

from __future__ import annotations

import pytest

from kachedb import AsyncKacheClient, KacheClient
from tests.conftest import (
    MockKacheDBServer,
    resp_array,
    resp_bulk_string,
    resp_integer,
    resp_simple_string,
)


class TestPipeline:
    def test_pipeline_execute_multiple_commands(self, mock_server: MockKacheDBServer) -> None:
        # Program 3 responses: OK, OK, bulk string.
        mock_server.program_responses(
            resp_simple_string("OK") + resp_simple_string("OK") + resp_bulk_string(b"value_a")
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            pipe = client.pipeline()
            pipe.set("a", "value_a")
            pipe.set("b", "value_b")
            pipe.get("a")
            results = pipe.execute()

            assert len(results) == 3
            assert results[0] == "OK"
            assert results[1] == "OK"
            assert results[2] == b"value_a"

    def test_pipeline_empty_execute(self, mock_server: MockKacheDBServer) -> None:
        port = mock_server.start()

        with KacheClient(port=port) as client:
            pipe = client.pipeline()
            results = pipe.execute()
            assert results == []

    def test_pipeline_len(self, mock_server: MockKacheDBServer) -> None:
        port = mock_server.start()

        with KacheClient(port=port) as client:
            pipe = client.pipeline()
            assert len(pipe) == 0
            pipe.set("a", "1")
            pipe.get("a")
            assert len(pipe) == 2

    def test_pipeline_chaining(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("OK") + resp_bulk_string(b"1"))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            pipe = client.pipeline()
            pipe.set("x", "1").get("x")
            results = pipe.execute()
            assert len(results) == 2

    def test_pipeline_with_delete_and_exists(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(1) + resp_integer(0))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            pipe = client.pipeline()
            pipe.delete("key1")
            pipe.exists("key1")
            results = pipe.execute()
            assert results == [1, 0]

    def test_pipeline_extended_commands(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_simple_string("OK")
            + resp_integer(1)
            + resp_integer(5)
            + resp_integer(4)
            + resp_integer(1)
            + resp_integer(5)
            + resp_integer(5)
            + resp_integer(1)
            + resp_integer(60)
            + resp_integer(1)
            + resp_integer(1)
            + resp_integer(1)
            + resp_integer(60000)
            + resp_integer(1)
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            pipe = client.pipeline()
            pipe.mset({"k1": "v1", "k2": "v2"})
            pipe.incr("counter")
            pipe.incrby("counter", 4)
            pipe.decr("counter", 1)
            pipe.decrby("counter", 3)
            pipe.append("str", "hello")
            pipe.strlen("str")
            pipe.expire("temp", 60)
            pipe.ttl("temp")
            pipe.pexpire("temp", 60000)
            pipe.expireat("temp", 1893456000)
            pipe.pexpireat("temp", 1893456000000)
            pipe.pttl("temp")
            pipe.persist("temp")
            results = pipe.execute()
            assert len(results) == 14
            assert results[0] == "OK"
            assert results[1] == 1
            assert results[2] == 5
            assert results[3] == 4
            assert results[4] == 1
            assert results[5] == 5
            assert results[6] == 5
            assert results[7] == 1
            assert results[8] == 60
            assert results[9] == 1
            assert results[10] == 1
            assert results[11] == 1
            assert results[12] == 60000
            assert results[13] == 1

    def test_pipeline_admin_commands(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_integer(10)
            + resp_simple_string("string")
            + resp_simple_string("OK")
            + resp_simple_string("OK")
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            pipe = client.pipeline()
            pipe.dbsize().type("mykey").flushdb().flushall()
            results = pipe.execute()
            assert len(results) == 4
            assert results[0] == 10
            assert results[1] == "string"
            assert results[2] == "OK"
            assert results[3] == "OK"


class TestAsyncPipeline:
    @pytest.mark.asyncio
    async def test_async_pipeline_execute(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_simple_string("OK") + resp_bulk_string(b"bar") + resp_integer(1)
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            pipe = client.pipeline()
            pipe.set("foo", "bar").get("foo").delete("foo")
            assert len(pipe) == 3
            results = await pipe.execute()
            assert len(results) == 3
            assert results[0] == "OK"
            assert results[1] == b"bar"
            assert results[2] == 1
            assert len(pipe) == 0

    @pytest.mark.asyncio
    async def test_async_pipeline_empty(self, mock_server: MockKacheDBServer) -> None:
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            pipe = client.pipeline()
            assert len(pipe) == 0
            results = await pipe.execute()
            assert results == []

    @pytest.mark.asyncio
    async def test_async_pipeline_context_manager(self, mock_server: MockKacheDBServer) -> None:
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            async with client.pipeline() as pipe:
                pipe.ping()
                assert len(pipe) == 1
            # Exiting resets queued commands
            assert len(pipe) == 0

    @pytest.mark.asyncio
    async def test_async_pipeline_all_commands(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_simple_string("PONG")
            + resp_simple_string("OK")
            + resp_bulk_string(b"1")
            + resp_simple_string("OK")
            + resp_array(resp_bulk_string(b"1"))
            + resp_integer(1)
            + resp_integer(1)
            + resp_integer(2)
            + resp_integer(1)
            + resp_integer(1)
            + resp_integer(5)
            + resp_integer(5)
            + resp_integer(1)
            + resp_integer(1)
            + resp_integer(1)
            + resp_integer(1)
            + resp_integer(60)
            + resp_integer(60000)
            + resp_integer(1)
            + resp_integer(10)
            + resp_simple_string("string")
            + resp_simple_string("OK")
            + resp_simple_string("OK")
        )
        port = mock_server.start()

        async with AsyncKacheClient(port=port) as client:
            pipe = client.pipeline()
            pipe.ping()
            pipe.set("k", "v", ex=60)
            pipe.get("k")
            pipe.mset({"a": "1", "b": "2"})
            pipe.mget("a")
            pipe.exists("k")
            pipe.incr("c")
            pipe.incrby("c", 1)
            pipe.decr("c")
            pipe.decrby("c", 1)
            pipe.append("str", "abc")
            pipe.strlen("str")
            pipe.expire("k", 60)
            pipe.pexpire("k", 60000)
            pipe.expireat("k", 1893456000)
            pipe.pexpireat("k", 1893456000000)
            pipe.ttl("k")
            pipe.pttl("k")
            pipe.persist("k")
            pipe.dbsize()
            pipe.type("k")
            pipe.flushdb()
            pipe.flushall()

            assert len(pipe) == 23
            results = await pipe.execute()
            assert len(results) == 23
            assert results[0] == "PONG"
            assert results[1] == "OK"
            assert results[2] == b"1"
            assert results[3] == "OK"
            assert results[4] == [b"1"]
