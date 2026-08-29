"""Unit tests for pipeline batching."""

from __future__ import annotations

from kachedb import KacheClient
from tests.conftest import (
    MockKacheDBServer,
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
