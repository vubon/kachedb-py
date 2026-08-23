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
