"""
Tests for VADD_BATCH and VSEARCH_BATCH in KacheClient and AsyncKacheClient.
"""

from __future__ import annotations

import pytest

from kachedb import AsyncKacheClient, KacheClient
from tests.conftest import (
    MockKacheDBServer,
    resp_array,
    resp_bulk_string,
    resp_integer,
)


def test_batch_empty_calls() -> None:
    """Empty batch calls should return immediately without attempting connection."""
    client = KacheClient(port=12345)
    assert client.vadd_batch("idx", []) == 0
    assert client.vsearch_batch("idx", []) == []


@pytest.mark.asyncio
async def test_async_batch_empty_calls() -> None:
    """Empty async batch calls should return immediately without attempting connection."""
    client = AsyncKacheClient(port=12345)
    assert await client.vadd_batch("idx", []) == 0
    assert await client.vsearch_batch("idx", []) == []


def test_vadd_batch_wire(mock_server: MockKacheDBServer) -> None:
    """Verify VADD_BATCH wire command and integer response parsing."""
    mock_server.program_responses(resp_integer(3))
    port = mock_server.start()

    with KacheClient(port=port) as client:
        added = client.vadd_batch(
            "test:idx",
            [
                ("item1", [1.0, 0.0, 0.0], "Payload 1"),
                ("item2", [0.0, 1.0, 0.0], "Payload 2"),
                ("item3", [0.0, 0.0, 1.0], "Payload 3"),
            ],
        )
        assert added == 3


def test_vsearch_batch_wire(mock_server: MockKacheDBServer) -> None:
    """Verify VSEARCH_BATCH wire command and nested array response parsing."""
    mock_server.program_responses(
        resp_array(
            resp_array(
                resp_array(
                    resp_bulk_string(b"item1"),
                    resp_bulk_string(b"0.990000"),
                    resp_bulk_string(b"Payload 1"),
                )
            ),
            resp_array(
                resp_array(
                    resp_bulk_string(b"item2"),
                    resp_bulk_string(b"0.980000"),
                    resp_bulk_string(b"Payload 2"),
                )
            ),
        )
    )
    port = mock_server.start()

    with KacheClient(port=port) as client:
        results = client.vsearch_batch(
            "test:idx", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], top_k=1, threshold=0.8
        )
        assert len(results) == 2
        assert results[0][0][0] == b"item1"
        assert results[0][0][2] == b"Payload 1"
        assert results[1][0][0] == b"item2"
        assert results[1][0][2] == b"Payload 2"


@pytest.mark.integration
def test_sync_batch_vector_crud_live() -> None:
    """Live integration test requiring a running KacheDB server on 127.0.0.1:6379."""
    with KacheClient() as client:
        index = "test:sync:batch"
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0]

        items = [
            ("item1", v1, "Payload 1"),
            ("item2", v2, "Payload 2"),
        ]

        added = client.vadd_batch(index, items)
        assert added == 2

        queries = [v1, v2]
        batch_results = client.vsearch_batch(index, queries, top_k=1, threshold=0.8)
        assert len(batch_results) == 2
        assert batch_results[0][0][0] in ("item1", b"item1")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_batch_vector_crud_live() -> None:
    """Live async integration test requiring a running KacheDB server on 127.0.0.1:6379."""
    async with AsyncKacheClient() as async_client:
        index = "test:async:batch"
        v1 = [1.0, 0.0, 0.0, 0.0]
        items = [("a1", v1, "Async Payload 1")]

        added = await async_client.vadd_batch(index, items)
        assert added == 1

        queries = [v1]
        batch_results = await async_client.vsearch_batch(index, queries, top_k=1, threshold=0.8)
        assert len(batch_results) == 1
        assert batch_results[0][0][0] in ("a1", b"a1")
