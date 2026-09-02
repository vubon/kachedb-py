"""
Tests for VADD_BATCH and VSEARCH_BATCH in KacheClient and AsyncKacheClient.
"""

from __future__ import annotations

import pytest

from kachedb import AsyncKacheClient, KacheClient


@pytest.fixture
def client() -> KacheClient:
    return KacheClient()


def test_sync_batch_vector_crud(client: KacheClient) -> None:
    index = "test:sync:batch"
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0, 0.0]
    v3 = [0.0, 0.0, 1.0, 0.0]

    items = [
        ("item1", v1, "Payload 1"),
        ("item2", v2, "Payload 2"),
        ("item3", v3, "Payload 3"),
    ]

    # Add batch
    added = client.vadd_batch(index, items)
    assert added == 3

    # Search batch
    queries = [v1, v2]
    batch_results = client.vsearch_batch(index, queries, top_k=1, threshold=0.8)
    assert len(batch_results) == 2

    # Query 1 match
    assert len(batch_results[0]) == 1
    assert batch_results[0][0][0] in ("item1", b"item1")
    assert batch_results[0][0][2] in ("Payload 1", b"Payload 1")

    # Query 2 match
    assert len(batch_results[1]) == 1
    assert batch_results[1][0][0] in ("item2", b"item2")
    assert batch_results[1][0][2] in ("Payload 2", b"Payload 2")


@pytest.mark.asyncio
async def test_async_batch_vector_crud() -> None:
    async with AsyncKacheClient() as async_client:
        index = "test:async:batch"
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0]

        items = [
            ("a1", v1, "Async Payload 1"),
            ("a2", v2, "Async Payload 2"),
        ]

        added = await async_client.vadd_batch(index, items)
        assert added == 2

        queries = [v1]
        batch_results = await async_client.vsearch_batch(index, queries, top_k=1, threshold=0.8)
        assert len(batch_results) == 1
        assert len(batch_results[0]) == 1
        assert batch_results[0][0][0] in ("a1", b"a1")
