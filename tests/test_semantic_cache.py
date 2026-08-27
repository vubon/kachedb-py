"""Unit tests for KacheDB SemanticCache and vector commands."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

from kachedb import KacheClient, SearchResult, SemanticCache
from kachedb.semantic.embedders import CallableAdapter, MockEmbedder
from tests.conftest import (
    MockKacheDBServer,
    resp_array,
    resp_bulk_string,
    resp_integer,
)


class TestMockEmbedder:
    def test_dimension_and_normalization(self) -> None:
        embedder = MockEmbedder(dimension=128)
        vec = embedder.encode("How do I reset my password?")
        assert len(vec) == 128
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5

    def test_deterministic_vectors(self) -> None:
        embedder = MockEmbedder(dimension=64)
        v1 = embedder.encode("hello world")
        v2 = embedder.encode("hello world")
        assert v1 == v2

    def test_empty_string_vector(self) -> None:
        embedder = MockEmbedder(dimension=32)
        v = embedder.encode("")
        assert len(v) == 32
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-5


class TestCallableAdapter:
    def test_custom_callable(self) -> None:
        adapter = CallableAdapter(lambda s: [1.0, 0.0, 0.0])
        vec = adapter.encode("any text")
        assert vec == [1.0, 0.0, 0.0]


class TestSemanticCacheWithMockClient:
    def test_set_and_get_hit(self) -> None:
        mock_client = MagicMock(spec=KacheClient)
        mock_client.vadd.return_value = True
        mock_client.vsearch.return_value = [
            (b"How do I reset password?", 0.94, b"Go to Settings -> Reset Password")
        ]

        cache = SemanticCache(
            client=mock_client,
            index_name="faq",
            similarity_threshold=0.85,
            embedder=MockEmbedder(dimension=64),
        )

        # Set
        stored = cache.set("How do I reset password?", "Go to Settings -> Reset Password")
        assert stored is True
        assert mock_client.vadd.called

        # Get
        result = cache.get("Where to change my password?")
        assert result is not None
        assert isinstance(result, SearchResult)
        assert result.value == "Go to Settings -> Reset Password"
        assert result.similarity == 0.94
        assert str(result) == "Go to Settings -> Reset Password"

    def test_get_miss(self) -> None:
        mock_client = MagicMock(spec=KacheClient)
        mock_client.vsearch.return_value = []

        cache = SemanticCache(
            client=mock_client,
            index_name="faq",
            similarity_threshold=0.85,
            embedder=MockEmbedder(dimension=64),
        )

        result = cache.get("Completely unrelated query about cooking recipes")
        assert result is None

    def test_delete_and_stats(self) -> None:
        mock_client = MagicMock(spec=KacheClient)
        mock_client.vdel.return_value = True
        mock_client.vstats.return_value = {"total_vectors": 10, "dimension": 64}

        cache = SemanticCache(
            client=mock_client,
            index_name="faq",
            embedder=MockEmbedder(dimension=64),
        )

        assert cache.delete("old prompt") is True
        stats = cache.stats()
        assert stats["total_vectors"] == 10


class TestKacheClientVectorCommands:
    def test_vadd_wire(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(1))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            ok = client.vadd("faq", "doc1", [1.0, 0.0, 0.0], payload="Answer 1", ex=3600)
            assert ok is True

    def test_vsearch_wire(self, mock_server: MockKacheDBServer) -> None:
        # RESP array of array: [ [ "doc1", "0.950000", "Answer 1" ] ]
        mock_server.program_responses(
            resp_array(
                resp_array(
                    resp_bulk_string(b"doc1"),
                    resp_bulk_string(b"0.950000"),
                    resp_bulk_string(b"Answer 1"),
                )
            )
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            matches = client.vsearch("faq", [1.0, 0.0, 0.0], top_k=1, threshold=0.8)
            assert len(matches) == 1
            item_id, sim, payload = matches[0]
            assert item_id == b"doc1"
            assert abs(sim - 0.95) < 1e-4
            assert payload == b"Answer 1"

    def test_vdel_and_vstats_wire(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_integer(1),
            resp_array(
                resp_bulk_string(b"dimension"),
                resp_integer(384),
                resp_bulk_string(b"total_vectors"),
                resp_integer(42),
            ),
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.vdel("faq", "doc1") is True
            stats = client.vstats("faq")
            assert stats is not None
            assert stats["dimension"] == 384
            assert stats["total_vectors"] == 42
