"""Unit tests for KacheDB SemanticCache and vector commands."""

from __future__ import annotations

import math
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from kachedb import AsyncSemanticCache, KacheClient, SearchResult, SemanticCache
from kachedb.semantic.embedders import (
    CallableAdapter,
    FastEmbedAdapter,
    MockEmbedder,
    OpenAIAdapter,
    SentenceTransformersAdapter,
    TransformersEmbedder,
)
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


class TestEmbedderAdapters:
    def test_transformers_embedder_mocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        fake_tok = MagicMock()
        mock_inputs = {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }

        class MockInputs(dict):  # type: ignore[type-arg]
            def to(self, _dev: str) -> MockInputs:
                return self

        fake_tok.return_value = MockInputs(mock_inputs)

        fake_model = MagicMock()
        fake_model.return_value = [torch.ones((1, 4, 16))]
        fake_model.to.return_value = fake_model

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = fake_tok
        mock_transformers.AutoModel.from_pretrained.return_value = fake_model
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        embedder = TransformersEmbedder(model_name="fake-model", device="cpu")
        vec = embedder.encode("hello world")
        assert len(vec) == 16
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-4

    def test_transformers_embedder_device_detection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        fake_tok = MagicMock()
        fake_model = MagicMock()
        fake_model.to.return_value = fake_model

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = fake_tok
        mock_transformers.AutoModel.from_pretrained.return_value = fake_model
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        # Test mps device branch
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        embedder = TransformersEmbedder()
        assert embedder.device == "mps"

        # Test cuda device branch
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        embedder2 = TransformersEmbedder()
        assert embedder2.device == "cuda"

        # Test cpu fallback branch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        embedder3 = TransformersEmbedder()
        assert embedder3.device == "cpu"

    def test_transformers_embedder_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(ImportError, match="transformers and torch are required"):
            TransformersEmbedder()

    def test_fastembed_adapter_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_fe = MagicMock()
        mock_inst = MagicMock()
        mock_inst.embed.return_value = iter([[0.1, 0.2, 0.3]])
        mock_fe.TextEmbedding.return_value = mock_inst
        monkeypatch.setitem(sys.modules, "fastembed", mock_fe)

        adapter = FastEmbedAdapter()
        vec = adapter.encode("query")
        assert vec == [0.1, 0.2, 0.3]

    def test_fastembed_adapter_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "fastembed", None)
        with pytest.raises(ImportError, match="FastEmbed is not installed"):
            FastEmbedAdapter()

    def test_sentence_transformers_adapter_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_st = MagicMock()
        mock_inst = MagicMock()
        mock_inst.encode.return_value = [0.4, 0.5, 0.6]
        mock_st.SentenceTransformer.return_value = mock_inst
        monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st)

        adapter = SentenceTransformersAdapter()
        vec = adapter.encode("query")
        assert vec == [0.4, 0.5, 0.6]

    def test_sentence_transformers_adapter_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(ImportError, match="SentenceTransformers is not installed"):
            SentenceTransformersAdapter()

    def test_openai_adapter_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_openai_mod = MagicMock()
        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = [0.7, 0.8, 0.9]
        mock_resp = MagicMock()
        mock_resp.data = [mock_data]
        mock_client.embeddings.create.return_value = mock_resp
        mock_openai_mod.OpenAI.return_value = mock_client
        monkeypatch.setitem(sys.modules, "openai", mock_openai_mod)

        adapter = OpenAIAdapter(api_key="fake-key")
        vec = adapter.encode("query")
        assert vec == [0.7, 0.8, 0.9]

    def test_openai_adapter_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(ImportError, match="OpenAI SDK is not installed"):
            OpenAIAdapter()


class TestAsyncSemanticCache:
    @pytest.mark.asyncio
    async def test_async_semantic_cache_operations(self) -> None:
        mock_client = AsyncMock()
        mock_client.vadd.return_value = True
        mock_client.vsearch.return_value = [(b"How to reset?", 0.96, b"Click reset link")]
        mock_client.vdel.return_value = True
        mock_client.vstats.return_value = {"total_vectors": 5}

        cache = AsyncSemanticCache(
            client=mock_client,
            index_name="faq_async",
            similarity_threshold=0.9,
            embedder=lambda _s: [0.1, 0.2],
        )
        assert isinstance(cache.embedder, CallableAdapter)

        # set
        ok = await cache.set("How to reset?", "Click reset link")
        assert ok is True

        # get hit
        res = await cache.get("How to reset?")
        assert res is not None
        assert res.value == "Click reset link"
        assert res.similarity == 0.96

        # get miss
        mock_client.vsearch.return_value = []
        miss = await cache.get("unknown")
        assert miss is None

        # delete
        deleted = await cache.delete("How to reset?")
        assert deleted is True

        # stats
        st = await cache.stats()
        assert st["total_vectors"] == 5

    def test_cache_auto_select_fallback(self) -> None:
        mock_client = MagicMock()
        # FastEmbed and SentenceTransformers missing -> falls back to MockEmbedder
        cache = SemanticCache(client=mock_client, index_name="test_auto")
        assert isinstance(cache.embedder, MockEmbedder)

        # Pass custom callable to sync SemanticCache
        sync_callable_cache = SemanticCache(
            client=mock_client, index_name="test_callable", embedder=lambda _s: [0.1]
        )
        assert isinstance(sync_callable_cache.embedder, CallableAdapter)

    def test_async_cache_init_branches(self) -> None:
        mock_client = AsyncMock()
        # None embedder -> auto-selects MockEmbedder
        async_cache_default = AsyncSemanticCache(client=mock_client, index_name="test_default")
        assert isinstance(async_cache_default.embedder, MockEmbedder)

        # explicit EmbeddingAdapter
        mock_embedder = MockEmbedder()
        async_cache_explicit = AsyncSemanticCache(
            client=mock_client, index_name="test_explicit", embedder=mock_embedder
        )
        assert async_cache_explicit.embedder is mock_embedder
