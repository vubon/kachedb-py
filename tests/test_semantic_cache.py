"""Unit tests for KacheDB SemanticCache and vector commands."""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from kachedb import (
    AsyncSemanticCache,
    EmbeddingGemmaAdapter,
    EmbeddingTask,
    KacheClient,
    SearchResult,
    SemanticCache,
)
from kachedb.semantic.embedders import (
    DEFAULT_MAX_TOKENS,
    PREFIX_DOCUMENT_INDEXING,
    PREFIX_SEARCH_QUERY,
    CallableAdapter,
    FastEmbedAdapter,
    MockEmbedder,
    ModelResolutionConfig,
    ModelSourceMode,
    OpenAIAdapter,
    SentenceTransformersAdapter,
    TransformersEmbedder,
    format_task_prompt,
    l2_normalize,
    resolve_model_paths,
    slice_and_normalize,
)

if TYPE_CHECKING:
    from pathlib import Path
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
        torch = pytest.importorskip("torch")

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
        torch = pytest.importorskip("torch")

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

    def test_cache_auto_select_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "fastembed", None)
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)
        monkeypatch.delenv("KACHEDB_EMBEDDER", raising=False)
        mock_client = MagicMock()
        # FastEmbed and SentenceTransformers missing -> falls back to MockEmbedder
        cache = SemanticCache(client=mock_client, index_name="test_auto")
        assert isinstance(cache.embedder, MockEmbedder)

        # Pass custom callable to sync SemanticCache
        sync_callable_cache = SemanticCache(
            client=mock_client, index_name="test_callable", embedder=lambda _s: [0.1]
        )
        assert isinstance(sync_callable_cache.embedder, CallableAdapter)

    def test_async_cache_init_branches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "fastembed", None)
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)
        monkeypatch.delenv("KACHEDB_EMBEDDER", raising=False)
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


class TestEmbeddingGemmaUnit1:
    """Comprehensive unit tests for EmbeddingGemma 2 MRL, task prefixes, resolver, and adapter."""

    def test_task_prompt_formatting_and_idempotence(self) -> None:
        q = format_task_prompt(EmbeddingTask.SEARCH_QUERY, "What is an LLM?")
        assert q == f"{PREFIX_SEARCH_QUERY}What is an LLM?"
        # Idempotent if called again
        assert format_task_prompt(EmbeddingTask.SEARCH_QUERY, q) == q
        assert format_task_prompt(EmbeddingTask.DOCUMENT_INDEXING, q) == q

        d = format_task_prompt(EmbeddingTask.DOCUMENT_INDEXING, "LLMs are neural networks.")
        assert d == f"{PREFIX_DOCUMENT_INDEXING}LLMs are neural networks."
        assert format_task_prompt(EmbeddingTask.DOCUMENT_INDEXING, d) == d

    def test_mrl_slice_and_normalize_384_and_768(self) -> None:
        raw_768 = [float(i + 1) for i in range(768)]

        sliced_384 = slice_and_normalize(raw_768, 384)
        assert len(sliced_384) == 384
        norm_384 = math.sqrt(sum(x * x for x in sliced_384))
        assert abs(norm_384 - 1.0) < 1e-6

        full_768 = slice_and_normalize(raw_768, 768)
        assert len(full_768) == 768
        norm_768 = math.sqrt(sum(x * x for x in full_768))
        assert abs(norm_768 - 1.0) < 1e-6

    def test_mrl_slice_and_normalize_rejects_invalid_inputs(self) -> None:
        raw_768 = [1.0] * 768
        for invalid_dim in (0, 64, 128, 256, 512, 1024):
            with pytest.raises(ValueError, match="Unsupported MRL target dimension"):
                slice_and_normalize(raw_768, invalid_dim)

        with pytest.raises(ValueError, match="smaller than target dimension"):
            slice_and_normalize([1.0] * 128, 384)

        with pytest.raises(ValueError, match="Cannot normalize an empty vector"):
            l2_normalize([])

        with pytest.raises(ValueError, match="zero or sub-epsilon"):
            slice_and_normalize([0.0] * 768, 384)

        with pytest.raises(ValueError, match="NaN or infinite"):
            l2_normalize([float("nan")] * 384)

    def test_model_path_resolver_precedence(self, tmp_path: Path) -> None:
        def _populate_dir(d: Path) -> Path:
            d.mkdir(parents=True, exist_ok=True)
            (d / "model.onnx").write_bytes(b"fake-onnx")
            (d / "tokenizer.json").write_text("{}", encoding="utf-8")
            return d

        cli_dir = _populate_dir(tmp_path / "cli_models")
        env_dir = _populate_dir(tmp_path / "env_models")
        cache_dir = _populate_dir(tmp_path / "cache_models")

        # 1. Explicit cli_path wins over env_path and cache_dir
        cfg1 = ModelResolutionConfig(cli_path=cli_dir, env_path=env_dir, cache_dir=cache_dir)
        res1 = resolve_model_paths(cfg1)
        assert res1.source is ModelSourceMode.EXPLICIT_PATH
        assert res1.root_dir == cli_dir

        # 2. env_path wins over cache_dir when cli_path is None
        cfg2 = ModelResolutionConfig(cli_path=None, env_path=env_dir, cache_dir=cache_dir)
        res2 = resolve_model_paths(cfg2)
        assert res2.source is ModelSourceMode.ENV_VAR
        assert res2.root_dir == env_dir

        # 3. cache_dir fallback when cli_path and env_path are None
        cfg3 = ModelResolutionConfig(cli_path=None, env_path=None, cache_dir=cache_dir)
        res3 = resolve_model_paths(cfg3)
        assert res3.source is ModelSourceMode.HUGGINGFACE_CACHE
        assert res3.root_dir == cache_dir

    def test_model_path_resolver_errors(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_model_paths(ModelResolutionConfig(cli_path=missing_dir))

        no_onnx_dir = tmp_path / "no_onnx"
        no_onnx_dir.mkdir()
        (no_onnx_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match=r"model\.onnx"):
            resolve_model_paths(ModelResolutionConfig(cli_path=no_onnx_dir))

        no_tok_dir = tmp_path / "no_tok"
        no_tok_dir.mkdir()
        (no_tok_dir / "model.onnx").write_bytes(b"onnx")
        with pytest.raises(FileNotFoundError, match=r"tokenizer\.json"):
            resolve_model_paths(ModelResolutionConfig(cli_path=no_tok_dir))

    def test_embedding_gemma_adapter_synthetic_onnx_3d_and_2d(self) -> None:
        recorded_prompts: list[str] = []

        def _make_mock_tokenizer(token_len: int | None = None) -> MagicMock:
            tok = MagicMock()

            def _encode(text: str, add_special_tokens: bool = True) -> MagicMock:
                assert add_special_tokens is True
                recorded_prompts.append(text)
                n = token_len if token_len is not None else max(1, len(text.split()))
                enc = MagicMock()
                enc.ids = [101] * n
                enc.attention_mask = [1] * n
                return enc

            tok.encode.side_effect = _encode
            return tok

        def _make_mock_session(rank: int = 3) -> MagicMock:
            sess = MagicMock()
            inp1 = MagicMock()
            inp1.name = "input_ids"
            inp2 = MagicMock()
            inp2.name = "attention_mask"
            sess.get_inputs.return_value = [inp1, inp2]

            def _run(_out_names: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
                seq_len = feeds["input_ids"].shape[1]
                if rank == 3:
                    # [1, seq_len, 768]
                    arr = np.ones((1, seq_len, 768), dtype=np.float32)
                    arr[0, :, 0] = 5.0
                    return [arr]
                # rank == 2 -> [1, 768]
                arr2 = np.ones((1, 768), dtype=np.float32)
                arr2[0, 0] = 3.0
                return [arr2]

            sess.run.side_effect = _run
            return sess

        # 1. Test 384-d MRL slicing with 3D mean pooling and asymmetric task prefixes
        adapter_384 = EmbeddingGemmaAdapter(
            dimension=384,
            _session=_make_mock_session(rank=3),
            _tokenizer=_make_mock_tokenizer(),
        )
        q_vec = adapter_384.encode_query("What is KacheDB?")
        assert len(q_vec) == 384
        assert abs(math.sqrt(sum(x * x for x in q_vec)) - 1.0) < 1e-6
        assert recorded_prompts[-1] == "task: search query | text: What is KacheDB?"

        d_vec = adapter_384.encode_document("KacheDB is a sub-millisecond vector database.")
        assert len(d_vec) == 384
        assert abs(math.sqrt(sum(x * x for x in d_vec)) - 1.0) < 1e-6
        assert (
            recorded_prompts[-1]
            == "task: document indexing | text: KacheDB is a sub-millisecond vector database."
        )

        # 2. Test 768-d full output with 2D pooled ONNX output
        adapter_768 = EmbeddingGemmaAdapter(
            dimension=768,
            _session=_make_mock_session(rank=2),
            _tokenizer=_make_mock_tokenizer(),
        )
        vec_768 = adapter_768.encode("Hello EmbeddingGemma 2")
        assert len(vec_768) == 768
        assert abs(math.sqrt(sum(x * x for x in vec_768)) - 1.0) < 1e-6

        # 3. Test 8,192-token context window boundary (pass at 8192, reject at 8193)
        adapter_8192 = EmbeddingGemmaAdapter(
            dimension=384,
            max_tokens=DEFAULT_MAX_TOKENS,
            _session=_make_mock_session(rank=2),
            _tokenizer=_make_mock_tokenizer(token_len=8192),
        )
        ok_vec = adapter_8192.encode("valid 8k input")
        assert len(ok_vec) == 384

        adapter_8193 = EmbeddingGemmaAdapter(
            dimension=384,
            max_tokens=DEFAULT_MAX_TOKENS,
            _session=_make_mock_session(rank=2),
            _tokenizer=_make_mock_tokenizer(token_len=8193),
        )
        with pytest.raises(ValueError, match="exceeds maximum context window of 8192 tokens"):
            adapter_8193.encode("oversized 8k+1 input")

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            adapter_384.encode("   ")

    def test_embedding_gemma_adapter_config_validation(self) -> None:
        with pytest.raises(ValueError, match="Unsupported MRL target dimension"):
            EmbeddingGemmaAdapter(dimension=256, _session=MagicMock(), _tokenizer=MagicMock())
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            EmbeddingGemmaAdapter(max_tokens=0, _session=MagicMock(), _tokenizer=MagicMock())
        with pytest.raises(ValueError, match="concurrency_limit must be >= 1"):
            EmbeddingGemmaAdapter(concurrency_limit=0, _session=MagicMock(), _tokenizer=MagicMock())

    def test_embedding_gemma_adapter_loads_from_model_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model_dir = tmp_path / "gemma2"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"fake-onnx")
        (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

        mock_ort = MagicMock()
        mock_sess = MagicMock()
        inp = MagicMock()
        inp.name = "input_ids"
        mock_sess.get_inputs.return_value = [inp]
        mock_sess.run.return_value = [np.ones((1, 768), dtype=np.float32)]
        mock_ort.InferenceSession.return_value = mock_sess

        mock_tok_mod = MagicMock()
        mock_tok = MagicMock()
        enc = MagicMock()
        enc.ids = [1, 2, 3]
        enc.attention_mask = [1, 1, 1]
        mock_tok.encode.return_value = enc
        mock_tok_mod.Tokenizer.from_file.return_value = mock_tok

        monkeypatch.setitem(sys.modules, "onnxruntime", mock_ort)
        monkeypatch.setitem(sys.modules, "tokenizers", mock_tok_mod)

        adapter = EmbeddingGemmaAdapter(model_path=model_dir, dimension=384)
        assert adapter.resolved_model is not None
        assert adapter.resolved_model.source is ModelSourceMode.EXPLICIT_PATH
        vec = adapter.encode("test local onnx")
        assert len(vec) == 384

        # Verify SemanticCache._auto_select_embedder picks up EMBEDDING_MODEL_PATH
        monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(model_dir))
        mock_client = MagicMock(spec=KacheClient)
        cache = SemanticCache(client=mock_client, index_name="gemma_auto")
        assert isinstance(cache.embedder, EmbeddingGemmaAdapter)

    def test_embedding_gemma_hf_fallback_and_truncation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)

        # 1. Default allow_hf_fallback=False fails fast when no local ONNX directory exists
        with pytest.raises(FileNotFoundError, match="EMBEDDING_MODEL_PATH"):
            EmbeddingGemmaAdapter(allow_hf_fallback=False)

        # 2. Explicit allow_hf_fallback=True falls back to HuggingFace Transformers
        torch = pytest.importorskip("torch")
        fake_tok = MagicMock()
        mock_inputs = {
            "input_ids": torch.zeros((1, 6), dtype=torch.long),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
        }

        class MockInputs(dict):  # type: ignore[type-arg]
            def to(self, _dev: str) -> MockInputs:
                return self

        fake_tok.return_value = MockInputs(mock_inputs)
        fake_model = MagicMock()
        fake_model.return_value = [torch.ones((1, 6, 768))]
        fake_model.to.return_value = fake_model

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = fake_tok
        mock_transformers.AutoModel.from_pretrained.return_value = fake_model
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        hf_adapter = EmbeddingGemmaAdapter(dimension=384, device="cpu", allow_hf_fallback=True)
        vec = hf_adapter.encode_document("test hf fallback", truncate=True)
        assert len(vec) == 384
        assert abs(math.sqrt(sum(x * x for x in vec)) - 1.0) < 1e-5
