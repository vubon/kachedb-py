"""
Embedding adapters for KacheDB Semantic Cache.

Supports pluggable embedding providers including Google EmbeddingGemma 2 (ONNX Runtime
with Matryoshka Representation Learning), FastEmbed (ONNX Runtime),
HuggingFace Transformers, SentenceTransformers, OpenAI, custom callable functions,
and a lightweight semantic MockEmbedder for zero-dependency testing.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# ── EmbeddingGemma 2 Constants & Task Formatting ──────────────────────────────

PREFIX_SEARCH_QUERY = "task: search query | text: "
PREFIX_DOCUMENT_INDEXING = "task: document indexing | text: "
DEFAULT_MAX_TOKENS = 8192
DEFAULT_GEMMA_DIMENSION = 384
NATIVE_GEMMA_DIMENSION = 768
EPSILON = 1e-12
ENV_EMBEDDING_MODEL_PATH = "EMBEDDING_MODEL_PATH"


def _default_cache_dir() -> Path:
    """Return local CWD cache if present, otherwise user home .fastembed_cache."""
    cwd_cache = Path(".fastembed_cache")
    if cwd_cache.exists() and cwd_cache.is_dir():
        return cwd_cache
    try:
        return Path.home() / ".fastembed_cache"
    except Exception:
        return cwd_cache


DEFAULT_CACHE_DIR = str(_default_cache_dir())


class EmbeddingTask(str, Enum):
    """Asymmetric task types supported by EmbeddingGemma 2."""

    SEARCH_QUERY = "search_query"
    DOCUMENT_INDEXING = "document_indexing"

    @property
    def prefix(self) -> str:
        """Return the canonical EmbeddingGemma 2 task prefix string."""
        if self is EmbeddingTask.SEARCH_QUERY:
            return PREFIX_SEARCH_QUERY
        return PREFIX_DOCUMENT_INDEXING


def format_task_prompt(task: EmbeddingTask, text: str) -> str:
    """Format an input string with the appropriate asymmetric task prefix.

    Idempotent: if ``text`` already begins with a recognized EmbeddingGemma 2
    task prefix, it is returned unchanged.
    """
    if text.startswith(PREFIX_SEARCH_QUERY) or text.startswith(PREFIX_DOCUMENT_INDEXING):
        return text
    return f"{task.prefix}{text}"


# ── Matryoshka Representation Learning (MRL) Slicing & L2 Normalization ──────


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """L2-normalize a float vector in-place / copy to unit length (||v||_2 = 1.0)."""
    if not vector:
        raise ValueError("Cannot normalize an empty vector")

    sum_sq = 0.0
    for val in vector:
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError("Vector contains NaN or infinite values")
        sum_sq += fval * fval

    norm = math.sqrt(sum_sq)
    if norm < EPSILON or not math.isfinite(norm):
        raise ValueError(f"Vector norm is zero or sub-epsilon ({norm:.3e}), cannot L2-normalize")

    inv_norm = 1.0 / norm
    return [float(x) * inv_norm for x in vector]


def slice_and_normalize(
    vector: Sequence[float],
    target_dim: int = DEFAULT_GEMMA_DIMENSION,
) -> list[float]:
    """Slice a raw EmbeddingGemma 2 output vector to ``target_dim`` (384 or 768)
    and re-normalize to unit L2 length.
    """
    if target_dim not in (DEFAULT_GEMMA_DIMENSION, NATIVE_GEMMA_DIMENSION):
        raise ValueError(f"Unsupported MRL target dimension: {target_dim} (supported: 384, 768)")
    if len(vector) < target_dim:
        raise ValueError(
            f"Source vector length ({len(vector)}) is smaller than target dimension ({target_dim})"
        )
    return l2_normalize(vector[:target_dim])


# ── Multi-Stage Model Path Resolver ───────────────────────────────────────────


class ModelSourceMode(str, Enum):
    """Describes which configuration tier resolved the model artifacts."""

    EXPLICIT_PATH = "explicit_path"
    ENV_VAR = "env_var"
    LOCAL_CACHE = "local_cache"
    HUGGINGFACE_CACHE = "local_cache"


@dataclass(frozen=True)
class ModelResolutionConfig:
    """Configuration inputs for resolving EmbeddingGemma 2 ONNX artifacts."""

    cli_path: Path | None = None
    env_path: Path | None = None
    cache_dir: Path = field(default_factory=lambda: Path(DEFAULT_CACHE_DIR))

    @classmethod
    def from_env(
        cls,
        cli_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> ModelResolutionConfig:
        """Build a resolution config reading ``EMBEDDING_MODEL_PATH`` from the environment."""
        raw_env = os.environ.get(ENV_EMBEDDING_MODEL_PATH, "").strip()
        env_p = Path(raw_env) if raw_env else None
        cli_p = Path(cli_path) if cli_path is not None and str(cli_path).strip() else None
        cache_p = Path(cache_dir) if cache_dir is not None else Path(DEFAULT_CACHE_DIR)
        return cls(cli_path=cli_p, env_path=env_p, cache_dir=cache_p)


@dataclass(frozen=True)
class ResolvedModel:
    """Verified paths to the ONNX model graph and HuggingFace tokenizer."""

    root_dir: Path
    onnx_path: Path
    tokenizer_path: Path
    source: ModelSourceMode


def _validate_model_dir(directory: Path, source: ModelSourceMode) -> ResolvedModel:
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(
            f"Model directory does not exist or is not a directory "
            f"(source: {source.value}, path: {directory})"
        )

    onnx_path = directory / "model.onnx"
    if not onnx_path.exists() or not onnx_path.is_file():
        raise FileNotFoundError(
            f"Missing required ONNX model file 'model.onnx' in {directory} (source: {source.value})"
        )

    tokenizer_path = directory / "tokenizer.json"
    if not tokenizer_path.exists() or not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Missing required tokenizer file 'tokenizer.json' in {directory} "
            f"(source: {source.value})"
        )

    return ResolvedModel(
        root_dir=directory,
        onnx_path=onnx_path,
        tokenizer_path=tokenizer_path,
        source=source,
    )


def resolve_model_paths(config: ModelResolutionConfig | None = None) -> ResolvedModel:
    """Resolve EmbeddingGemma 2 ONNX model and tokenizer paths using strict priority:

    1. Explicit ``cli_path`` / ``model_path`` argument
    2. ``EMBEDDING_MODEL_PATH`` environment variable
    3. Local cache directory fallback (``.fastembed_cache``)
    """
    cfg = config if config is not None else ModelResolutionConfig.from_env()

    if cfg.cli_path is not None:
        return _validate_model_dir(cfg.cli_path, ModelSourceMode.EXPLICIT_PATH)

    if cfg.env_path is not None:
        return _validate_model_dir(cfg.env_path, ModelSourceMode.ENV_VAR)

    return _validate_model_dir(cfg.cache_dir, ModelSourceMode.HUGGINGFACE_CACHE)


class EmbeddingAdapter(Protocol):
    """Protocol for embedding model backends."""

    def encode(self, text: str) -> list[float]:
        """Convert input text into a normalized float32 vector embedding."""
        ...


class EmbeddingGemmaAdapter:
    """Google EmbeddingGemma 2 (300M) embedding adapter with Matryoshka (MRL)
    dimensionality slicing (384-d / 768-d), asymmetric task prefixes, 8,192-token
    context window validation, and bounded concurrency.

    Supports ONNX Runtime (``onnxruntime`` + ``tokenizers``) as the primary zero-PyTorch
    execution backend and automatically falls back to HuggingFace ``transformers`` + ``torch``
    when local ONNX artifacts are not configured.
    """

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        *,
        model_path: str | Path | None = None,
        dimension: int = DEFAULT_GEMMA_DIMENSION,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        concurrency_limit: int = 2,
        search_prefix: str = PREFIX_SEARCH_QUERY,
        indexing_prefix: str = PREFIX_DOCUMENT_INDEXING,
        default_task: EmbeddingTask = EmbeddingTask.SEARCH_QUERY,
        device: str | None = None,
        allow_hf_fallback: bool = False,
        _session: Any | None = None,
        _tokenizer: Any | None = None,
    ) -> None:
        if dimension not in (DEFAULT_GEMMA_DIMENSION, NATIVE_GEMMA_DIMENSION):
            raise ValueError(f"Unsupported MRL target dimension: {dimension} (supported: 384, 768)")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if concurrency_limit < 1:
            raise ValueError(f"concurrency_limit must be >= 1, got {concurrency_limit}")

        self.model_name = model_name
        self.dimension = dimension
        self.max_tokens = max_tokens
        self.concurrency_limit = concurrency_limit
        self.search_prefix = search_prefix
        self.indexing_prefix = indexing_prefix
        self.default_task = default_task
        self.allow_hf_fallback = allow_hf_fallback
        self._semaphore = threading.Semaphore(concurrency_limit)
        self.resolved_model: ResolvedModel | None = None

        self._session: Any = None
        self._tokenizer: Any = None
        self._hf_model: Any = None
        self._hf_tokenizer: Any = None
        self._device: str = device or "cpu"

        if _session is not None and _tokenizer is not None:
            self._backend = "onnx"
            self._session = _session
            self._tokenizer = _tokenizer
            return

        # 1. Try resolving local ONNX Runtime model artifacts
        resolved: ResolvedModel | None = None
        try:
            cfg = ModelResolutionConfig.from_env(cli_path=model_path)
            resolved = resolve_model_paths(cfg)
        except FileNotFoundError as exc:
            if model_path is not None or not allow_hf_fallback:
                raise FileNotFoundError(
                    f"{exc}. Set `model_path` or `EMBEDDING_MODEL_PATH` to a directory containing "
                    "`model.onnx` and `tokenizer.json`, or pass `allow_hf_fallback=True` to load "
                    "from HuggingFace Transformers."
                ) from exc
            resolved = None

        if resolved is not None:
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as e:
                raise ImportError(
                    "EmbeddingGemmaAdapter ONNX backend requires `onnxruntime` and `tokenizers`. "
                    "Install via `pip install onnxruntime tokenizers`"
                ) from e

            tok = Tokenizer.from_file(str(resolved.tokenizer_path))
            tok.no_truncation()

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess = ort.InferenceSession(
                str(resolved.onnx_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._backend = "onnx"
            self._tokenizer = tok
            self._session = sess
            self.resolved_model = resolved
            return

        # 2. Explicit Fallback to HuggingFace Transformers + PyTorch (when allow_hf_fallback=True)
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "EmbeddingGemmaAdapter requires either local ONNX model artifacts "
                "(via `model_path` or `EMBEDDING_MODEL_PATH` with `onnxruntime` + `tokenizers`) "
                "or `transformers` and `torch` installed."
            ) from e

        self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._hf_model = AutoModel.from_pretrained(model_name)
        if device:
            self._device = device
        elif torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        self._hf_model.to(self._device)
        self._hf_model.eval()
        self._backend = "transformers"

    def _format_prompt(self, task: EmbeddingTask, text: str) -> str:
        """Apply configured asymmetric task prefix idempotently."""
        if (
            text.startswith(PREFIX_SEARCH_QUERY)
            or text.startswith(PREFIX_DOCUMENT_INDEXING)
            or (self.search_prefix and text.startswith(self.search_prefix))
            or (self.indexing_prefix and text.startswith(self.indexing_prefix))
        ):
            return text
        prefix = self.search_prefix if task is EmbeddingTask.SEARCH_QUERY else self.indexing_prefix
        return f"{prefix}{text}"

    def encode_with_task(
        self,
        text: str,
        task: EmbeddingTask,
        *,
        truncate: bool = False,
    ) -> list[float]:
        """Encode ``text`` with an explicit ``EmbeddingTask`` (query vs. document)."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        formatted = self._format_prompt(task, text)
        with self._semaphore:
            if self._backend == "onnx":
                raw_vec = self._encode_onnx(formatted, truncate=truncate)
            else:
                raw_vec = self._encode_transformers(formatted, truncate=truncate)

        return slice_and_normalize(raw_vec, self.dimension)

    def encode_query(self, text: str) -> list[float]:
        """Encode a search query using the asymmetric query task prefix."""
        return self.encode_with_task(text, EmbeddingTask.SEARCH_QUERY)

    def encode_document(self, text: str, *, truncate: bool = False) -> list[float]:
        """Encode a document or symbol snippet using the asymmetric indexing task prefix."""
        return self.encode_with_task(text, EmbeddingTask.DOCUMENT_INDEXING, truncate=truncate)

    def encode(self, text: str) -> list[float]:
        """Encode text using the adapter's default task (``SEARCH_QUERY`` by default)."""
        return self.encode_with_task(text, self.default_task)

    def _encode_onnx(self, formatted_text: str, *, truncate: bool = False) -> list[float]:
        encoding = self._tokenizer.encode(formatted_text, add_special_tokens=True)
        ids = list(encoding.ids)
        mask = list(encoding.attention_mask)
        seq_len = len(ids)

        if seq_len == 0:
            raise ValueError("Tokenizer produced 0 tokens for input")
        if seq_len > self.max_tokens:
            if truncate:
                ids = ids[: self.max_tokens]
                mask = mask[: self.max_tokens]
                seq_len = self.max_tokens
            else:
                raise ValueError(
                    f"Input length ({seq_len} tokens) exceeds maximum context window "
                    f"of {self.max_tokens} tokens"
                )

        input_ids = np.asarray([ids], dtype=np.int64)
        attention_mask = np.asarray([mask], dtype=np.int64)

        input_names = [inp.name for inp in self._session.get_inputs()]
        feeds: dict[str, Any] = {"input_ids": input_ids}
        if "attention_mask" in input_names:
            feeds["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self._session.run(None, feeds)
        if not outputs:
            raise RuntimeError("ONNX session returned no output tensors")

        # Prefer named sentence_embedding or 2D pooled output if provided directly
        output_names = [out.name for out in self._session.get_outputs()]
        if "sentence_embedding" in output_names:
            out_idx = output_names.index("sentence_embedding")
            candidate = np.asarray(outputs[out_idx], dtype=np.float32)
        else:
            candidate = np.asarray(outputs[0], dtype=np.float32)

        if candidate.ndim == 2:
            if candidate.shape[0] < 1:
                raise RuntimeError(f"Invalid 2D ONNX output shape: {candidate.shape}")
            return [float(x) for x in candidate[0].tolist()]

        if candidate.ndim == 3:
            # Shape: [1, seq_len, hidden_dim] -> attention-masked mean pooling
            hidden = candidate[0]  # [seq_len, hidden_dim]
            mask_arr = np.asarray(mask, dtype=np.float32)[:, np.newaxis]  # [seq_len, 1]
            denom = float(np.sum(mask_arr))
            if denom <= 0.0:
                denom = float(seq_len)
            pooled = np.sum(hidden * mask_arr, axis=0) / denom
            return [float(x) for x in pooled.tolist()]

        raise RuntimeError(f"Unsupported ONNX output tensor rank: {candidate.ndim}")

    def _encode_transformers(self, formatted_text: str, *, truncate: bool = False) -> list[float]:
        import torch

        tokenized = self._hf_tokenizer(
            formatted_text,
            padding=False,
            truncation=truncate,
            max_length=self.max_tokens if truncate else None,
            return_tensors="pt",
        )
        seq_len = int(tokenized["input_ids"].shape[-1])
        if seq_len == 0:
            raise ValueError("Tokenizer produced 0 tokens for input")
        if seq_len > self.max_tokens:
            raise ValueError(
                f"Input length ({seq_len} tokens) exceeds maximum context window "
                f"of {self.max_tokens} tokens"
            )

        inputs = tokenized.to(self._device)
        with torch.no_grad():
            outputs = self._hf_model(**inputs)
            token_embeddings = outputs[0]
            input_mask_expanded = (
                inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            return [float(x) for x in mean_pooled[0].cpu().tolist()]


class TransformersEmbedder:
    """HuggingFace Transformers embedding provider using mean pooling.

    Requires ``transformers`` and ``torch``.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required. Install via `pip install transformers torch`"
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str) -> list[float]:
        import torch

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs[0]
            input_mask_expanded = (
                inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            return [float(x) for x in normalized[0].cpu().tolist()]


class FastEmbedAdapter:
    """FastEmbed (ONNX Runtime) embedding provider.

    Requires ``fastembed`` (``pip install fastembed``).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", **kwargs: Any) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "FastEmbed is not installed. Install it via `pip install fastembed`"
            ) from e
        self.model = TextEmbedding(model_name=model_name, **kwargs)

    def encode(self, text: str) -> list[float]:
        embeddings = list(self.model.embed([text]))
        return [float(x) for x in embeddings[0]]


class SentenceTransformersAdapter:
    """HuggingFace SentenceTransformers provider.

    Requires ``sentence-transformers`` (``pip install sentence-transformers``).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs: Any) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformers is not installed. "
                "Install it via `pip install sentence-transformers`"
            ) from e
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vec]


class OpenAIAdapter:
    """OpenAI Embeddings API provider.

    Requires ``openai`` (``pip install openai``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        **kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI SDK is not installed. Install it via `pip install openai`"
            ) from e
        self.client = OpenAI(api_key=api_key, **kwargs)
        self.model = model

    def encode(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(input=[text], model=self.model)
        return [float(x) for x in resp.data[0].embedding]


class CallableAdapter:
    """Wraps any user-defined function ``fn(text: str) -> list[float]``."""

    def __init__(self, fn: Callable[[str], list[float]]) -> None:
        self.fn = fn

    def encode(self, text: str) -> list[float]:
        return [float(x) for x in self.fn(text)]


# Semantic clusters for intent-aware mock embeddings
_SEMANTIC_CLUSTERS = {
    "password_reset": [
        "password",
        "reset",
        "change",
        "recover",
        "forgot",
        "pass",
        "login",
        "credentials",
    ],
    "subscription_cancel": [
        "cancel",
        "subscription",
        "membership",
        "unsubscribe",
        "billing",
        "terminate",
        "plan",
        "stop",
    ],
    "rate_limits": [
        "rate",
        "limit",
        "limits",
        "api",
        "quota",
        "throttling",
        "requests",
        "minute",
        "second",
        "rps",
    ],
    "invoices": [
        "invoice",
        "invoices",
        "receipt",
        "receipts",
        "billing",
        "pdf",
        "csv",
        "tax",
        "payment",
        "export",
        "download",
    ],
    "team_management": [
        "team",
        "invite",
        "coworker",
        "coworkers",
        "member",
        "members",
        "workspace",
        "colleague",
        "role",
        "add",
    ],
}


class MockEmbedder:
    """Deterministic token & semantic hash embedder for unit tests and benchmarks.

    Produces unit-normalized ($L_2 = 1.0$) pseudo-embeddings with semantic cluster awareness.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def encode(self, text: str) -> list[float]:
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            vec = [1.0 / math.sqrt(self.dimension)] * self.dimension
            return vec

        vec = [0.0] * self.dimension

        # 1. Project semantic intent cluster centroids
        for cluster_name, keywords in _SEMANTIC_CLUSTERS.items():
            matches = sum(1 for w in words if w in keywords)
            if matches > 0:
                cluster_seed = int(hashlib.md5(cluster_name.encode()).hexdigest(), 16)
                for offset in range(32):
                    idx = (cluster_seed + offset * 11) % self.dimension
                    vec[idx] += 3.0 * matches

        # 2. General token hash projection
        for word in words:
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            idx1 = h % self.dimension
            idx2 = (h >> 16) % self.dimension
            idx3 = (h >> 32) % self.dimension
            vec[idx1] += 1.0
            vec[idx2] += 0.5
            vec[idx3] += 0.25

        # L2 Normalize to unit vector
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-12:
            vec = [x / norm for x in vec]
        return vec
