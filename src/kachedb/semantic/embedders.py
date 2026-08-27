"""
Embedding adapters for KacheDB Semantic Cache.

Supports pluggable embedding providers including FastEmbed (ONNX Runtime),
HuggingFace Transformers, SentenceTransformers, OpenAI, custom callable functions,
and a lightweight semantic MockEmbedder for zero-dependency testing.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable


class EmbeddingAdapter(Protocol):
    """Protocol for embedding model backends."""

    def encode(self, text: str) -> list[float]:
        """Convert input text into a normalized float32 vector embedding."""
        ...


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
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
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
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
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
            from openai import OpenAI  # type: ignore[import-not-found]
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
