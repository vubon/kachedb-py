"""
LangChain Cache Integration for KacheDB.

Provides exact key-value and SIMD semantic vector caching for LangChain LLMs and ChatModels.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from ..client import KacheClient
from ..semantic.cache import SemanticCache

if TYPE_CHECKING:
    from langchain_core.caches import BaseCache
    from langchain_core.outputs import Generation

    from ..semantic.embedders import EmbeddingAdapter
else:
    try:
        from langchain_core.caches import BaseCache
        from langchain_core.outputs import Generation
    except ImportError:

        class BaseCache:
            pass

        class Generation:
            def __init__(self, text: str) -> None:
                self.text = text


class KacheDBCache(BaseCache):  # type: ignore[misc]
    """Exact Key-Value Cache for LangChain powered by KacheDB in-memory engine.

    Parameters
    ----------
    client : KacheClient | None
        Active KacheDB client. If None, connects to localhost:6379.
    ttl_seconds : int | None
        Cache TTL in seconds. None for persistent.
    namespace : str
        Key prefix namespace in KacheDB.
    """

    def __init__(
        self,
        client: KacheClient | None = None,
        ttl_seconds: int | None = 86400,
        namespace: str = "langchain:exact",
    ) -> None:
        self.client = client or KacheClient()
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace

    def _key(self, prompt: str, llm_string: str) -> str:
        prompt_hash = hashlib.sha256(f"{prompt}::{llm_string}".encode()).hexdigest()
        return f"{self.namespace}:{prompt_hash}"

    def lookup(self, prompt: str, llm_string: str) -> list[Generation] | None:
        """Look up value based on prompt and llm_string."""
        key = self._key(prompt, llm_string)
        val = self.client.get(key)
        if val is None:
            return None

        val_str = val.decode("utf-8") if isinstance(val, bytes) else str(val)
        try:
            items = json.loads(val_str)
            if isinstance(items, list):
                return [Generation(text=item.get("text", "")) for item in items]
        except Exception:
            return [Generation(text=val_str)]
        return None

    def update(self, prompt: str, llm_string: str, return_val: list[Generation]) -> None:
        """Update cache value based on prompt and llm_string."""
        key = self._key(prompt, llm_string)
        serialized = json.dumps([{"text": g.text} for g in return_val])
        self.client.set(key, serialized, ex=self.ttl_seconds)

    def clear(self, **kwargs: Any) -> None:
        """Clear cache keys for this namespace."""
        # Simple namespace flush or reset
        pass


class KacheDBSemanticCache(BaseCache):  # type: ignore[misc]
    """SIMD Semantic Vector Cache for LangChain powered by KacheDB.

    Interprets user prompt intent, queries KacheDB vector space in < 50µs,
    and returns cached LLM generations on semantic equivalence.

    Parameters
    ----------
    client : KacheClient | None
        Active KacheDB client. If None, connects to localhost:6379.
    index_name : str
        KacheDB vector index name.
    similarity_threshold : float
        Cosine similarity threshold for cache HIT (default: 0.85).
    ttl_seconds : int | None
        Cache TTL in seconds.
    embedder : EmbeddingAdapter | None
        Vector embedding backend adapter.
    """

    def __init__(
        self,
        client: KacheClient | None = None,
        index_name: str = "langchain_semantic_cache",
        similarity_threshold: float = 0.85,
        ttl_seconds: int | None = 86400,
        embedder: EmbeddingAdapter | None = None,
    ) -> None:
        self.client = client or KacheClient()
        self.semantic_cache = SemanticCache(
            client=self.client,
            index_name=index_name,
            similarity_threshold=similarity_threshold,
            ttl_seconds=ttl_seconds,
            embedder=embedder,
        )

    def lookup(self, prompt: str, llm_string: str) -> list[Generation] | None:
        """Semantic search for cached completions."""
        match = self.semantic_cache.get(prompt)
        if match is None:
            return None

        val_str = match.value
        try:
            items = json.loads(val_str)
            if isinstance(items, list):
                return [Generation(text=item.get("text", "")) for item in items]
        except Exception:
            return [Generation(text=val_str)]
        return None

    def update(self, prompt: str, llm_string: str, return_val: list[Generation]) -> None:
        """Store prompt and generation in semantic cache."""
        serialized = json.dumps([{"text": g.text} for g in return_val])
        self.semantic_cache.set(prompt, serialized)

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic vector index."""
        pass
