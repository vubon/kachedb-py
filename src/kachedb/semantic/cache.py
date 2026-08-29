"""
High-level Semantic Vector Cache for KacheDB.

Matches incoming prompts and queries by semantic intent and cosine similarity
rather than exact string equality, returning cached LLM completions in < 50 µs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .embedders import (
    CallableAdapter,
    EmbeddingAdapter,
    FastEmbedAdapter,
    MockEmbedder,
    SentenceTransformersAdapter,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..async_client import AsyncKacheClient
    from ..client import KacheClient


@dataclass(frozen=True)
class SearchResult:
    """Represents a successful semantic cache match."""

    key: str
    similarity: float
    value: str

    def __str__(self) -> str:
        return self.value


class SemanticCache:
    """High-level Semantic Cache engine powered by KacheDB SIMD vector search.

    Parameters
    ----------
    client : KacheClient
        Active KacheDB client connection.
    index_name : str
        Name of the vector cache index (e.g. "faq_cache", "llm_responses").
    similarity_threshold : float
        Minimum cosine similarity (0.0 to 1.0) required to trigger a cache HIT. Default: 0.85.
    ttl_seconds : int | None
        Cache item lifetime in seconds. Default: 86400 (24 hours). None for persistent.
    embedder : EmbeddingAdapter | Callable[[str], list[float]] | None
        Embedding model adapter. If None, automatically selects available backend.
    """

    def __init__(
        self,
        client: KacheClient,
        index_name: str = "default_semantic_cache",
        *,
        similarity_threshold: float = 0.85,
        ttl_seconds: int | None = 86400,
        embedder: EmbeddingAdapter | Callable[[str], list[float]] | None = None,
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

        if embedder is None:
            self.embedder = self._auto_select_embedder()
        elif callable(embedder) and not hasattr(embedder, "encode"):
            self.embedder = CallableAdapter(embedder)
        else:
            self.embedder = embedder

    @staticmethod
    def _auto_select_embedder() -> EmbeddingAdapter:
        """Attempts to instantiate FastEmbed or SentenceTransformers,
        falling back to MockEmbedder.
        """
        try:
            return FastEmbedAdapter()
        except (ImportError, Exception):
            pass

        try:
            return SentenceTransformersAdapter()
        except (ImportError, Exception):
            pass

        return MockEmbedder()

    def set(
        self,
        prompt: str,
        response: str,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store a prompt and its corresponding LLM response in the semantic cache.

        Parameters
        ----------
        prompt : str
            User query or prompt text to embed.
        response : str
            LLM answer or completion text to cache.
        ttl_seconds : int | None
            Optional TTL override in seconds.

        Returns
        -------
        bool
            True if stored successfully in KacheDB.
        """
        vector = self.embedder.encode(prompt)
        ex = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        return self.client.vadd(
            index=self.index_name,
            item_id=prompt,
            vector=vector,
            payload=response,
            ex=ex,
        )

    def get(
        self,
        prompt: str,
        *,
        threshold: float | None = None,
    ) -> SearchResult | None:
        """Search the cache for semantically equivalent prompts above the similarity threshold.

        Parameters
        ----------
        prompt : str
            Incoming user query or prompt.
        threshold : float | None
            Optional cosine similarity threshold override.

        Returns
        -------
        SearchResult | None
            Matched SearchResult (key, similarity, value) or None on cache MISS.
        """
        th = threshold if threshold is not None else self.similarity_threshold
        vector = self.embedder.encode(prompt)

        matches = self.client.vsearch(
            index=self.index_name,
            query_vector=vector,
            top_k=1,
            threshold=th,
        )

        if not matches:
            return None

        item_id, score, payload = matches[0]
        key_str = item_id.decode() if isinstance(item_id, bytes) else str(item_id)
        val_str = payload.decode() if isinstance(payload, bytes) else str(payload or "")

        return SearchResult(key=key_str, similarity=score, value=val_str)

    def delete(self, prompt: str) -> bool:
        """Delete a prompt entry from the semantic cache."""
        return self.client.vdel(self.index_name, prompt)

    def stats(self) -> dict[str, Any]:
        """Return index metrics including active vector count and memory usage."""
        stats = self.client.vstats(self.index_name)
        return stats or {}


class AsyncSemanticCache:
    """High-level async Semantic Cache engine powered by KacheDB SIMD vector search.

    Parameters
    ----------
    client : AsyncKacheClient
        Active KacheDB async client connection.
    index_name : str
        Name of the vector cache index (e.g. "faq_cache", "llm_responses").
    similarity_threshold : float
        Minimum cosine similarity (0.0 to 1.0) required to trigger a cache HIT. Default: 0.85.
    ttl_seconds : int | None
        Cache item lifetime in seconds. Default: 86400 (24 hours). None for persistent.
    embedder : EmbeddingAdapter | Callable[[str], list[float]] | None
        Embedding model adapter. If None, automatically selects available backend.
    """

    def __init__(
        self,
        client: AsyncKacheClient,
        index_name: str = "default_semantic_cache",
        *,
        similarity_threshold: float = 0.85,
        ttl_seconds: int | None = 86400,
        embedder: EmbeddingAdapter | Callable[[str], list[float]] | None = None,
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

        if embedder is None:
            self.embedder = SemanticCache._auto_select_embedder()
        elif callable(embedder) and not hasattr(embedder, "encode"):
            self.embedder = CallableAdapter(embedder)
        else:
            self.embedder = embedder

    async def set(
        self,
        prompt: str,
        response: str,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store a prompt and response in the semantic cache asynchronously."""
        vector = self.embedder.encode(prompt)
        ex = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        return await self.client.vadd(
            index=self.index_name,
            item_id=prompt,
            vector=vector,
            payload=response,
            ex=ex,
        )

    async def get(
        self,
        prompt: str,
        *,
        threshold: float | None = None,
    ) -> SearchResult | None:
        """Search the cache asynchronously for semantically equivalent prompts."""
        th = threshold if threshold is not None else self.similarity_threshold
        vector = self.embedder.encode(prompt)

        matches = await self.client.vsearch(
            index=self.index_name,
            query_vector=vector,
            top_k=1,
            threshold=th,
        )

        if not matches:
            return None

        item_id, score, payload = matches[0]
        key_str = item_id.decode() if isinstance(item_id, bytes) else str(item_id)
        val_str = payload.decode() if isinstance(payload, bytes) else str(payload or "")

        return SearchResult(key=key_str, similarity=score, value=val_str)

    async def delete(self, prompt: str) -> bool:
        """Delete a prompt entry from the semantic cache asynchronously."""
        return await self.client.vdel(self.index_name, prompt)

    async def stats(self) -> dict[str, Any]:
        """Return index metrics asynchronously."""
        stats = await self.client.vstats(self.index_name)
        return stats or {}
