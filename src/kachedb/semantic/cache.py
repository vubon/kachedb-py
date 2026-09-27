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
    from collections.abc import Callable, Iterable

    from ..async_client import AsyncKacheClient
    from ..client import KacheClient


@dataclass(frozen=True)
class SearchResult:
    """Represents a successful semantic cache match.

    Attributes
    ----------
    key : str
        Matched prompt or symbol key.
    similarity : float
        Cosine similarity score in range [0.0, 1.0] (1.0 for exact matches).
    value : str
        Matched content (full parent content if parent resolution succeeded, or chunk/completion).
    is_exact : bool
        True if resolved via direct symbol shortcut without vector embedding.
    parent_key : str | None
        Key of the parent document in KV store if this was a chunk.
    chunk : str | None
        Specific matching chunk content if parent was resolved.
    """

    key: str
    similarity: float
    value: str
    is_exact: bool = False
    parent_key: str | None = None
    chunk: str | None = None

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
    workspace : str
        Workspace identifier for isolation and document keys (e.g. "database", "cashflow").
    """

    def __init__(
        self,
        client: KacheClient,
        index_name: str = "default_semantic_cache",
        *,
        similarity_threshold: float = 0.85,
        ttl_seconds: int | None = 86400,
        embedder: EmbeddingAdapter | Callable[[str], list[float]] | None = None,
        workspace: str = "",
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.workspace = workspace

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
        tag_mask: int = 0,
        tags: Iterable[str] | None = None,
        parent_key: str | None = None,
        store_exact: bool = True,
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
        tag_mask : int
            64-bit integer bitmask for pre-filtering. Default: 0.
        tags : Iterable[str] | None
            Optional collection of tag strings to deterministically hash into the bitmask.
        parent_key : str | None
            Key of parent document in KV store.
        store_exact : bool
            If True, stores the prompt/value in the KV store under 'doc:{workspace}:{prompt}'
            for fast (< 50 ns) exact-match shortcutting.

        Returns
        -------
        bool
            True if stored successfully in KacheDB.
        """
        ex = ttl_seconds if ttl_seconds is not None else self.ttl_seconds

        if store_exact and self.workspace:
            doc_key = f"doc:{self.workspace}:{prompt.strip()}"
            self.client.set(doc_key, response, ex=ex)

        vector = self.embedder.encode(prompt)
        vadd_kwargs: dict[str, Any] = {
            "index": self.index_name,
            "item_id": prompt,
            "vector": vector,
            "payload": response,
            "ex": ex,
        }
        if tag_mask > 0:
            vadd_kwargs["tag_mask"] = tag_mask
        if tags is not None:
            vadd_kwargs["tags"] = tags
        if parent_key is not None:
            vadd_kwargs["parent_key"] = parent_key

        return self.client.vadd(**vadd_kwargs)

    def get(
        self,
        prompt: str,
        *,
        threshold: float | None = None,
        filter_mask: int = 0,
        filter_tags: Iterable[str] | None = None,
        exact_first: bool = True,
        resolve_parent: bool = True,
    ) -> SearchResult | None:
        """Search the cache for semantically equivalent prompts above the similarity threshold.

        Pipeline:
        1. Exact Symbol / Key Shortcut (< 50 ns direct KV lookup)
        2. Vector Semantic Retrieval with Bitmask Pre-filtering
        3. Parent Document Expansion (if match is a chunk with parent_key)

        Parameters
        ----------
        prompt : str
            Incoming user query or prompt.
        threshold : float | None
            Optional cosine similarity threshold override.
        filter_mask : int
            Optional 64-bit integer bitmask filter.
        filter_tags : Iterable[str] | None
            Optional collection of tag strings to require in bitmask pre-filtering.
        exact_first : bool
            Whether to attempt exact KV lookup before running embedding model (default: True).
        resolve_parent : bool
            Whether to expand a chunk match to the full parent document (default: True).

        Returns
        -------
        SearchResult | None
            Matched SearchResult or None on cache MISS.
        """
        clean_prompt = prompt.strip()

        # 1. Exact Symbol / Key Shortcut (active when workspace is configured)
        if exact_first and self.workspace:
            doc_key = f"doc:{self.workspace}:{clean_prompt}"
            try:
                exact_val = self.client.get(doc_key)
                if exact_val is not None and isinstance(exact_val, (bytes, str)):
                    val_str = (
                        exact_val.decode("utf-8")
                        if isinstance(exact_val, bytes)
                        else str(exact_val)
                    )
                    return SearchResult(
                        key=clean_prompt, similarity=1.0, value=val_str, is_exact=True
                    )
            except Exception:
                pass

        # 2. Vector Semantic Retrieval with Bitmask Pre-filtering
        th = threshold if threshold is not None else self.similarity_threshold
        vector = self.embedder.encode(prompt)

        vsearch_kwargs: dict[str, Any] = {
            "index": self.index_name,
            "query_vector": vector,
            "top_k": 1,
            "threshold": th,
        }
        if filter_mask > 0:
            vsearch_kwargs["filter_mask"] = filter_mask
        if filter_tags is not None:
            vsearch_kwargs["filter_tags"] = filter_tags

        matches = self.client.vsearch(**vsearch_kwargs)
        if not matches:
            return None

        best = matches[0]
        raw_key = best[0]
        key_str = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
        score = float(best[1])
        raw_payload = best[2] if len(best) > 2 else None
        val_str = (
            raw_payload.decode("utf-8")
            if isinstance(raw_payload, bytes)
            else str(raw_payload or "")
        )
        raw_parent = getattr(best, "parent_key", None) or (best[3] if len(best) > 3 else None)
        parent_k = (
            raw_parent.decode("utf-8")
            if isinstance(raw_parent, bytes)
            else (str(raw_parent) if raw_parent is not None else None)
        )

        # 3. Parent Document Expansion
        if resolve_parent and parent_k:
            try:
                parent_raw = self.client.get(parent_k)
                if parent_raw is not None and isinstance(parent_raw, (bytes, str)):
                    parent_content = (
                        parent_raw.decode("utf-8")
                        if isinstance(parent_raw, bytes)
                        else str(parent_raw)
                    )
                    return SearchResult(
                        key=key_str,
                        similarity=score,
                        value=parent_content,
                        parent_key=parent_k,
                        chunk=val_str,
                    )
            except Exception:
                pass

        return SearchResult(
            key=key_str,
            similarity=score,
            value=val_str,
            parent_key=parent_k,
            chunk=val_str if parent_k else None,
        )

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
    workspace : str
        Workspace identifier for isolation and document keys.
    """

    def __init__(
        self,
        client: AsyncKacheClient,
        index_name: str = "default_semantic_cache",
        *,
        similarity_threshold: float = 0.85,
        ttl_seconds: int | None = 86400,
        embedder: EmbeddingAdapter | Callable[[str], list[float]] | None = None,
        workspace: str = "",
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.workspace = workspace

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
        tag_mask: int = 0,
        tags: Iterable[str] | None = None,
        parent_key: str | None = None,
        store_exact: bool = True,
    ) -> bool:
        """Store a prompt and response in the semantic cache asynchronously."""
        ex = ttl_seconds if ttl_seconds is not None else self.ttl_seconds

        if store_exact and self.workspace:
            doc_key = f"doc:{self.workspace}:{prompt.strip()}"
            await self.client.set(doc_key, response, ex=ex)

        vector = self.embedder.encode(prompt)
        vadd_kwargs: dict[str, Any] = {
            "index": self.index_name,
            "item_id": prompt,
            "vector": vector,
            "payload": response,
            "ex": ex,
        }
        if tag_mask > 0:
            vadd_kwargs["tag_mask"] = tag_mask
        if tags is not None:
            vadd_kwargs["tags"] = tags
        if parent_key is not None:
            vadd_kwargs["parent_key"] = parent_key

        return await self.client.vadd(**vadd_kwargs)

    async def get(
        self,
        prompt: str,
        *,
        threshold: float | None = None,
        filter_mask: int = 0,
        filter_tags: Iterable[str] | None = None,
        exact_first: bool = True,
        resolve_parent: bool = True,
    ) -> SearchResult | None:
        """Search the cache asynchronously for semantically equivalent prompts."""
        clean_prompt = prompt.strip()

        # 1. Exact Symbol / Key Shortcut (active when workspace is configured)
        if exact_first and self.workspace:
            doc_key = f"doc:{self.workspace}:{clean_prompt}"
            try:
                exact_val = await self.client.get(doc_key)
                if exact_val is not None and isinstance(exact_val, (bytes, str)):
                    val_str = (
                        exact_val.decode("utf-8")
                        if isinstance(exact_val, bytes)
                        else str(exact_val)
                    )
                    return SearchResult(
                        key=clean_prompt, similarity=1.0, value=val_str, is_exact=True
                    )
            except Exception:
                pass

        # 2. Vector Semantic Retrieval with Bitmask Pre-filtering
        th = threshold if threshold is not None else self.similarity_threshold
        vector = self.embedder.encode(prompt)

        vsearch_kwargs: dict[str, Any] = {
            "index": self.index_name,
            "query_vector": vector,
            "top_k": 1,
            "threshold": th,
        }
        if filter_mask > 0:
            vsearch_kwargs["filter_mask"] = filter_mask
        if filter_tags is not None:
            vsearch_kwargs["filter_tags"] = filter_tags

        matches = await self.client.vsearch(**vsearch_kwargs)
        if not matches:
            return None

        best = matches[0]
        raw_key = best[0]
        key_str = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
        score = float(best[1])
        raw_payload = best[2] if len(best) > 2 else None
        val_str = (
            raw_payload.decode("utf-8")
            if isinstance(raw_payload, bytes)
            else str(raw_payload or "")
        )
        raw_parent = getattr(best, "parent_key", None) or (best[3] if len(best) > 3 else None)
        parent_k = (
            raw_parent.decode("utf-8")
            if isinstance(raw_parent, bytes)
            else (str(raw_parent) if raw_parent is not None else None)
        )

        # 3. Parent Document Expansion
        if resolve_parent and parent_k:
            try:
                parent_raw = await self.client.get(parent_k)
                if parent_raw is not None and isinstance(parent_raw, (bytes, str)):
                    parent_content = (
                        parent_raw.decode("utf-8")
                        if isinstance(parent_raw, bytes)
                        else str(parent_raw)
                    )
                    return SearchResult(
                        key=key_str,
                        similarity=score,
                        value=parent_content,
                        parent_key=parent_k,
                        chunk=val_str,
                    )
            except Exception:
                pass

        return SearchResult(
            key=key_str,
            similarity=score,
            value=val_str,
            parent_key=parent_k,
            chunk=val_str,
        )

    async def delete(self, prompt: str) -> bool:
        """Delete a prompt entry from the semantic cache asynchronously."""
        return await self.client.vdel(self.index_name, prompt)

    async def stats(self) -> dict[str, Any]:
        """Return index metrics asynchronously."""
        stats = await self.client.vstats(self.index_name)
        return stats or {}
