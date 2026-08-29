"""
KacheDB In-Memory Semantic Caching Module.

Provides SIMD-accelerated embedding similarity search and semantic LLM response caching.
"""

from .cache import AsyncSemanticCache, SearchResult, SemanticCache
from .embedders import (
    CallableAdapter,
    EmbeddingAdapter,
    FastEmbedAdapter,
    MockEmbedder,
    OpenAIAdapter,
    SentenceTransformersAdapter,
)

__all__ = [
    "AsyncSemanticCache",
    "CallableAdapter",
    "EmbeddingAdapter",
    "FastEmbedAdapter",
    "MockEmbedder",
    "OpenAIAdapter",
    "SearchResult",
    "SemanticCache",
    "SentenceTransformersAdapter",
]
