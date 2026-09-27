"""
KacheDB In-Memory Semantic Caching Module.

Provides SIMD-accelerated embedding similarity search and semantic LLM response caching.
"""

from .cache import AsyncSemanticCache, SearchResult, SemanticCache
from .chunker import DocumentChunk, MarkdownChunker
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
    "DocumentChunk",
    "EmbeddingAdapter",
    "FastEmbedAdapter",
    "MarkdownChunker",
    "MockEmbedder",
    "OpenAIAdapter",
    "SearchResult",
    "SemanticCache",
    "SentenceTransformersAdapter",
]
