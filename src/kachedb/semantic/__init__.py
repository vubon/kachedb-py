"""
KacheDB In-Memory Semantic Caching Module.

Provides SIMD-accelerated embedding similarity search and semantic LLM response caching.
"""

from .cache import AsyncSemanticCache, SearchResult, SemanticCache
from .chunker import DocumentChunk, MarkdownChunker
from .embedders import (
    DEFAULT_GEMMA_DIMENSION,
    DEFAULT_MAX_TOKENS,
    NATIVE_GEMMA_DIMENSION,
    PREFIX_DOCUMENT_INDEXING,
    PREFIX_SEARCH_QUERY,
    CallableAdapter,
    EmbeddingAdapter,
    EmbeddingGemmaAdapter,
    EmbeddingTask,
    FastEmbedAdapter,
    MockEmbedder,
    ModelResolutionConfig,
    ModelSourceMode,
    OpenAIAdapter,
    ResolvedModel,
    SentenceTransformersAdapter,
    TransformersEmbedder,
    format_task_prompt,
    l2_normalize,
    resolve_model_paths,
    slice_and_normalize,
)

__all__ = [
    "DEFAULT_GEMMA_DIMENSION",
    "DEFAULT_MAX_TOKENS",
    "NATIVE_GEMMA_DIMENSION",
    "PREFIX_DOCUMENT_INDEXING",
    "PREFIX_SEARCH_QUERY",
    "AsyncSemanticCache",
    "CallableAdapter",
    "DocumentChunk",
    "EmbeddingAdapter",
    "EmbeddingGemmaAdapter",
    "EmbeddingTask",
    "FastEmbedAdapter",
    "MarkdownChunker",
    "MockEmbedder",
    "ModelResolutionConfig",
    "ModelSourceMode",
    "OpenAIAdapter",
    "ResolvedModel",
    "SearchResult",
    "SemanticCache",
    "SentenceTransformersAdapter",
    "TransformersEmbedder",
    "format_task_prompt",
    "l2_normalize",
    "resolve_model_paths",
    "slice_and_normalize",
]
