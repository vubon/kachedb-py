"""
Unit tests for KacheDB Hybrid Context Engine (Phase 2):
- Deterministic tag bitmask encoding (kachedb.tags)
- Hierarchical Markdown chunker (kachedb.semantic.chunker)
- VectorMatch tuple compatibility & parent pointer access
- SemanticCache exact-first shortcut & parent resolution
"""

from __future__ import annotations

from kachedb import (
    MarkdownChunker,
    SemanticCache,
    VectorMatch,
    tag_to_bit,
    tags_to_bitmask,
)
from kachedb.semantic.embedders import MockEmbedder


def test_tag_to_bit_determinism() -> None:
    bit1 = tag_to_bit("workspace:database")
    bit2 = tag_to_bit("workspace:database")
    bit3 = tag_to_bit("  WORKSPACE:DATABASE  ")
    assert 0 <= bit1 < 64
    assert bit1 == bit2 == bit3

    # Different tags should generally produce different bits
    bit_cf = tag_to_bit("workspace:cashflow")
    assert 0 <= bit_cf < 64


def test_tags_to_bitmask() -> None:
    mask = tags_to_bitmask(["workspace:database", "type:architecture"])
    assert mask > 0

    bit1 = tag_to_bit("workspace:database")
    bit2 = tag_to_bit("type:architecture")
    expected = (1 << bit1) | (1 << bit2)
    assert mask == expected

    # Empty tags return base_mask
    assert tags_to_bitmask(None, base_mask=42) == 42
    assert tags_to_bitmask([], base_mask=42) == 42


def test_vector_match_tuple_compatibility() -> None:
    # 1. 3-tuple unpacking compatibility
    match = VectorMatch("doc1", 0.95, "hello", parent_key="doc:parent")
    item_id, score, payload = match
    assert item_id == "doc1"
    assert score == 0.95
    assert payload == "hello"

    # 2. Indexing
    assert match[0] == "doc1"
    assert match[1] == 0.95
    assert match[2] == "hello"
    assert match[3] == "doc:parent"

    # 3. Attributes
    assert match.id == "doc1"
    assert match.score == 0.95
    assert match.payload == "hello"
    assert match.parent_key == "doc:parent"

    # 4. Equality with 3-tuple
    assert match == ("doc1", 0.95, "hello")


def test_markdown_chunker_short_document() -> None:
    chunker = MarkdownChunker(target_chunk_size=350, overlap=50)
    short_doc = "# Title\nThis is a short note."
    chunks = chunker.chunk_document("doc:test", short_doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc:test:chunk_0"
    assert chunks[0].content == short_doc
    assert chunks[0].parent_key == "doc:test"


def test_markdown_chunker_hierarchical_sections() -> None:
    chunker = MarkdownChunker(target_chunk_size=100, overlap=20)
    doc = """# System Architecture

## Storage Engine
KacheDB uses a Megaslab memory bump allocator to avoid jemalloc contention.
The Swiss Table hash map enables SIMD-accelerated linear probing.

## Network Protocol
RESP3 protocol parsers use zero-copy frame parsing directly from byte buffers.
Pipelines achieve sub-microsecond latency per command.

```rust
fn handle_connection() {
    println!("Zero copy");
}
```
"""
    chunks = chunker.chunk_document("doc:arch", doc)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.parent_key == "doc:arch"
        assert chunk.chunk_id.startswith("doc:arch:chunk_")
        assert chunk.token_count > 0


class FakeKacheClient:
    """Mock client for testing SemanticCache exact shortcut & parent expansion."""

    def __init__(self) -> None:
        self.kv_store: dict[str, str | bytes] = {}
        self.vectors: dict[str, tuple[list[float], str, int, str | None]] = {}

    def set(self, key: str, value: str | bytes, ex: int | None = None) -> bool:
        self.kv_store[key] = value
        return True

    def get(self, key: str) -> str | bytes | None:
        return self.kv_store.get(key)

    def vadd(
        self,
        index: str | bytes,
        item_id: str | bytes,
        vector: list[float] | tuple[float, ...] | bytes,
        *,
        payload: str | bytes | None = None,
        ex: int | None = None,
        tag_mask: int = 0,
        tags: list[str] | None = None,
        parent_key: str | bytes | None = None,
    ) -> bool:
        vec = list(vector) if isinstance(vector, (list, tuple)) else [0.0]
        p_str = str(payload or "")
        pk_str = str(parent_key) if parent_key is not None else None
        self.vectors[str(item_id)] = (vec, p_str, tag_mask, pk_str)
        return True

    def vsearch(
        self,
        index: str | bytes,
        query_vector: list[float] | tuple[float, ...] | bytes,
        *,
        top_k: int = 1,
        threshold: float = 0.0,
        filter_mask: int = 0,
        filter_tags: list[str] | None = None,
    ) -> list[VectorMatch]:
        results: list[VectorMatch] = []
        for item_id, (_vec, payload, tag_mask, parent_key) in self.vectors.items():
            if filter_mask != 0 and (tag_mask & filter_mask) != filter_mask:
                continue
            results.append(VectorMatch(item_id, 0.92, payload, parent_key=parent_key))
        return results[:top_k]

    def vdel(self, index: str | bytes, item_id: str | bytes) -> bool:
        self.vectors.pop(str(item_id), None)
        return True

    def vstats(self, index: str | bytes) -> dict[str, int]:
        return {"total_vectors": len(self.vectors)}


def test_semantic_cache_exact_first_shortcut() -> None:
    fake_client = FakeKacheClient()
    cache = SemanticCache(
        client=fake_client,  # type: ignore[arg-type]
        workspace="database",
        embedder=MockEmbedder(dimension=4),
    )

    # Store entry with exact shortcut
    cache.set("eviction_policy", "S3-FIFO with Megaslab")
    assert "doc:database:eviction_policy" in fake_client.kv_store

    # Exact query hits in KV store without needing vector similarity
    res = cache.get("eviction_policy", exact_first=True)
    assert res is not None
    assert res.is_exact is True
    assert res.similarity == 1.0
    assert res.value == "S3-FIFO with Megaslab"


def test_semantic_cache_parent_expansion() -> None:
    fake_client = FakeKacheClient()
    fake_client.kv_store["doc:parent_full"] = "Full 5,000-word architectural document"

    cache = SemanticCache(
        client=fake_client,  # type: ignore[arg-type]
        workspace="database",
        embedder=MockEmbedder(dimension=4),
    )

    # Store a chunk pointing to the parent
    cache.set(
        prompt="storage chunk 1",
        response="Chunk summary",
        parent_key="doc:parent_full",
        store_exact=False,
    )

    # Search with parent resolution
    res = cache.get("storage chunk 1", exact_first=False, resolve_parent=True)
    assert res is not None
    assert res.value == "Full 5,000-word architectural document"
    assert res.chunk == "Chunk summary"
    assert res.parent_key == "doc:parent_full"


def test_semantic_cache_filter_mask() -> None:
    fake_client = FakeKacheClient()
    cache = SemanticCache(
        client=fake_client,  # type: ignore[arg-type]
        workspace="database",
        embedder=MockEmbedder(dimension=4),
    )

    cache.set("doc_db", "DB content", tag_mask=0b01, store_exact=False)
    cache.set("doc_cf", "Cashflow content", tag_mask=0b10, store_exact=False)

    # Filter for 0b01 only
    res = cache.get("query", exact_first=False, filter_mask=0b01)
    assert res is not None
    assert res.key == "doc_db"

    # Filter for non-matching 0b100
    res_none = cache.get("query", exact_first=False, filter_mask=0b100)
    assert res_none is None
