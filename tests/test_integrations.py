from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest

from kachedb.integrations.langchain import Generation, KacheDBCache, KacheDBSemanticCache
from kachedb.integrations.llamaindex import KacheDBIndexStore, KacheDBKVStore
from kachedb.semantic.embedders import MockEmbedder

if TYPE_CHECKING:
    from kachedb import KacheClient


class InMemoryKacheClient:
    """Lightweight in-memory mock client for offline integration tests."""

    def __init__(self) -> None:
        self._kv: dict[str, bytes] = {}
        self._vectors: dict[str, dict[str, tuple[list[float], str]]] = {}

    def get(self, key: str) -> bytes | None:
        return self._kv.get(key)

    def set(self, key: str, value: Any, ex: int | None = None, px: int | None = None) -> bool:
        if isinstance(value, str):
            val_bytes = value.encode("utf-8")
        elif isinstance(value, bytes):
            val_bytes = value
        else:
            val_bytes = str(value).encode("utf-8")
        self._kv[key] = val_bytes
        return True

    def pipeline(self) -> Any:
        class MockPipe:
            def __init__(self, parent: Any) -> None:
                self.parent = parent
                self.cmds: list[tuple[str, Any]] = []

            def set(self, k: str, v: Any) -> MockPipe:
                self.cmds.append((k, v))
                return self

            def execute(self) -> list[Any]:
                for k, v in self.cmds:
                    self.parent.set(k, v)
                return [True] * len(self.cmds)

        return MockPipe(self)

    def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self._kv:
                del self._kv[k]
                count += 1
        return count

    def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if k in self._kv)

    def vadd(
        self,
        index: str,
        item_id: str,
        vector: list[float],
        payload: str = "",
        ex: int | None = None,
    ) -> bool:
        if index not in self._vectors:
            self._vectors[index] = {}
        self._vectors[index][item_id] = (vector, payload)
        return True

    def vsearch(
        self,
        index: str,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[bytes, float, bytes]]:
        if index not in self._vectors:
            return []
        results: list[tuple[bytes, float, bytes]] = []
        for item_id, (v, payload) in self._vectors[index].items():
            dot = sum(a * b for a, b in zip(query_vector, v, strict=False))
            norm_q = math.sqrt(sum(a * a for a in query_vector)) or 1.0
            norm_v = math.sqrt(sum(b * b for b in v)) or 1.0
            sim = dot / (norm_q * norm_v)
            if sim >= threshold:
                results.append((item_id.encode("utf-8"), sim, payload.encode("utf-8")))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


@pytest.fixture
def client() -> Any:
    return InMemoryKacheClient()


def test_langchain_exact_cache(client: KacheClient) -> None:
    cache = KacheDBCache(client=client, namespace="test:lc:exact:fresh")
    prompt = "Translate hello to French"
    llm_string = "gpt-4o"

    # Ensure key is clear
    client.delete(cache._key(prompt, llm_string))

    # Initially None
    assert cache.lookup(prompt, llm_string) is None

    # Update
    cache.update(prompt, llm_string, [Generation(text="Bonjour")])

    # Lookup hit
    res = cache.lookup(prompt, llm_string)
    assert res is not None
    assert len(res) == 1
    assert res[0].text == "Bonjour"

    # Non-JSON cached fallback
    client.set(cache._key("raw", "model"), "plain text completion")
    raw_res = cache.lookup("raw", "model")
    assert raw_res is not None
    assert raw_res[0].text == "plain text completion"

    # Clear
    cache.clear()


def test_langchain_semantic_cache(client: KacheClient, monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = MockEmbedder(dimension=64)
    cache = KacheDBSemanticCache(
        client=client,
        index_name="test:lc:semantic",
        embedder=embedder,
        similarity_threshold=0.80,
    )

    prompt = "What is the capital of France?"
    llm_string = "gpt-4o"

    cache.update(prompt, llm_string, [Generation(text="The capital of France is Paris.")])

    # Query with exact or similar prompt
    res = cache.lookup(prompt, llm_string)
    assert res is not None
    assert "Paris" in res[0].text

    # Cache miss
    miss = cache.lookup("Completely unknown question about rocket propulsion", "gpt-4o")
    assert miss is None

    # Clear
    cache.clear()

    # Auto-selected embedder constructor branch
    from unittest.mock import MagicMock

    mock_kc = MagicMock()
    auto_cache = KacheDBSemanticCache(client=mock_kc, embedder=None)
    assert auto_cache.semantic_cache.embedder is not None


def test_llamaindex_kv_and_index_store(
    client: KacheClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    kvstore = KacheDBKVStore(client=client, namespace="test:llama:kv")
    index_store = KacheDBIndexStore(kvstore=kvstore)

    doc_data = {"id": "doc_123", "text": "KacheDB is an ultra-fast vector cache."}
    kvstore.put("doc_123", doc_data)

    retrieved = kvstore.get("doc_123")
    assert retrieved == doc_data

    # put_all
    kvstore.put_all([("doc_2", {"a": 1}), ("doc_3", {"b": 2})])
    assert kvstore.get("doc_2") == {"a": 1}
    assert kvstore.get("doc_3") == {"b": 2}

    # Corrupt / non-JSON value returns None
    client.set(kvstore._format_key("corrupt", "data"), "invalid json {{{{")
    assert kvstore.get("corrupt") is None

    # Index store
    index_meta = {"index_id": "idx_1", "type": "vector", "nodes": 100}
    index_store.put("idx_1", index_meta)
    assert index_store.get("idx_1") == index_meta

    # Delete
    assert index_store.delete("idx_1") is True
    assert index_store.get("idx_1") is None

    # Default constructors with None
    from unittest.mock import MagicMock

    monkeypatch.setattr("kachedb.integrations.llamaindex.KacheClient", MagicMock)
    default_store = KacheDBIndexStore()
    assert default_store.kvstore is not None
