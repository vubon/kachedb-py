"""
Unit & integration tests for LangChain & LlamaIndex KacheDB providers.
"""

from __future__ import annotations

import pytest

from kachedb import KacheClient
from kachedb.integrations.langchain import Generation, KacheDBCache, KacheDBSemanticCache
from kachedb.integrations.llamaindex import KacheDBIndexStore, KacheDBKVStore
from kachedb.semantic.embedders import MockEmbedder


@pytest.fixture
def client() -> KacheClient:
    return KacheClient()


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


def test_langchain_semantic_cache(client: KacheClient) -> None:
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


def test_llamaindex_kv_and_index_store(client: KacheClient) -> None:
    kvstore = KacheDBKVStore(client=client, namespace="test:llama:kv")
    index_store = KacheDBIndexStore(kvstore=kvstore)

    doc_data = {"id": "doc_123", "text": "KacheDB is an ultra-fast vector cache."}
    kvstore.put("doc_123", doc_data)

    retrieved = kvstore.get("doc_123")
    assert retrieved == doc_data

    # Index store
    index_meta = {"index_id": "idx_1", "type": "vector", "nodes": 100}
    index_store.put("idx_1", index_meta)
    assert index_store.get("idx_1") == index_meta

    # Delete
    assert index_store.delete("idx_1") is True
    assert index_store.get("idx_1") is None
