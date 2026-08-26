"""
Unit tests for KacheDB SGLang (RadixAttention) KV-cache integration.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kachedb.sglang import (  # noqa: E402
    KacheDBRadixAdapter,
    KacheDBSGLangConnector,
)
from kachedb.vllm import KacheDBConnector  # noqa: E402


def test_radix_node_hash_variable_length():
    """Verify deterministic 64-bit chained Blake2b hashing for arbitrary token slice lengths."""
    tokens_root = [101, 2054, 2003, 1037]  # 4 tokens
    tokens_child_a = [1024, 7592, 2005]  # 3 tokens
    tokens_child_b = [9999, 8888]  # 2 tokens

    h_root1 = KacheDBRadixAdapter.compute_node_hash(tokens_root, parent_hash=0)
    h_root2 = KacheDBRadixAdapter.compute_node_hash(tokens_root, parent_hash=0)
    assert h_root1 == h_root2
    assert isinstance(h_root1, int)
    assert h_root1 > 0

    # Child hashes chained to root
    h_child_a = KacheDBRadixAdapter.compute_node_hash(tokens_child_a, parent_hash=h_root1)
    h_child_b = KacheDBRadixAdapter.compute_node_hash(tokens_child_b, parent_hash=h_root1)
    assert h_child_a != h_child_b
    assert h_child_a != h_root1


def test_radix_adapter_prefix_tree_matching():
    """Verify multi-branch Radix tree prefix resolution."""
    adapter = KacheDBRadixAdapter()

    # Node 1 (Root): "You are an assistant"
    root_tokens = [1, 2, 3, 4]
    desc_root = adapter.register_node(
        node_id=1,
        token_ids=root_tokens,
        parent_hash=0,
        layer_offsets=[0, 1024],
    )

    # Node 2 (Branch A): "Write Rust code"
    branch_a_tokens = [10, 20, 30]
    adapter.register_node(
        node_id=2,
        token_ids=branch_a_tokens,
        parent_hash=desc_root.node_hash,
        layer_offsets=[2048, 3072],
    )

    # Test full match for Branch A
    prompt_a = [1, 2, 3, 4, 10, 20, 30]
    matched_len, nodes = adapter.find_matching_nodes(prompt_a)
    assert matched_len == 7
    assert len(nodes) == 2
    assert nodes[0].node_id == 1
    assert nodes[1].node_id == 2

    # Test partial match (only root tokens match)
    prompt_divergent = [1, 2, 3, 4, 99, 99]
    matched_len_div, nodes_div = adapter.find_matching_nodes(prompt_divergent)
    assert matched_len_div == 4
    assert len(nodes_div) == 1
    assert nodes_div[0].node_id == 1

    # Test complete miss
    prompt_miss = [999, 888]
    matched_len_miss, nodes_miss = adapter.find_matching_nodes(prompt_miss)
    assert matched_len_miss == 0
    assert len(nodes_miss) == 0


def test_sglang_connector_single_node_roundtrip():
    """Test offloading a Radix tree node's KV tensors and restoring with exact bit parity."""
    num_heads = 4
    head_dim = 64
    num_tokens = 15  # Variable length slice
    num_layers = 2
    dtype = torch.float16

    connector = KacheDBSGLangConnector(rank=0, local_rank=0, pool_size_mb=64)

    token_ids = list(range(100, 100 + num_tokens))
    k_tensors = [
        torch.randn((num_heads, num_tokens, head_dim), dtype=dtype) for _ in range(num_layers)
    ]
    v_tensors = [
        torch.randn((num_heads, num_tokens, head_dim), dtype=dtype) for _ in range(num_layers)
    ]

    # Offload node to KacheDB shared memory
    desc = connector.offload_node(
        node_id=1,
        token_ids=token_ids,
        k_tensors=k_tensors,
        v_tensors=v_tensors,
        parent_hash=0,
    )
    assert desc.node_id == 1
    assert desc.num_tokens == num_tokens
    assert len(desc.layer_offsets) == num_layers

    # Allocate target restoration buffers
    target_k = [
        torch.zeros((num_heads, num_tokens, head_dim), dtype=dtype) for _ in range(num_layers)
    ]
    target_v = [
        torch.zeros((num_heads, num_tokens, head_dim), dtype=dtype) for _ in range(num_layers)
    ]

    # Restore from KacheDB
    matched_count, is_hit = connector.restore_prefix(
        prompt_tokens=token_ids,
        target_k_buffers=target_k,
        target_v_buffers=target_v,
    )

    assert is_hit is True
    assert matched_count == num_tokens

    # Verify bit-exact tensor equality
    for l_idx in range(num_layers):
        assert torch.equal(target_k[l_idx], k_tensors[l_idx])
        assert torch.equal(target_v[l_idx], v_tensors[l_idx])

    connector.close()


def test_sglang_connector_multi_branch_restore():
    """Test multi-node sequential restoration across tree hierarchy."""
    num_heads = 2
    head_dim = 32
    num_layers = 2
    dtype = torch.float16

    connector = KacheDBSGLangConnector(rank=0, local_rank=0, pool_size_mb=64)

    # Node 1: Root (10 tokens)
    root_tokens = list(range(10))
    k1 = [torch.randn((num_heads, 10, head_dim), dtype=dtype) for _ in range(num_layers)]
    v1 = [torch.randn((num_heads, 10, head_dim), dtype=dtype) for _ in range(num_layers)]
    desc1 = connector.offload_node(
        node_id=1,
        token_ids=root_tokens,
        k_tensors=k1,
        v_tensors=v1,
        parent_hash=0,
    )

    # Node 2: Child (20 tokens)
    child_tokens = list(range(10, 30))
    k2 = [torch.randn((num_heads, 20, head_dim), dtype=dtype) for _ in range(num_layers)]
    v2 = [torch.randn((num_heads, 20, head_dim), dtype=dtype) for _ in range(num_layers)]
    connector.offload_node(
        node_id=2,
        token_ids=child_tokens,
        k_tensors=k2,
        v_tensors=v2,
        parent_hash=desc1.node_hash,
    )

    # Restore 30 total tokens (root + child)
    full_prompt = list(range(30))
    target_k = [torch.zeros((num_heads, 30, head_dim), dtype=dtype) for _ in range(num_layers)]
    target_v = [torch.zeros((num_heads, 30, head_dim), dtype=dtype) for _ in range(num_layers)]

    matched_count, is_hit = connector.restore_prefix(
        prompt_tokens=full_prompt,
        target_k_buffers=target_k,
        target_v_buffers=target_v,
    )

    assert is_hit is True
    assert matched_count == 30

    # Verify root part
    for l_idx in range(num_layers):
        assert torch.equal(target_k[l_idx][:, :10, :], k1[l_idx])
        assert torch.equal(target_v[l_idx][:, :10, :], v1[l_idx])
        # Verify child part
        assert torch.equal(target_k[l_idx][:, 10:30, :], k2[l_idx])
        assert torch.equal(target_v[l_idx][:, 10:30, :], v2[l_idx])

    connector.close()


def test_vllm_sglang_cross_engine_coexistence():
    """Verify that both vLLM and SGLang connectors coexist in the same process seamlessly."""
    vllm_conn = KacheDBConnector(rank=0, local_rank=0, pool_size_mb=64)
    sglang_conn = KacheDBSGLangConnector(rank=0, local_rank=0, pool_size_mb=64)

    assert vllm_conn.block_size == 16
    assert sglang_conn.pool_size_mb == 64

    vllm_conn.close()
    sglang_conn.close()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int8])
def test_sglang_connector_all_precisions(dtype: torch.dtype):
    """Verify that FP16, BF16 (LLaMA default), FP32, and INT8 serialize/restore perfectly."""
    num_heads = 2
    head_dim = 64
    num_tokens = 8
    num_layers = 2

    connector = KacheDBSGLangConnector(rank=0, local_rank=0, pool_size_mb=64)

    token_ids = list(range(200, 200 + num_tokens))
    k_tensors = [
        torch.randint(-100, 100, (num_heads, num_tokens, head_dim)).to(dtype)
        if dtype == torch.int8
        else torch.randn((num_heads, num_tokens, head_dim), dtype=dtype)
        for _ in range(num_layers)
    ]
    v_tensors = [
        torch.randint(-100, 100, (num_heads, num_tokens, head_dim)).to(dtype)
        if dtype == torch.int8
        else torch.randn((num_heads, num_tokens, head_dim), dtype=dtype)
        for _ in range(num_layers)
    ]

    desc = connector.offload_node(
        node_id=10,
        token_ids=token_ids,
        k_tensors=k_tensors,
        v_tensors=v_tensors,
        parent_hash=0,
    )
    assert desc.node_id == 10

    target_k = [
        torch.zeros((num_heads, num_tokens, head_dim), dtype=dtype) for _ in range(num_layers)
    ]
    target_v = [
        torch.zeros((num_heads, num_tokens, head_dim), dtype=dtype) for _ in range(num_layers)
    ]

    matched, is_hit = connector.restore_prefix(
        prompt_tokens=token_ids,
        target_k_buffers=target_k,
        target_v_buffers=target_v,
    )

    assert is_hit is True
    assert matched == num_tokens

    for l_idx in range(num_layers):
        assert torch.equal(target_k[l_idx], k_tensors[l_idx])
        assert torch.equal(target_v[l_idx], v_tensors[l_idx])

    connector.close()
