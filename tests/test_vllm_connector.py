"""
Unit and integration tests for KacheDB vLLM connector and prefix cache.
"""

from __future__ import annotations

import types

import torch

from kachedb.vllm import (
    KacheDBConnector,
    KacheDBMemoryManager,
    KacheDBPrefixCache,
)


class TestKacheDBPrefixCache:
    """Test suite for token chunk prefix hashing and lookup."""

    def test_block_hash_determinism(self):
        cache = KacheDBPrefixCache(block_size=16)
        tokens = [100, 200, 300, 400]
        h1 = cache.compute_block_hash(tokens, parent_hash=0)
        h2 = cache.compute_block_hash(tokens, parent_hash=0)
        assert h1 == h2
        assert isinstance(h1, int)
        assert h1 > 0

    def test_block_hash_chaining(self):
        cache = KacheDBPrefixCache(block_size=16)
        chunk1 = list(range(16))
        chunk2 = list(range(16, 32))

        h1 = cache.compute_block_hash(chunk1, parent_hash=0)
        h2_chained = cache.compute_block_hash(chunk2, parent_hash=h1)
        h2_unchained = cache.compute_block_hash(chunk2, parent_hash=0)

        # Chained hash MUST differ from unchained hash
        assert h2_chained != h2_unchained

    def test_sequence_hashes(self):
        cache = KacheDBPrefixCache(block_size=16)
        prompt = list(range(48))  # 3 full blocks
        hashes = cache.compute_sequence_hashes(prompt)
        assert len(hashes) == 3
        assert len(set(hashes)) == 3

    def test_find_longest_prefix(self):
        cache = KacheDBPrefixCache(block_size=16)
        prompt_system = list(range(32))  # 2 blocks
        hashes = cache.compute_sequence_hashes(prompt_system)

        # Register both blocks
        cache.register_block(hashes[0], block_id=0, num_tokens=16, metadata={"layer": 0})
        cache.register_block(hashes[1], block_id=1, num_tokens=16, metadata={"layer": 0})

        # Match prompt with shared prefix + new tokens
        extended_prompt = [*prompt_system, 999, 1000, 1001]
        matched_tokens, blocks = cache.find_longest_prefix(extended_prompt)

        assert matched_tokens == 32
        assert len(blocks) == 2
        assert blocks[0][0] == 0  # block_id 0
        assert blocks[1][0] == 1  # block_id 1


class TestKacheDBMemoryManager:
    """Test suite for DMA and zero-copy block swapping."""

    def test_write_and_read_paged_block(self):
        mm = KacheDBMemoryManager(core_id=0, pool_size_mb=64)

        # Shape: [num_heads=8, block_size=16, head_dim=64]
        shape = (8, 16, 64)
        k_tensor = torch.randn(shape, dtype=torch.float16)
        v_tensor = torch.randn(shape, dtype=torch.float16)

        offset, _size, desc = mm.write_paged_block(
            key_tensor=k_tensor,
            val_tensor=v_tensor,
            layer_idx=3,
            seq_prefix_hash=0x12345678,
            block_id=5,
        )

        assert offset >= 0
        assert desc.layer_id == 3
        assert desc.num_heads == 8
        assert desc.block_size == 16
        assert desc.head_dim == 64

        # Read back zero-copy
        k_loaded, v_loaded = mm.read_paged_block(offset=offset, shape=shape)

        assert torch.allclose(k_tensor, k_loaded)
        assert torch.allclose(v_tensor, v_loaded)


class TestKacheDBConnector:
    """Test suite for vLLM KacheDBConnector lifecycle and caching."""

    def test_connector_send_and_recv_cycle(self):
        connector = KacheDBConnector(rank=0, local_rank=0, block_size=16, pool_size_mb=64)

        # Simulated PagedAttention tensor for 2 layers
        # Shape: [num_blocks=4, 2, num_heads=8, block_size=16, head_dim=64]
        kv_shape = (4, 2, 8, 16, 64)
        kv_caches = [
            torch.randn(kv_shape, dtype=torch.float16),
            torch.randn(kv_shape, dtype=torch.float16),
        ]

        # 32 tokens = 2 blocks
        prompt_tokens = list(range(100, 132))
        model_input = types.SimpleNamespace(input_tokens=prompt_tokens)

        # 1. Offload KV caches into KacheDB
        connector.send_kv_caches_and_hidden_states(
            model_executable=None,
            model_input=model_input,
            kv_caches=kv_caches,
        )

        # 2. Simulate subsequent request with matching prefix + new query
        new_prompt = [*prompt_tokens, 999, 1000]
        new_input = types.SimpleNamespace(input_tokens=new_prompt)

        # Target KV cache to receive restored blocks
        recv_kv_caches = [
            torch.zeros(kv_shape, dtype=torch.float16),
            torch.zeros(kv_shape, dtype=torch.float16),
        ]

        _, is_hit = connector.recv_kv_caches_and_hidden_states(
            model_executable=None,
            model_input=new_input,
            kv_caches=recv_kv_caches,
        )

        assert is_hit is True
        # Verify that blocks 0 and 1 were restored losslessly
        for layer in range(2):
            assert torch.allclose(kv_caches[layer][0], recv_kv_caches[layer][0])
            assert torch.allclose(kv_caches[layer][1], recv_kv_caches[layer][1])

        connector.close()
