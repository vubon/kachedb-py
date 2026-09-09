"""
Unit and integration tests for KacheDB vLLM connector and prefix cache.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from kachedb.vllm import (
    KacheDBConnector,
    KacheDBMemoryManager,
    KacheDBPrefixCache,
)

torch = pytest.importorskip("torch")


class TestKacheDBPrefixCache:
    """Test suite for token chunk prefix hashing and lookup."""

    def test_block_hash_determinism(self) -> None:
        cache = KacheDBPrefixCache(block_size=16)
        tokens = [100, 200, 300, 400]
        h1 = cache.compute_block_hash(tokens, parent_hash=0)
        h2 = cache.compute_block_hash(tokens, parent_hash=0)
        assert h1 == h2
        assert isinstance(h1, int)
        assert h1 > 0

    def test_block_hash_chaining(self) -> None:
        cache = KacheDBPrefixCache(block_size=16)
        chunk1 = list(range(16))
        chunk2 = list(range(16, 32))

        h1 = cache.compute_block_hash(chunk1, parent_hash=0)
        h2_chained = cache.compute_block_hash(chunk2, parent_hash=h1)
        h2_unchained = cache.compute_block_hash(chunk2, parent_hash=0)

        # Chained hash MUST differ from unchained hash
        assert h2_chained != h2_unchained

    def test_sequence_hashes(self) -> None:
        cache = KacheDBPrefixCache(block_size=16)
        prompt = list(range(48))  # 3 full blocks
        hashes = cache.compute_sequence_hashes(prompt)
        assert len(hashes) == 3
        assert len(set(hashes)) == 3

    def test_find_longest_prefix(self) -> None:
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

    def test_write_and_read_paged_block(self) -> None:
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

    def test_ring_buffer_wrap(self) -> None:
        # 1 MB pool
        mm = KacheDBMemoryManager(core_id=0, pool_size_mb=1)
        # Advance write head near the end of pool
        mm._write_head = mm.pool_size_bytes - 100
        k_tensor = torch.randn((2, 16, 16), dtype=torch.float16)
        v_tensor = torch.randn((2, 16, 16), dtype=torch.float16)
        # Writing now wraps ring buffer to offset 0
        offset, _, _ = mm.write_paged_block(
            key_tensor=k_tensor,
            val_tensor=v_tensor,
            layer_idx=0,
            seq_prefix_hash=0x999,
            block_id=0,
        )
        assert offset == 0


class TestKacheDBConnector:
    """Test suite for vLLM KacheDBConnector lifecycle and caching."""

    def test_connector_send_and_recv_cycle(self) -> None:
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

    def test_connector_seq_data_and_empty(self) -> None:
        connector = KacheDBConnector(rank=0, local_rank=0, block_size=16, pool_size_mb=64)

        # Empty tokens in send and recv
        empty_input = types.SimpleNamespace()
        connector.send_kv_caches_and_hidden_states(None, empty_input, [])
        _, is_hit = connector.recv_kv_caches_and_hidden_states(None, empty_input, [])
        assert is_hit is False

        # seq_data attribute
        seq_input = types.SimpleNamespace(
            seq_data=types.SimpleNamespace(get_token_ids=lambda: list(range(32)))
        )
        kv_shape = (2, 2, 4, 16, 32)
        kv_caches = [torch.randn(kv_shape, dtype=torch.float16)]
        connector.send_kv_caches_and_hidden_states(None, seq_input, kv_caches)

        # Sending again skips already registered blocks
        connector.send_kv_caches_and_hidden_states(None, seq_input, kv_caches)

        # Cache miss with completely different tokens
        miss_input = types.SimpleNamespace(input_tokens=[99999, 88888])
        _, hit = connector.recv_kv_caches_and_hidden_states(None, miss_input, kv_caches)
        assert hit is False

        # Recv using seq_data
        recv_kv = [torch.zeros(kv_shape, dtype=torch.float16)]
        _, hit = connector.recv_kv_caches_and_hidden_states(None, seq_input, recv_kv)
        assert hit is True

        connector.close()

    def test_connector_tuple_layer_kv(self) -> None:
        from unittest.mock import MagicMock

        connector = KacheDBConnector(rank=0, local_rank=0, block_size=16, pool_size_mb=64)

        # 3D tensor tuple: [num_heads=4, seq_len=32, head_dim=32]
        k_3d = torch.randn((4, 32, 32), dtype=torch.float16)
        v_3d = torch.randn((4, 32, 32), dtype=torch.float16)
        kv_caches = [(k_3d, v_3d)]

        tokens = list(range(32))
        model_input = types.SimpleNamespace(input_tokens=tokens)
        connector.send_kv_caches_and_hidden_states(None, model_input, kv_caches)

        # 2D indexed tuple/list: k_t[b_idx]
        k_blocks = torch.randn((2, 4, 16, 32), dtype=torch.float16)
        v_blocks = torch.randn((2, 4, 16, 32), dtype=torch.float16)
        connector.send_kv_caches_and_hidden_states(
            None,
            types.SimpleNamespace(input_tokens=list(range(50, 82))),
            [(k_blocks, v_blocks)],
        )

        # Fallback else branch: layer_kv[0][b_idx]
        class CustomKV:
            def __init__(self, k: Any, v: Any) -> None:
                self.k = k
                self.v = v

            def __getitem__(self, idx: int) -> Any:
                return [self.k, self.v][idx]

        connector.send_kv_caches_and_hidden_states(
            None,
            types.SimpleNamespace(input_tokens=list(range(100, 132))),
            [CustomKV(k_blocks, v_blocks)],
        )

        # Recv with mocked ndim != 5 to exercise else branch
        fake_layer_kv = MagicMock()
        fake_layer_kv.shape = (2, 2, 4, 16, 32)
        fake_layer_kv.ndim = 4
        fake_layer_kv.device = "cpu"
        fake_k_block = MagicMock()
        fake_v_block = MagicMock()
        fake_layer_kv.__getitem__.side_effect = lambda idx: [fake_k_block, fake_v_block][idx]

        _, hit = connector.recv_kv_caches_and_hidden_states(None, model_input, [fake_layer_kv])
        assert hit is True

        connector.close()
