"""
KacheDB KV-cache connector for vLLM distributed inference engine.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from .dma import KacheDBMemoryManager
from .prefix_cache import KacheDBPrefixCache

logger = logging.getLogger("kachedb.vllm")


# Dynamically attempt import of vLLM base connector
try:
    from vllm.distributed.kv_transfer.kv_connector.base import (  # type: ignore[import-not-found]
        KVConnectorBase,
    )
except ImportError:
    # Graceful fallback base class when vLLM is not installed in the environment
    class KVConnectorBase:  # type: ignore[no-redef]
        def __init__(self, rank: int, local_rank: int, config: Any):
            self.rank = rank
            self.local_rank = local_rank
            self.config = config


class KacheDBConnector(KVConnectorBase):  # type: ignore[misc]
    """Zero-copy KacheDB KV-cache connector for vLLM.

    Provides microsecond-level prompt prefix cache restoration and PagedAttention
    block offloading via POSIX Shared Memory (/dev/shm).

    Usage in vLLM CLI::

        vllm serve meta-llama/Meta-Llama-3-8B-Instruct \\
            --kv-transfer-config \\
            '{"kv_connector": "kachedb.vllm.KacheDBConnector", "kv_role": "kv_both"}'
    """

    def __init__(
        self,
        rank: int = 0,
        local_rank: int = 0,
        config: Any | None = None,
        block_size: int = 16,
        pool_size_mb: int = 512,
    ):
        super().__init__(rank, local_rank, config)
        self.rank = rank
        self.local_rank = local_rank
        self.block_size = block_size

        # Initialize prefix cache & memory manager for this rank's core
        self.prefix_cache = KacheDBPrefixCache(block_size=block_size)
        self.memory_manager = KacheDBMemoryManager(core_id=local_rank, pool_size_mb=pool_size_mb)

        logger.info(
            "KacheDBConnector initialized for worker rank %d (core_id=%d, block_size=%d)",
            rank,
            local_rank,
            block_size,
        )

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: Any,
        model_input: Any,
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: list[torch.Tensor] | None = None,
    ) -> None:
        """Offload computed PagedAttention KV-cache blocks into KacheDB shared memory.

        Parameters
        ----------
        model_executable : Any
            vLLM model runner instance.
        model_input : Any
            Model input metadata containing prompt token IDs and slot mappings.
        kv_caches : List[torch.Tensor]
            List of multi-layer PagedAttention KV tensors across all layers.
        hidden_or_intermediate_states : Optional[List[torch.Tensor]]
            Optional intermediate states.
        """
        # Extract prompt tokens from input if available
        prompt_tokens: list[int] = getattr(model_input, "input_tokens", [])
        if not prompt_tokens and hasattr(model_input, "seq_data"):
            # vLLM SequenceData extraction
            prompt_tokens = list(model_input.seq_data.get_token_ids())

        if not prompt_tokens:
            return

        hashes = self.prefix_cache.compute_sequence_hashes(prompt_tokens)
        num_layers = len(kv_caches)

        for b_idx, prefix_h in enumerate(hashes):
            # Check if block is already registered in index
            if self.prefix_cache.lookup_block(prefix_h) is not None:
                continue

            # Offload each layer's KV block for this sequence chunk
            layer_offsets = []
            for l_idx in range(num_layers):
                layer_kv = kv_caches[l_idx]
                if hasattr(layer_kv, "ndim") and layer_kv.ndim == 5:
                    k_block = layer_kv[b_idx, 0]
                    v_block = layer_kv[b_idx, 1]
                elif isinstance(layer_kv, (list, tuple)):
                    k_t = layer_kv[0]
                    v_t = layer_kv[1]
                    if k_t.ndim == 3:  # [num_heads, seq_len, head_dim]
                        start_pos = b_idx * self.block_size
                        end_pos = min(k_t.shape[1], (b_idx + 1) * self.block_size)
                        k_block = k_t[:, start_pos:end_pos, :]
                        v_block = v_t[:, start_pos:end_pos, :]
                    else:
                        k_block = k_t[b_idx]
                        v_block = v_t[b_idx]
                else:
                    k_block = layer_kv[0][b_idx]
                    v_block = layer_kv[1][b_idx]

                offset, _, _ = self.memory_manager.write_paged_block(
                    key_tensor=k_block,
                    val_tensor=v_block,
                    layer_idx=l_idx,
                    seq_prefix_hash=prefix_h,
                    block_id=b_idx,
                )
                layer_offsets.append(offset)

            # Register block in prefix index
            self.prefix_cache.register_block(
                prefix_hash=prefix_h,
                block_id=b_idx,
                num_tokens=self.block_size,
                metadata={"layer_offsets": layer_offsets, "num_layers": num_layers},
            )

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: Any,
        model_input: Any,
        kv_caches: list[torch.Tensor],
    ) -> tuple[torch.Tensor | None, bool]:
        """Query KacheDB and restore matching cached prefix blocks into GPU HBM.

        Returns
        -------
        Tuple[Optional[torch.Tensor], bool]
            (matched_tokens_or_states, is_cache_hit)
        """
        prompt_tokens: list[int] = getattr(model_input, "input_tokens", [])
        if not prompt_tokens and hasattr(model_input, "seq_data"):
            prompt_tokens = list(model_input.seq_data.get_token_ids())

        if not prompt_tokens:
            return None, False

        matched_tokens, matched_blocks = self.prefix_cache.find_longest_prefix(prompt_tokens)
        if matched_tokens == 0:
            return None, False

        num_layers = len(kv_caches)
        # Restore matched blocks into GPU PagedAttention buffers
        for b_idx, (_, _, meta) in enumerate(matched_blocks):
            layer_offsets = meta.get("layer_offsets", [])
            for l_idx in range(num_layers):
                if l_idx >= len(layer_offsets):
                    continue
                layer_kv = kv_caches[l_idx]
                target_shape = (
                    layer_kv.shape[2],
                    layer_kv.shape[3],
                    layer_kv.shape[4],
                )
                k_block, v_block = self.memory_manager.read_paged_block(
                    offset=layer_offsets[l_idx],
                    shape=target_shape,
                    device=str(layer_kv.device),
                )
                if layer_kv.ndim == 5:
                    layer_kv[b_idx, 0].copy_(k_block, non_blocking=True)
                    layer_kv[b_idx, 1].copy_(v_block, non_blocking=True)
                else:
                    layer_kv[0][b_idx].copy_(k_block, non_blocking=True)
                    layer_kv[1][b_idx].copy_(v_block, non_blocking=True)

        logger.debug(
            "Restored %d prefix tokens (%d blocks) from KacheDB",
            matched_tokens,
            len(matched_blocks),
        )
        return None, True

    def close(self) -> None:
        """Clean up connector resources."""
        logger.info("Closing KacheDBConnector for rank %d", self.rank)
