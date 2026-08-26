"""
KacheDB zero-copy KV-cache connector for SGLang (RadixAttention).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from ..dma import KacheDBMemoryManager
from .radix_adapter import KacheDBRadixAdapter, RadixNodeDescriptor

logger = logging.getLogger("kachedb.sglang")


class KacheDBSGLangConnector:
    """High-performance zero-copy KV cache offloader and restorer for SGLang.

    Coordinates with SGLang's dynamic RadixTree memory pool to offload evicted
    tree nodes into KacheDB's POSIX shared memory Megaslabs (/dev/shm) and restore
    them with sub-millisecond latency.
    """

    def __init__(
        self,
        rank: int = 0,
        local_rank: int = 0,
        pool_size_mb: int = 1024,
        host: str = "127.0.0.1",
        port: int = 6379,
    ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self.host = host
        self.port = port
        self.pool_size_mb = pool_size_mb

        self.radix_adapter = KacheDBRadixAdapter()
        self.memory_manager = KacheDBMemoryManager(core_id=local_rank, pool_size_mb=pool_size_mb)

        logger.info(
            "KacheDBSGLangConnector initialized for worker rank %d (core_id=%d, pool_size=%dMB)",
            rank,
            local_rank,
            pool_size_mb,
        )

    def offload_node(
        self,
        node_id: int,
        token_ids: list[int],
        k_tensors: list[torch.Tensor],
        v_tensors: list[torch.Tensor],
        parent_hash: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> RadixNodeDescriptor:
        """Offload an evicted Radix tree node's multi-layer KV tensors to KacheDB shared memory.

        Parameters
        ----------
        node_id : int
            SGLang internal tree node ID.
        token_ids : list[int]
            Token IDs contained in this Radix node.
        k_tensors : list[torch.Tensor]
            Per-layer Key tensors for this node's tokens.
        v_tensors : list[torch.Tensor]
            Per-layer Value tensors for this node's tokens.
        parent_hash : int
            Chained 64-bit Blake2b hash of the parent tree node.
        metadata : dict[str, Any], optional
            Extra metadata tags.

        Returns
        -------
        RadixNodeDescriptor
            Registered metadata descriptor with memory offsets.
        """
        node_hash = self.radix_adapter.compute_node_hash(token_ids, parent_hash=parent_hash)
        num_layers = len(k_tensors)
        layer_offsets: list[int] = []

        for l_idx in range(num_layers):
            k_layer = k_tensors[l_idx]
            v_layer = v_tensors[l_idx]

            offset, _, _ = self.memory_manager.write_paged_block(
                key_tensor=k_layer,
                val_tensor=v_layer,
                layer_idx=l_idx,
                seq_prefix_hash=node_hash,
                block_id=node_id,
            )
            layer_offsets.append(offset)

        desc = self.radix_adapter.register_node(
            node_id=node_id,
            token_ids=token_ids,
            parent_hash=parent_hash,
            layer_offsets=layer_offsets,
            metadata=metadata or {},
        )

        logger.debug(
            "Offloaded SGLang Radix node %d (%d tokens, hash=%x) across %d layers",
            node_id,
            len(token_ids),
            node_hash,
            num_layers,
        )
        return desc

    def restore_prefix(
        self,
        prompt_tokens: list[int],
        target_k_buffers: list[torch.Tensor],
        target_v_buffers: list[torch.Tensor],
    ) -> tuple[int, bool]:
        """Look up matching cached Radix nodes and restore KV tensors via zero-copy DMA.

        Parameters
        ----------
        prompt_tokens : list[int]
            Incoming prompt token IDs to match.
        target_k_buffers : list[torch.Tensor]
            Target memory buffers for Key tensors (per layer).
        target_v_buffers : list[torch.Tensor]
            Target memory buffers for Value tensors (per layer).

        Returns
        -------
        tuple[int, bool]
            (matched_tokens_count, is_cache_hit)
        """
        matched_tokens, matched_nodes = self.radix_adapter.find_matching_nodes(prompt_tokens)
        if matched_tokens == 0 or not matched_nodes:
            return 0, False

        num_layers = len(target_k_buffers)
        token_offset = 0

        for desc in matched_nodes:
            n_tokens = desc.num_tokens
            for l_idx in range(num_layers):
                if l_idx >= len(desc.layer_offsets):
                    continue

                target_k = target_k_buffers[l_idx]
                target_v = target_v_buffers[l_idx]

                # Expected shape for this node slice: (num_heads, num_tokens, head_dim)
                shape = (
                    target_k.shape[0],
                    n_tokens,
                    target_k.shape[2],
                )

                k_block, v_block = self.memory_manager.read_paged_block(
                    offset=desc.layer_offsets[l_idx],
                    shape=shape,
                    device=str(target_k.device),
                    dtype=target_k.dtype,
                )

                # Slice-copy into target buffer
                target_k[:, token_offset : token_offset + n_tokens, :].copy_(
                    k_block, non_blocking=True
                )
                target_v[:, token_offset : token_offset + n_tokens, :].copy_(
                    v_block, non_blocking=True
                )

            token_offset += n_tokens

        logger.debug(
            "Restored %d SGLang Radix prefix tokens across %d nodes",
            matched_tokens,
            len(matched_nodes),
        )
        return matched_tokens, True

    def close(self) -> None:
        """Release connector resources."""
        logger.info("Closing KacheDBSGLangConnector for rank %d", self.rank)
