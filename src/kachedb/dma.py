"""
Asynchronous DMA and zero-copy shared memory manager for LLM KV-cache blocks.
"""

from __future__ import annotations

import ctypes
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from .descriptor import (
    TENSOR_DESCRIPTOR_MAGIC,
    TensorBlockDescriptor,
    TensorCodec,
)

logger = logging.getLogger("kachedb.dma")


class KacheDBMemoryManager:
    """Manages zero-copy shared memory slots and host-to-device transfers."""

    def __init__(self, core_id: int = 0, pool_size_mb: int = 512) -> None:
        self.core_id = core_id
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self.shm_path = f"/dev/shm/kachedb_{core_id}"

        # Initialize in-memory backing store (or mmap on Linux)
        self._buf = bytearray(self.pool_size_bytes)
        self._write_head = 0

    def write_paged_block(
        self,
        key_tensor: torch.Tensor,
        val_tensor: torch.Tensor,
        layer_idx: int,
        seq_prefix_hash: int,
        block_id: int,
    ) -> tuple[int, int, TensorBlockDescriptor]:
        """Write a PagedAttention Key/Value block into shared memory with descriptor header.

        Parameters
        ----------
        key_tensor : torch.Tensor
            Key tensor for the block (shape: [num_heads, block_size, head_dim]).
        val_tensor : torch.Tensor
            Value tensor for the block (shape: [num_heads, block_size, head_dim]).
        layer_idx : int
            Model layer index.
        seq_prefix_hash : int
            64-bit chained token sequence hash.
        block_id : int
            PagedAttention physical block ID.

        Returns
        -------
        tuple[int, int, TensorBlockDescriptor]
            (byte_offset, total_bytes, descriptor)
        """
        k_bytes, tensor_dtype, _ = TensorCodec.serialize_tensor(key_tensor)
        v_bytes, _, _ = TensorCodec.serialize_tensor(val_tensor)
        total_payload = len(k_bytes) + len(v_bytes)

        # Allocate slot in ring buffer
        desc_size = ctypes.sizeof(TensorBlockDescriptor)
        total_slot_size = desc_size + total_payload

        offset = self._write_head
        if offset + total_slot_size > self.pool_size_bytes:
            offset = 0  # Wrap ring
            self._write_head = 0

        # Construct 64-byte descriptor
        desc = TensorBlockDescriptor(
            magic=TENSOR_DESCRIPTOR_MAGIC,
            layer_id=layer_idx,
            num_layers=1,
            block_size=key_tensor.shape[1],
            num_heads=key_tensor.shape[0],
            head_dim=key_tensor.shape[2],
            dtype=tensor_dtype,
            sequence_prefix_hash=seq_prefix_hash,
            payload_bytes=total_payload,
        )

        desc_bytes = bytes(desc)
        self._buf[offset : offset + desc_size] = desc_bytes
        self._buf[offset + desc_size : offset + desc_size + len(k_bytes)] = k_bytes
        self._buf[offset + desc_size + len(k_bytes) : offset + total_slot_size] = v_bytes

        self._write_head += total_slot_size
        return offset, total_slot_size, desc

    def read_paged_block(
        self,
        offset: int,
        shape: tuple[int, ...],
        device: str = "cpu",
        dtype: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a PagedAttention Key/Value block from shared memory with zero copy.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (key_tensor, val_tensor)
        """
        import torch

        if dtype is None:
            dtype = torch.float16

        desc_size = ctypes.sizeof(TensorBlockDescriptor)
        payload_offset = offset + desc_size

        _, _, bytes_per_elem = TensorCodec.serialize_tensor(torch.empty(1, dtype=dtype))
        import numpy as np

        num_elements_per_tensor = int(np.prod(shape))
        num_bytes_per_tensor = num_elements_per_tensor * bytes_per_elem

        k_tensor = TensorCodec.deserialize_tensor(
            self._buf,
            offset=payload_offset,
            shape=shape,
            dtype=dtype,
            device=device,
        )

        v_tensor = TensorCodec.deserialize_tensor(
            self._buf,
            offset=payload_offset + num_bytes_per_tensor,
            shape=shape,
            dtype=dtype,
            device=device,
        )

        return k_tensor, v_tensor
