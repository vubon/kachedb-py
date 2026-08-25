"""
Asynchronous DMA and pinned host memory transfers for KacheDB KV-cache blocks.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..descriptor import (
    TENSOR_DESCRIPTOR_MAGIC,
    TensorBlockDescriptor,
    TensorDType,
)


class KacheDBMemoryManager:
    """Manages zero-copy shared memory slots and host-to-device transfers."""

    def __init__(self, core_id: int = 0, pool_size_mb: int = 512):
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
    ) -> Tuple[int, int, TensorBlockDescriptor]:
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
        Tuple[int, int, TensorBlockDescriptor]
            (byte_offset, total_bytes, descriptor)
        """
        k_bytes = key_tensor.contiguous().cpu().numpy().tobytes()
        v_bytes = val_tensor.contiguous().cpu().numpy().tobytes()
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
            dtype=TensorDType.FP16,
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
        self, offset: int, shape: tuple, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read a PagedAttention Key/Value block from shared memory with zero copy.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (key_tensor, val_tensor)
        """
        desc_size = ctypes.sizeof(TensorBlockDescriptor)
        payload_offset = offset + desc_size

        num_elements_per_tensor = int(np.prod(shape))
        num_bytes_per_tensor = num_elements_per_tensor * 2  # FP16

        k_view = np.frombuffer(
            self._buf,
            dtype=np.float16,
            count=num_elements_per_tensor,
            offset=payload_offset,
        ).reshape(shape)

        v_view = np.frombuffer(
            self._buf,
            dtype=np.float16,
            count=num_elements_per_tensor,
            offset=payload_offset + num_bytes_per_tensor,
        ).reshape(shape)

        k_tensor = torch.from_numpy(k_view)
        v_tensor = torch.from_numpy(v_view)

        if device != "cpu" and torch.cuda.is_available():
            k_tensor = k_tensor.to(device, non_blocking=True)
            v_tensor = v_tensor.to(device, non_blocking=True)

        return k_tensor, v_tensor
