"""
Binary decoder for the 64-byte ``TensorBlockDescriptor`` matching the
KacheDB Rust ``kachedb-proto-tensor`` crate layout.

The descriptor sits at the base of every KV-cache slab block and encodes
enough metadata for Python workers to reconstruct a ``torch.Tensor``
(or ``numpy.ndarray``) via ``torch.frombuffer()`` / ``np.frombuffer()``
with zero data copies.

Memory Layout (1 cache line = 64 bytes)::

    ┌──────────────────────────────── 64 Bytes ────────────────────────────────┐
    │ magic(4) │ layer_id(2) │ num_layers(2) │ block_size(2) │ num_heads(2)  │
    │ head_dim(2) │ dtype(1) │ _pad(7) │ seq_prefix_hash(8) │               │
    │ payload_bytes(4) │ _pad(28)                                            │
    └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import ctypes
from enum import IntEnum

#: Magic sentinel value: ``KACH`` in little-endian ASCII bytes.
TENSOR_DESCRIPTOR_MAGIC: int = 0x4B41_4348


class TensorDType(IntEnum):
    """Supported numeric precision types for LLM KV-cache tensors.

    Matches the dtypes supported by vLLM PagedAttention and PyTorch.
    """

    FP32 = 0
    """32-bit IEEE 754 float (4 bytes per element)."""

    FP16 = 1
    """16-bit IEEE 754 half-precision float (2 bytes per element)."""

    BF16 = 2
    """16-bit brain float (2 bytes per element). Default for LLaMA/Gemma."""

    FP8E4M3 = 3
    """8-bit float, E4M3 encoding (1 byte per element)."""

    FP8E5M2 = 4
    """8-bit float, E5M2 encoding (1 byte per element)."""

    INT8 = 5
    """8-bit signed integer (1 byte per element)."""

    INT4 = 6
    """4-bit integer, packed 2 per byte (0.5 bytes per element)."""

    def element_size_bytes(self) -> int:
        """Return the storage size in bytes for one element of this dtype."""
        if self == TensorDType.FP32:
            return 4
        if self in (TensorDType.FP16, TensorDType.BF16):
            return 2
        return 1  # FP8*, INT8, INT4 (INT4 packs 2 per byte)


class TensorBlockDescriptor(ctypes.Structure):
    """64-byte cache-line aligned descriptor at the start of every KV-cache
    slab block.

    Mirrors the Rust ``#[repr(C, align(64))]`` struct defined in
    ``crates/kachedb-proto-tensor/src/descriptor.rs``.
    """

    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("layer_id", ctypes.c_uint16),
        ("num_layers", ctypes.c_uint16),
        ("block_size", ctypes.c_uint16),
        ("num_heads", ctypes.c_uint16),
        ("head_dim", ctypes.c_uint16),
        ("dtype", ctypes.c_uint8),
        ("_reserved", ctypes.c_uint8 * 7),
        ("sequence_prefix_hash", ctypes.c_uint64),
        ("payload_bytes", ctypes.c_uint32),
        ("_cacheline_pad", ctypes.c_uint8 * 28),
    ]

    def is_valid(self) -> bool:
        """Return ``True`` if the magic sentinel matches ``0x4B414348``."""
        return self.magic == TENSOR_DESCRIPTOR_MAGIC

    def compute_shape(self) -> tuple[int, int, int, int, int]:
        """Return the 5D tensor shape: ``(2, num_layers, num_heads, block_size, head_dim)``.

        The leading dimension ``2`` represents Key and Value tensors.
        """
        return (2, self.num_layers, self.num_heads, self.block_size, self.head_dim)

    @classmethod
    def from_bytes(cls, source: bytes) -> TensorBlockDescriptor:
        """Construct a descriptor from a 64-byte buffer.

        Raises
        ------
        ValueError
            If the source buffer is shorter than 64 bytes.
        """
        if len(source) < 64:
            raise ValueError(f"Descriptor requires 64 bytes, got {len(source)}")
        return ctypes.Structure.__class__.from_buffer_copy(cls, source[:64])  # type: ignore[arg-type]
