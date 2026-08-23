"""Unit tests for tensor descriptor and zero-copy access."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from kachedb.descriptor import (
    TENSOR_DESCRIPTOR_MAGIC,
    TensorBlockDescriptor,
    TensorDType,
)

# ── TensorDType Tests ─────────────────────────────────────────────────────


class TestTensorDType:
    def test_fp32_size(self) -> None:
        assert TensorDType.FP32.element_size_bytes() == 4

    def test_fp16_size(self) -> None:
        assert TensorDType.FP16.element_size_bytes() == 2

    def test_bf16_size(self) -> None:
        assert TensorDType.BF16.element_size_bytes() == 2

    def test_fp8_sizes(self) -> None:
        assert TensorDType.FP8E4M3.element_size_bytes() == 1
        assert TensorDType.FP8E5M2.element_size_bytes() == 1

    def test_int8_size(self) -> None:
        assert TensorDType.INT8.element_size_bytes() == 1

    def test_int4_size(self) -> None:
        assert TensorDType.INT4.element_size_bytes() == 1


# ── TensorBlockDescriptor Tests ──────────────────────────────────────────


class TestTensorBlockDescriptor:
    def test_struct_is_64_bytes(self) -> None:
        """Descriptor must be exactly 1 cache line (64 bytes)."""
        assert ctypes.sizeof(TensorBlockDescriptor) == 64

    def test_magic_validation(self) -> None:
        desc = TensorBlockDescriptor()
        desc.magic = TENSOR_DESCRIPTOR_MAGIC
        assert desc.is_valid()

    def test_invalid_magic(self) -> None:
        desc = TensorBlockDescriptor()
        desc.magic = 0xDEADBEEF
        assert not desc.is_valid()

    def test_compute_shape(self) -> None:
        desc = TensorBlockDescriptor()
        desc.num_layers = 32
        desc.num_heads = 8
        desc.block_size = 16
        desc.head_dim = 128
        assert desc.compute_shape() == (2, 32, 8, 16, 128)

    def test_llama3_8b_bf16_fields(self) -> None:
        """LLaMA-3 8B: 32 layers, 8 GQA heads, 16 tokens/block, 128 head-dim, BF16."""
        desc = TensorBlockDescriptor()
        desc.magic = TENSOR_DESCRIPTOR_MAGIC
        desc.layer_id = 0
        desc.num_layers = 32
        desc.block_size = 16
        desc.num_heads = 8
        desc.head_dim = 128
        desc.dtype = TensorDType.BF16
        desc.payload_bytes = 2 * 32 * 8 * 16 * 128 * 2  # 2 MB

        assert desc.is_valid()
        assert desc.payload_bytes == 2 * 1024 * 1024
        assert desc.compute_shape() == (2, 32, 8, 16, 128)

    def test_round_trip_bytes(self) -> None:
        """Verify descriptor can be serialized to bytes and back."""
        desc = TensorBlockDescriptor()
        desc.magic = TENSOR_DESCRIPTOR_MAGIC
        desc.num_layers = 4
        desc.num_heads = 2
        desc.block_size = 8
        desc.head_dim = 64
        desc.dtype = TensorDType.FP16
        desc.payload_bytes = 2 * 4 * 2 * 8 * 64 * 2  # = 16384

        raw = bytes(desc)
        assert len(raw) == 64

        restored = TensorBlockDescriptor.from_bytes(raw)
        assert restored.is_valid()
        assert restored.num_layers == 4
        assert restored.num_heads == 2
        assert restored.block_size == 8
        assert restored.head_dim == 64
        assert restored.payload_bytes == 16384

    def test_from_bytes_too_short(self) -> None:
        with pytest.raises(ValueError, match="64 bytes"):
            TensorBlockDescriptor.from_bytes(b"\x00" * 32)


# ── Zero-Copy numpy frombuffer Test ───────────────────────────────────────


class TestZeroCopyNumpy:
    def test_frombuffer_zero_copy(self) -> None:
        """Verify np.frombuffer creates a zero-copy view (in-place mutation)."""
        # Construct 64-byte header + 1024 floats payload.
        desc = TensorBlockDescriptor()
        desc.magic = TENSOR_DESCRIPTOR_MAGIC
        desc.num_layers = 1
        desc.num_heads = 1
        desc.block_size = 16
        desc.head_dim = 64
        desc.dtype = TensorDType.FP32
        desc.payload_bytes = 1024 * 4

        header_bytes = bytes(desc)
        assert len(header_bytes) == 64

        raw_payload = np.arange(1024, dtype=np.float32).tobytes()
        full_buffer = bytearray(header_bytes + raw_payload)

        # Zero-copy view.
        tensor_view = np.frombuffer(full_buffer, dtype=np.float32, count=1024, offset=64)

        assert tensor_view[0] == 0.0
        assert tensor_view[100] == 100.0
        assert tensor_view[1023] == 1023.0

        # Modify in-place — proves zero-copy.
        tensor_view[0] = 999.0
        readback = np.frombuffer(full_buffer, dtype=np.float32, count=1, offset=64)
        assert readback[0] == 999.0

    def test_bf16_as_uint16_view(self) -> None:
        """BF16 tensors use uint16 view in standard numpy."""
        data = bytearray(100 * 2)  # 100 elements x 2 bytes
        view = np.frombuffer(data, dtype=np.uint16, count=100)
        assert view.shape == (100,)
        assert view.dtype == np.uint16
