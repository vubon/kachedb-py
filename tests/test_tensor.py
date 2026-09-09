"""Unit tests for tensor descriptor and zero-copy access."""

from __future__ import annotations

import ctypes
from typing import Any

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

    def test_to_bytes(self) -> None:
        desc = TensorBlockDescriptor()
        desc.magic = TENSOR_DESCRIPTOR_MAGIC
        raw = desc.to_bytes()
        assert len(raw) == 64
        assert raw == bytes(desc)


class TestTensorCodec:
    def test_codec_bfloat16_and_unknown_dtype(self) -> None:
        import torch

        from kachedb.descriptor import TensorCodec, TensorDType

        # bfloat16
        t_bf16 = torch.ones((2, 4), dtype=torch.bfloat16)
        raw_b, dt, size = TensorCodec.serialize_tensor(t_bf16)
        assert dt == TensorDType.BF16
        assert size == 2
        assert len(raw_b) == 16

        # unknown dtype (e.g. float64 -> fallback to FP16)
        t_f64 = torch.ones((2, 4), dtype=torch.float64)
        _raw_b2, dt2, size2 = TensorCodec.serialize_tensor(t_f64)
        assert dt2 == TensorDType.FP16
        assert size2 == 2

        # deserialize with bytearray, raw bytes, and readonly memoryview
        deserialized = TensorCodec.deserialize_tensor(
            bytearray(raw_b), offset=0, shape=(2, 4), dtype=torch.bfloat16, device="cpu"
        )
        assert deserialized.shape == (2, 4)

        from_bytes = TensorCodec.deserialize_tensor(
            raw_b, offset=0, shape=(2, 4), dtype=torch.bfloat16
        )
        assert from_bytes.shape == (2, 4)

        from_mv = TensorCodec.deserialize_tensor(
            memoryview(raw_b), offset=0, shape=(2, 4), dtype=torch.bfloat16
        )
        assert from_mv.shape == (2, 4)


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


# ── kachedb.tensor Module Tests ───────────────────────────────────────────


class TestTensorOperations:
    def test_get_dtype_to_numpy(self) -> None:
        from kachedb.tensor import _get_dtype_to_numpy

        dtypes = _get_dtype_to_numpy()
        assert TensorDType.FP32 in dtypes
        assert TensorDType.FP16 in dtypes
        assert TensorDType.BF16 in dtypes
        assert TensorDType.INT8 in dtypes
        assert dtypes[TensorDType.FP32] == np.float32

    def test_get_dtype_to_numpy_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from kachedb.tensor import _get_dtype_to_numpy

        monkeypatch.setitem(sys.modules, "numpy", None)
        with pytest.raises(ImportError, match="numpy is required"):
            _get_dtype_to_numpy()

    def test_attach_shm_not_found(self) -> None:
        from kachedb.tensor import attach_shm, detach_all

        detach_all()
        with pytest.raises(FileNotFoundError, match="Shared memory region not found"):
            attach_shm(core_id=99999)

    def test_attach_shm_and_cache(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        from kachedb.tensor import _shm_cache, attach_shm, detach_all

        detach_all()
        shm_file = tmp_path / "kachedb_shm_test"
        size = 1024 * 1024
        shm_file.write_bytes(b"\x00" * size)

        real_exists = os.path.exists
        real_open = os.open

        target_shm = "/dev/shm/kachedb_42"

        def fake_exists(p: str | bytes | os.PathLike[Any]) -> bool:
            if str(p) == target_shm:
                return True
            return real_exists(p)

        def fake_open(p: str | bytes | os.PathLike[Any], flags: int, *args: Any) -> int:
            if str(p) == target_shm:
                return real_open(str(shm_file), flags, *args)
            return real_open(p, flags, *args)

        monkeypatch.setattr(os.path, "exists", fake_exists)
        monkeypatch.setattr(os, "open", fake_open)

        # First attach maps file
        handle1 = attach_shm(core_id=42, size_bytes=size)
        assert target_shm in _shm_cache

        # Second attach returns cached
        handle2 = attach_shm(core_id=42, size_bytes=size)
        assert handle1 is handle2

        detach_all()
        assert len(_shm_cache) == 0

    def test_read_tensor_and_torch(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        from kachedb.tensor import detach_all, read_tensor, read_torch_tensor

        detach_all()
        shm_file = tmp_path / "kachedb_tensor_data"
        size = 1024 * 1024

        # Prepare payload: 64-byte descriptor + 16 float32 values (64 bytes)
        desc = TensorBlockDescriptor()
        desc.magic = TENSOR_DESCRIPTOR_MAGIC
        desc.dtype = TensorDType.FP32
        desc.payload_bytes = 16 * 4

        payload_floats = np.array([float(i) * 1.5 for i in range(16)], dtype=np.float32)
        content = bytearray(size)
        header_bytes = bytes(desc)
        content[:64] = header_bytes
        content[64 : 64 + len(payload_floats.tobytes())] = payload_floats.tobytes()

        # Write invalid magic at offset 512
        bad_desc = TensorBlockDescriptor()
        bad_desc.magic = 0x12345678
        content[512 : 512 + 64] = bytes(bad_desc)

        shm_file.write_bytes(content)

        target_shm = "/dev/shm/kachedb_7"
        real_exists = os.path.exists
        real_open = os.open

        monkeypatch.setattr(
            os.path, "exists", lambda p: True if str(p) == target_shm else real_exists(p)
        )
        monkeypatch.setattr(
            os,
            "open",
            lambda p, flags, *args: (
                real_open(str(shm_file), flags, *args)
                if str(p) == target_shm
                else real_open(p, flags, *args)
            ),
        )

        arr = read_tensor(core_id=7, byte_offset=0, size_bytes=size)
        assert len(arr) == 16
        assert arr[0] == 0.0
        assert arr[1] == 1.5
        assert arr[15] == 22.5

        # Test invalid magic
        with pytest.raises(ValueError, match="Invalid tensor descriptor magic"):
            read_tensor(core_id=7, byte_offset=512, size_bytes=size)

        # Test torch tensor read
        torch_tensor = read_torch_tensor(core_id=7, byte_offset=0, size_bytes=size)
        assert torch_tensor.shape[0] == 16
        assert float(torch_tensor[1]) == 1.5

        detach_all()

    def test_read_torch_tensor_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from kachedb.tensor import read_torch_tensor

        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ImportError, match="torch are required"):
            read_torch_tensor(core_id=1, byte_offset=0)
