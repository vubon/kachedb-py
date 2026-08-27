"""
Zero-copy tensor access via POSIX shared memory (``/dev/shm``).

Provides functions to attach to KacheDB's per-core shared memory regions
and read KV-cache tensor blocks without any data copying.

On Linux, KacheDB creates shared memory files at ``/dev/shm/kachedb_{core_id}``
for each worker core.  This module memory-maps those files and returns
``numpy.ndarray`` (or ``torch.Tensor``) views directly into the mapped region.
"""

from __future__ import annotations

import contextlib
import mmap
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from .descriptor import TENSOR_DESCRIPTOR_MAGIC, TensorBlockDescriptor, TensorDType

# Cached mmap handles keyed by (shm_path, size_bytes).
_shm_cache: dict[str, mmap.mmap] = {}


def _get_dtype_to_numpy() -> dict[TensorDType, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for tensor operations. Install it with: pip install numpy"
        ) from exc
    return {
        TensorDType.FP32: np.dtype(np.float32),
        TensorDType.FP16: np.dtype(np.float16),
        TensorDType.BF16: np.dtype(np.uint16),
        TensorDType.FP8E4M3: np.dtype(np.uint8),
        TensorDType.FP8E5M2: np.dtype(np.uint8),
        TensorDType.INT8: np.dtype(np.int8),
        TensorDType.INT4: np.dtype(np.uint8),
    }


def attach_shm(
    core_id: int,
    size_bytes: int = 64 * 1024 * 1024,
) -> mmap.mmap:
    """Attach to the named shared memory region ``/dev/shm/kachedb_{core_id}``.

    Parameters
    ----------
    core_id : int
        KacheDB worker core index (0-based).
    size_bytes : int
        Size of the shared memory region to map.  Defaults to 64 MB.

    Returns
    -------
    mmap.mmap
        Memory-mapped file handle.

    Raises
    ------
    FileNotFoundError
        If the shared memory file does not exist (server not running or
        SHM disabled).
    """
    shm_path = f"/dev/shm/kachedb_{core_id}"

    if shm_path in _shm_cache:
        return _shm_cache[shm_path]

    if not os.path.exists(shm_path):
        raise FileNotFoundError(
            f"Shared memory region not found: {shm_path}. Is KacheDB running with --shm enabled?"
        )

    fd = os.open(shm_path, os.O_RDWR)
    try:
        shm = mmap.mmap(fd, size_bytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)

    _shm_cache[shm_path] = shm
    return shm


def read_tensor(
    core_id: int,
    byte_offset: int,
    size_bytes: int = 64 * 1024 * 1024,
) -> np.ndarray[Any, Any]:
    """Read a KV-cache tensor block from shared memory as a zero-copy numpy array.

    Reads the 64-byte :class:`~kachedb.descriptor.TensorBlockDescriptor`
    at *byte_offset*, validates the magic sentinel, and returns a
    ``numpy.ndarray`` view into the raw tensor payload immediately following
    the descriptor.

    Parameters
    ----------
    core_id : int
        KacheDB worker core index.
    byte_offset : int
        Byte offset into the shared memory region where the descriptor starts.
    size_bytes : int
        Total size of the SHM region to map.

    Returns
    -------
    numpy.ndarray
        Zero-copy view into the tensor payload.

    Raises
    ------
    ValueError
        If the magic sentinel is invalid.
    """
    shm = attach_shm(core_id, size_bytes)
    shm.seek(byte_offset)

    # Read 64-byte descriptor.
    header_bytes = shm.read(64)
    desc = TensorBlockDescriptor.from_bytes(header_bytes)

    if not desc.is_valid():
        raise ValueError(
            f"Invalid tensor descriptor magic: {hex(desc.magic)}, "
            f"expected {hex(TENSOR_DESCRIPTOR_MAGIC)}"
        )

    import numpy as np

    payload_offset = byte_offset + 64
    payload_bytes = desc.payload_bytes
    dtype_enum = TensorDType(desc.dtype)
    dtype_map = _get_dtype_to_numpy()
    np_dtype = dtype_map.get(dtype_enum, np.dtype(np.uint8))
    element_count = payload_bytes // np_dtype.itemsize

    # Zero-copy buffer view.
    return np.frombuffer(
        shm,
        dtype=np_dtype,
        count=element_count,
        offset=payload_offset,
    )


def read_torch_tensor(
    core_id: int,
    byte_offset: int,
    size_bytes: int = 64 * 1024 * 1024,
) -> Any:
    """Read a KV-cache tensor block as a ``torch.Tensor`` (zero-copy when possible).

    Requires ``torch`` to be installed (``pip install kachedb[torch]``).

    Parameters
    ----------
    core_id : int
        KacheDB worker core index.
    byte_offset : int
        Byte offset into the shared memory region.
    size_bytes : int
        Total SHM region size to map.

    Returns
    -------
    torch.Tensor
        Tensor view into the shared memory payload.

    Raises
    ------
    ImportError
        If ``torch`` is not installed.
    """
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise ImportError("numpy and torch are required for read_torch_tensor().") from exc

    np_array = read_tensor(core_id, byte_offset, size_bytes)

    # Map numpy dtype to torch dtype.
    np_to_torch = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.uint16): torch.bfloat16,  # BF16 stored as uint16
    }

    torch_dtype = np_to_torch.get(np_array.dtype, torch.uint8)
    return torch.frombuffer(np_array, dtype=torch_dtype)


def detach_all() -> None:
    """Close and unmap all cached shared memory regions."""
    for shm in _shm_cache.values():
        with contextlib.suppress(Exception):
            shm.close()
    _shm_cache.clear()
