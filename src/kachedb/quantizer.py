"""
Scalar Quantization (SQ8) utility module for KacheDB vectors.

Encodes floating-point vectors into 8-bit quantized representations
with min/max bounding range metadata, reducing memory footprint by 4x.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def sq8_encode(vector: Sequence[float]) -> tuple[bytes, float, float]:
    """Quantize an f32 vector into 8-bit unsigned integers.

    Parameters
    ----------
    vector : Sequence[float]
        Input floating-point vector.

    Returns
    -------
    tuple[bytes, float, float]
        A 3-tuple of (quantized_bytes, min_val, max_val).
    """
    if not vector:
        return b"", 0.0, 0.0

    min_val = float(min(vector))
    max_val = float(max(vector))
    diff = max_val - min_val

    if diff == 0.0 or abs(diff) < 1e-12:
        return bytes([128] * len(vector)), min_val, max_val

    scale = 255.0 / diff
    quantized = bytearray(len(vector))
    for i, val in enumerate(vector):
        q = round((val - min_val) * scale)
        quantized[i] = max(0, min(255, q))

    return bytes(quantized), min_val, max_val


def sq8_decode(quantized: bytes | bytearray, min_val: float, max_val: float) -> list[float]:
    """Dequantize an 8-bit integer vector back to floating-point numbers.

    Parameters
    ----------
    quantized : bytes | bytearray
        8-bit unsigned integer quantized bytes.
    min_val : float
        Minimum value of original bounding range.
    max_val : float
        Maximum value of original bounding range.

    Returns
    -------
    list[float]
        Reconstructed floating-point vector.
    """
    if not quantized:
        return []

    diff = max_val - min_val
    if diff == 0.0 or abs(diff) < 1e-12:
        return [min_val] * len(quantized)

    scale = diff / 255.0
    return [min_val + byte * scale for byte in quantized]
