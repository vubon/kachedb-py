"""Unit tests for the async KacheDB client."""

from __future__ import annotations

import pytest

from kachedb import AsyncKacheClient


@pytest.mark.asyncio
async def test_async_client_mget_empty() -> None:
    """MGET with no keys should return empty list without connecting."""
    client = AsyncKacheClient(port=12345)
    result = await client.mget()
    assert result == []


@pytest.mark.asyncio
async def test_async_client_delete_empty() -> None:
    """DEL with no keys should return 0 without connecting."""
    client = AsyncKacheClient(port=12345)
    result = await client.delete()
    assert result == 0


@pytest.mark.asyncio
async def test_async_client_exists_empty() -> None:
    """EXISTS with no keys should return 0 without connecting."""
    client = AsyncKacheClient(port=12345)
    result = await client.exists()
    assert result == 0
