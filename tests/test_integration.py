"""
Integration tests for KacheDB Python SDK.

These tests require a running KacheDB server on 127.0.0.1:6379.
Run with: pytest tests/test_integration.py -m integration
"""

from __future__ import annotations

import os

import pytest

from kachedb import AsyncKacheClient, KacheClient

pytestmark = pytest.mark.integration
TEST_PORT = int(os.environ.get("KACHEDB_PORT", "6379"))


class TestSyncIntegration:
    """Integration tests using the synchronous client."""

    def test_ping(self) -> None:
        with KacheClient(port=TEST_PORT) as client:
            assert client.ping() == "PONG"

    def test_set_get_delete_cycle(self) -> None:
        with KacheClient(port=TEST_PORT) as client:
            assert client.set("integration:key1", "hello") is True
            assert client.get("integration:key1") == b"hello"
            assert client.exists("integration:key1") == 1
            assert client.delete("integration:key1") == 1
            assert client.get("integration:key1") is None

    def test_mget(self) -> None:
        with KacheClient(port=TEST_PORT) as client:
            client.set("integration:a", "1")
            client.set("integration:b", "2")
            result = client.mget("integration:a", "integration:b", "integration:missing")
            assert result == [b"1", b"2", None]
            client.delete("integration:a", "integration:b")

    def test_set_with_ttl(self) -> None:
        import time

        with KacheClient(port=TEST_PORT) as client:
            client.set("integration:ttl", "temp", ex=1)
            assert client.get("integration:ttl") == b"temp"
            time.sleep(2)
            assert client.get("integration:ttl") is None

    def test_pipeline(self) -> None:
        with KacheClient(port=TEST_PORT) as client:
            pipe = client.pipeline()
            pipe.set("integration:p1", "v1")
            pipe.set("integration:p2", "v2")
            pipe.get("integration:p1")
            pipe.get("integration:p2")
            results = pipe.execute()
            assert results[0] == "OK"
            assert results[1] == "OK"
            assert results[2] == b"v1"
            assert results[3] == b"v2"
            client.delete("integration:p1", "integration:p2")

    def test_binary_values(self) -> None:
        with KacheClient(port=TEST_PORT) as client:
            binary_data = b"\x00\x01\x02\xff\xfe\xfd"
            client.set("integration:binary", binary_data)
            assert client.get("integration:binary") == binary_data
            client.delete("integration:binary")


class TestAsyncIntegration:
    """Integration tests using the async client."""

    @pytest.mark.asyncio
    async def test_async_ping(self) -> None:
        async with AsyncKacheClient(port=TEST_PORT) as client:
            result = await client.ping()
            assert result == "PONG"

    @pytest.mark.asyncio
    async def test_async_set_get(self) -> None:
        async with AsyncKacheClient(port=TEST_PORT) as client:
            assert await client.set("integration:async_key", "async_val") is True
            assert await client.get("integration:async_key") == b"async_val"
            await client.delete("integration:async_key")

    @pytest.mark.asyncio
    async def test_async_pipeline(self) -> None:
        async with AsyncKacheClient(port=TEST_PORT) as client:
            pipe = client.pipeline()
            pipe.set("integration:ap1", "v1")
            pipe.get("integration:ap1")
            results = await pipe.execute()
            assert results[0] == "OK"
            assert results[1] == b"v1"
            await client.delete("integration:ap1")

            await client.delete("integration:ap1")
