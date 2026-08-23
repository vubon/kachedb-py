"""Unit tests for connection pool."""

from __future__ import annotations

import pytest

from kachedb.exceptions import PoolExhaustedError
from kachedb.pool import ConnectionPool


class TestConnectionPool:
    def test_pool_exhaustion(self) -> None:
        """Pool should raise PoolExhaustedError when max_connections is reached."""
        pool = ConnectionPool(
            host="127.0.0.1",
            port=19999,  # Non-listening port
            max_connections=2,
        )
        # Simulate exhaustion by incrementing active count.
        pool._active_count = 2
        with pytest.raises(PoolExhaustedError, match="max_connections=2"):
            pool.get_connection()

    def test_disconnect_all_resets_count(self) -> None:
        """disconnect_all should reset the active connection count."""
        pool = ConnectionPool(
            host="127.0.0.1",
            port=19999,
            max_connections=5,
        )
        pool._active_count = 3
        pool.disconnect_all()
        assert pool._active_count == 0

    def test_pool_default_max_connections(self) -> None:
        pool = ConnectionPool()
        assert pool.max_connections == 10
