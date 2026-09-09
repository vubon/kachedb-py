"""Unit tests for connection pools (sync and async)."""

from __future__ import annotations

import asyncio
import queue
from unittest.mock import AsyncMock, MagicMock

import pytest

from kachedb.connection import Connection
from kachedb.exceptions import ConnectionError, PoolExhaustedError
from kachedb.pool import AsyncConnectionPool, ConnectionPool
from tests.conftest import MockKacheDBServer, resp_simple_string


class TestConnectionPool:
    def test_pool_exhaustion(self) -> None:
        """Pool should raise PoolExhaustedError when max_connections is reached."""
        pool = ConnectionPool(
            host="127.0.0.1",
            port=19999,
            max_connections=2,
        )
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

    def test_acquire_release_and_reuse(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("PONG"))
        port = mock_server.start()

        pool = ConnectionPool(port=port, max_connections=2)
        conn = pool.get_connection()
        assert conn.is_connected
        assert pool._active_count == 1

        # Return to pool
        pool.release_connection(conn)
        assert pool._pool.qsize() == 1

        # Re-acquire: should get the same cached connection
        conn2 = pool.get_connection()
        assert conn2 is conn
        pool.release_connection(conn2)
        pool.disconnect_all()

    def test_stale_connection_reconnection(self, mock_server: MockKacheDBServer) -> None:
        port = mock_server.start()
        pool = ConnectionPool(port=port, max_connections=2)

        stale_conn = Connection(port=port)
        stale_conn.connect()
        stale_conn.disconnect()  # Stale

        pool._pool.put_nowait(stale_conn)
        pool._active_count = 1

        # get_connection discards stale and creates fresh
        fresh = pool.get_connection()
        assert fresh.is_connected
        assert fresh is not stale_conn
        pool.disconnect_all()

    def test_pool_auth_success(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("OK"))
        port = mock_server.start()

        pool_ok = ConnectionPool(port=port, password="secret")
        conn = pool_ok.get_connection()
        assert conn.is_connected
        pool_ok.disconnect_all()

    def test_pool_auth_failure(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("ERR_INVALID_PASSWORD"))
        port = mock_server.start()

        pool_fail = ConnectionPool(port=port, password="badpassword")
        with pytest.raises(ConnectionError, match="Authentication failed"):
            pool_fail.get_connection()
        assert pool_fail._active_count == 0

    def test_release_when_full(self) -> None:
        pool = ConnectionPool(max_connections=1)
        mock_conn = MagicMock(spec=Connection)
        mock_conn.is_connected = True

        # Fill queue
        pool._pool = queue.Queue(maxsize=1)
        dummy = MagicMock(spec=Connection)
        pool._pool.put_nowait(dummy)
        pool._active_count = 2

        # Releasing when queue is full forces disconnect
        pool.release_connection(mock_conn)
        mock_conn.disconnect.assert_called_once()
        assert pool._active_count == 1


class TestAsyncConnectionPool:
    def test_async_pool_defaults(self) -> None:
        pool = AsyncConnectionPool()
        assert pool.max_connections == 10

    @pytest.mark.asyncio
    async def test_async_pool_exhaustion(self) -> None:
        pool = AsyncConnectionPool(host="127.0.0.1", port=19999, max_connections=2)
        pool._active_count = 2
        with pytest.raises(PoolExhaustedError, match="max_connections=2"):
            await pool.get_connection()

    @pytest.mark.asyncio
    async def test_async_acquire_release_and_reuse(self, mock_server: MockKacheDBServer) -> None:
        port = mock_server.start()
        pool = AsyncConnectionPool(port=port, max_connections=2)

        reader, writer, resp_reader = await pool.get_connection()
        assert not writer.is_closing()
        assert pool._active_count == 1

        await pool.release_connection(reader, writer, resp_reader)
        assert pool._pool.qsize() == 1

        # Re-acquire returns cached
        r2, w2, _ = await pool.get_connection()
        assert w2 is writer
        await pool.release_connection(r2, w2, _)
        await pool.disconnect_all()
        assert pool._active_count == 0

    @pytest.mark.asyncio
    async def test_async_stale_connection(self, mock_server: MockKacheDBServer) -> None:
        port = mock_server.start()
        pool = AsyncConnectionPool(port=port, max_connections=2)

        fake_reader = MagicMock()
        fake_writer = MagicMock()
        fake_writer.is_closing.return_value = True  # stale
        fake_resp = MagicMock()

        pool._pool.put_nowait((fake_reader, fake_writer, fake_resp))
        pool._active_count = 1

        # Acquiring discards stale and creates fresh
        _, writer, _ = await pool.get_connection()
        assert not writer.is_closing()
        fake_writer.close.assert_called_once()
        await pool.disconnect_all()

    @pytest.mark.asyncio
    async def test_async_pool_auth_success(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("OK"))
        port = mock_server.start()

        pool_ok = AsyncConnectionPool(port=port, password="secret")
        r, w, rr = await pool_ok.get_connection()
        assert not w.is_closing()
        await pool_ok.release_connection(r, w, rr)
        await pool_ok.disconnect_all()

    @pytest.mark.asyncio
    async def test_async_pool_auth_failure(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("ERR_INVALID_PASSWORD"))
        port = mock_server.start()

        pool_fail = AsyncConnectionPool(port=port, password="badpassword")
        with pytest.raises(ConnectionError, match="Authentication failed"):
            await pool_fail.get_connection()
        assert pool_fail._active_count == 0

    @pytest.mark.asyncio
    async def test_async_connection_error(self) -> None:
        pool = AsyncConnectionPool(host="127.0.0.1", port=59998)
        with pytest.raises(ConnectionError, match="Failed to connect"):
            await pool.get_connection()
        assert pool._active_count == 0

    @pytest.mark.asyncio
    async def test_async_release_when_full(self) -> None:
        pool = AsyncConnectionPool(max_connections=1)
        pool._pool = asyncio.Queue(maxsize=1)
        pool._active_count = 2

        # Put one connection
        pool._pool.put_nowait((MagicMock(), MagicMock(), MagicMock()))

        # Releasing another when full closes it
        mock_writer = MagicMock()
        mock_writer.is_closing.return_value = False
        await pool.release_connection(MagicMock(), mock_writer, MagicMock())
        mock_writer.close.assert_called_once()
        assert pool._active_count == 1

    @pytest.mark.asyncio
    async def test_async_pool_ssl_setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ssl as _ssl

        mock_ctx = MagicMock()
        monkeypatch.setattr(_ssl, "create_default_context", lambda **_kw: mock_ctx)

        mock_open = AsyncMock(return_value=(MagicMock(), MagicMock()))
        monkeypatch.setattr(asyncio, "open_connection", mock_open)

        pool = AsyncConnectionPool(
            host="example.com",
            port=6380,
            ssl=True,
            ssl_check_hostname=False,
            ssl_certfile="cert.pem",
            ssl_keyfile="key.pem",
        )
        await pool.get_connection()
        assert mock_ctx.check_hostname is False
        assert mock_ctx.verify_mode == _ssl.CERT_NONE
        mock_ctx.load_cert_chain.assert_called_once_with(certfile="cert.pem", keyfile="key.pem")
        await pool.disconnect_all()
