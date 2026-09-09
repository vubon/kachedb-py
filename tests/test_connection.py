"""Unit tests for kachedb.connection.Connection."""

from __future__ import annotations

import builtins
import ssl
from unittest.mock import MagicMock

import pytest

from kachedb.connection import Connection
from kachedb.exceptions import ConnectionError, TimeoutError
from tests.conftest import MockKacheDBServer, resp_bulk_string, resp_simple_string


class TestConnection:
    def test_not_connected_errors(self) -> None:
        conn = Connection(host="127.0.0.1", port=6379)
        assert not conn.is_connected

        with pytest.raises(ConnectionError, match="Not connected"):
            conn.send_command("PING")

        with pytest.raises(ConnectionError, match="Not connected"):
            conn.send_packed(b"*1\r\n$4\r\nPING\r\n")

        with pytest.raises(ConnectionError, match="Not connected"):
            conn.read_response()

        # Disconnect on disconnected instance is a safe no-op
        conn.disconnect()
        assert not conn.is_connected

    def test_connect_and_disconnect(self, mock_server: MockKacheDBServer) -> None:
        port = mock_server.start()
        conn = Connection(port=port)
        conn.connect()
        assert conn.is_connected

        # Idempotent connect
        conn.connect()
        assert conn.is_connected

        conn.disconnect()
        assert not conn.is_connected

    def test_connect_failure(self) -> None:
        conn = Connection(host="127.0.0.1", port=59999, socket_timeout=0.2)
        with pytest.raises(ConnectionError, match="Failed to connect"):
            conn.connect()
        assert not conn.is_connected

    def test_send_packed_and_decode_responses(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_bulk_string(b"hello world"))
        port = mock_server.start()

        conn = Connection(port=port, decode_responses=True)
        conn.connect()
        conn.send_packed(b"*2\r\n$3\r\nGET\r\n$3\r\nkey\r\n")
        val = conn.read_response()
        assert val == "hello world"
        conn.disconnect()

    def test_check_health(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_simple_string("PONG"),
            resp_simple_string("WRONG"),
        )
        port = mock_server.start()

        conn = Connection(port=port)
        conn.connect()
        assert conn.check_health() is True
        assert conn.check_health() is False
        conn.disconnect()
        # Health check on disconnected returns False
        assert conn.check_health() is False

    def test_send_and_read_error_handling(self) -> None:
        conn = Connection(port=6379)
        mock_sock = MagicMock()
        conn._sock = mock_sock
        conn._reader = MagicMock()

        # Timeout on send_command
        mock_sock.sendall.side_effect = builtins.TimeoutError("timed out")
        with pytest.raises(TimeoutError, match="Timeout sending command"):
            conn.send_command("PING")
        assert not conn.is_connected

        # OSError on send_command
        conn._sock = mock_sock
        conn._reader = MagicMock()
        mock_sock.sendall.side_effect = OSError("broken pipe")
        with pytest.raises(ConnectionError, match="Error sending command"):
            conn.send_command("PING")
        assert not conn.is_connected

        # Timeout on send_packed
        conn._sock = mock_sock
        conn._reader = MagicMock()
        mock_sock.sendall.side_effect = builtins.TimeoutError("timed out")
        with pytest.raises(TimeoutError, match="Timeout sending data"):
            conn.send_packed(b"raw")
        assert not conn.is_connected

        # OSError on send_packed
        conn._sock = mock_sock
        conn._reader = MagicMock()
        mock_sock.sendall.side_effect = OSError("connection reset")
        with pytest.raises(ConnectionError, match="Error sending data"):
            conn.send_packed(b"raw")
        assert not conn.is_connected

        # Timeout on read_response
        conn._sock = mock_sock
        conn._reader = MagicMock()
        conn._reader.read_response.side_effect = builtins.TimeoutError("timed out")
        with pytest.raises(TimeoutError, match="Timeout reading response"):
            conn.read_response()
        assert not conn.is_connected

        # OSError on read_response
        conn._sock = mock_sock
        conn._reader = MagicMock()
        conn._reader.read_response.side_effect = OSError("read error")
        with pytest.raises(ConnectionError, match="Error reading response"):
            conn.read_response()
        assert not conn.is_connected

    def test_ssl_connection_setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_ctx = MagicMock()
        mock_wrap = MagicMock()
        mock_ctx.wrap_socket.return_value = mock_wrap
        monkeypatch.setattr(ssl, "create_default_context", lambda **_kw: mock_ctx)

        mock_socket_mod = MagicMock()
        mock_raw_sock = MagicMock()
        mock_socket_mod.create_connection.return_value = mock_raw_sock
        monkeypatch.setattr("socket.create_connection", mock_socket_mod.create_connection)

        conn = Connection(
            host="example.com",
            port=6380,
            ssl=True,
            ssl_check_hostname=False,
            ssl_cert_reqs=ssl.CERT_NONE,
            ssl_certfile="cert.pem",
            ssl_keyfile="key.pem",
        )
        conn.connect()

        assert mock_ctx.check_hostname is False
        assert mock_ctx.verify_mode == ssl.CERT_NONE
        mock_ctx.load_cert_chain.assert_called_once_with(certfile="cert.pem", keyfile="key.pem")
        assert conn.is_connected
        conn.disconnect()
