"""Unit tests for the RESP2/RESP3 encoder and decoder."""

from __future__ import annotations

import socket

import pytest

from kachedb.exceptions import ProtocolError, ResponseError
from kachedb.resp import RespReader, encode_command, encode_commands

# ── Encoder Tests ─────────────────────────────────────────────────────────


class TestEncodeCommand:
    def test_ping(self) -> None:
        result = encode_command(["PING"])
        assert result == b"*1\r\n$4\r\nPING\r\n"

    def test_get(self) -> None:
        result = encode_command(["GET", "my_key"])
        assert result == b"*2\r\n$3\r\nGET\r\n$6\r\nmy_key\r\n"

    def test_set_with_ex(self) -> None:
        result = encode_command(["SET", "user", "alice", "EX", "60"])
        assert result == (
            b"*5\r\n$3\r\nSET\r\n$4\r\nuser\r\n$5\r\nalice\r\n$2\r\nEX\r\n$2\r\n60\r\n"
        )

    def test_binary_value(self) -> None:
        result = encode_command(["SET", "key", b"\x00\x01\x02"])
        assert b"$3\r\n\x00\x01\x02\r\n" in result

    def test_mget(self) -> None:
        result = encode_command(["MGET", "k1", "k2", "k3"])
        assert result.startswith(b"*4\r\n")

    def test_del(self) -> None:
        result = encode_command(["DEL", "key1", "key2"])
        assert result.startswith(b"*3\r\n")


class TestEncodeCommands:
    def test_pipeline_encoding(self) -> None:
        commands: list[list[str | bytes]] = [["SET", "a", "1"], ["GET", "a"]]
        result = encode_commands(commands)
        # Should be two concatenated RESP arrays.
        assert result.count(b"*") == 2
        assert b"SET" in result
        assert b"GET" in result


# ── Decoder Tests ─────────────────────────────────────────────────────────


def _make_reader(data: bytes) -> RespReader:
    """Create a RespReader backed by a socket pair with pre-loaded data."""
    server_sock, client_sock = socket.socketpair()
    server_sock.sendall(data)
    server_sock.close()
    return RespReader(client_sock)


class TestRespReader:
    def test_simple_string(self) -> None:
        reader = _make_reader(b"+OK\r\n")
        assert reader.read_response() == "OK"

    def test_simple_string_pong(self) -> None:
        reader = _make_reader(b"+PONG\r\n")
        assert reader.read_response() == "PONG"

    def test_error_raises(self) -> None:
        reader = _make_reader(b"-ERR unknown command 'FOO'\r\n")
        with pytest.raises(ResponseError, match="ERR unknown command"):
            reader.read_response()

    def test_integer(self) -> None:
        reader = _make_reader(b":42\r\n")
        assert reader.read_response() == 42

    def test_integer_zero(self) -> None:
        reader = _make_reader(b":0\r\n")
        assert reader.read_response() == 0

    def test_integer_negative(self) -> None:
        reader = _make_reader(b":-1\r\n")
        assert reader.read_response() == -1

    def test_bulk_string(self) -> None:
        reader = _make_reader(b"$6\r\nfoobar\r\n")
        assert reader.read_response() == b"foobar"

    def test_bulk_string_binary(self) -> None:
        reader = _make_reader(b"$3\r\n\x00\x01\x02\r\n")
        assert reader.read_response() == b"\x00\x01\x02"

    def test_null_bulk_string(self) -> None:
        reader = _make_reader(b"$-1\r\n")
        assert reader.read_response() is None

    def test_array(self) -> None:
        data = b"*2\r\n$5\r\nalice\r\n$3\r\nbob\r\n"
        reader = _make_reader(data)
        result = reader.read_response()
        assert result == [b"alice", b"bob"]

    def test_array_with_nulls(self) -> None:
        data = b"*3\r\n$5\r\nalice\r\n$-1\r\n$3\r\nbob\r\n"
        reader = _make_reader(data)
        result = reader.read_response()
        assert result == [b"alice", None, b"bob"]

    def test_empty_array(self) -> None:
        reader = _make_reader(b"*0\r\n")
        assert reader.read_response() == []

    def test_null_array(self) -> None:
        reader = _make_reader(b"*-1\r\n")
        assert reader.read_response() is None

    def test_multiple_sequential_responses(self) -> None:
        data = b"+OK\r\n$5\r\nhello\r\n:100\r\n"
        reader = _make_reader(data)
        assert reader.read_response() == "OK"
        assert reader.read_response() == b"hello"
        assert reader.read_response() == 100

    def test_unknown_marker_raises_protocol_error(self) -> None:
        reader = _make_reader(b"!invalid\r\n")
        with pytest.raises(ProtocolError, match="Unknown RESP type marker"):
            reader.read_response()

    def test_connection_closed_raises(self) -> None:
        server_sock, client_sock = socket.socketpair()
        server_sock.close()  # immediate close = empty read
        reader = RespReader(client_sock)
        with pytest.raises(ProtocolError, match="Connection closed"):
            reader.read_response()
