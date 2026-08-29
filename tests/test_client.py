"""Unit tests for the synchronous KacheClient using mock server."""

from __future__ import annotations

from kachedb import KacheClient
from tests.conftest import (
    MockKacheDBServer,
    resp_array,
    resp_bulk_string,
    resp_integer,
    resp_null,
    resp_simple_string,
)


class TestKacheClientPing:
    def test_ping_returns_pong(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("PONG"))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            result = client.ping()
            assert result == "PONG"


class TestKacheClientGetSet:
    def test_set_returns_true(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("OK"))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            result = client.set("user:1", "alice")
            assert result is True

    def test_get_existing_key(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_bulk_string(b"alice"))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            result = client.get("user:1")
            assert result == b"alice"

    def test_get_missing_key(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_null())
        port = mock_server.start()

        with KacheClient(port=port) as client:
            result = client.get("nonexistent")
            assert result is None

    def test_set_with_ttl(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("OK"))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            result = client.set("temp", "data", ex=60)
            assert result is True

    def test_set_with_px(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("OK"))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            result = client.set("temp", "data", px=5000)
            assert result is True


class TestKacheClientMGet:
    def test_mget_returns_list(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_array(resp_bulk_string(b"alice"), resp_null(), resp_bulk_string(b"bob"))
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            result = client.mget("user:1", "user:2", "user:3")
            assert result == [b"alice", None, b"bob"]

    def test_mget_empty(self) -> None:
        """MGET with no keys returns empty list without connecting."""
        client = KacheClient(port=12345)
        result = client.mget()
        assert result == []


class TestKacheClientDelete:
    def test_delete_returns_count(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(2))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            result = client.delete("key1", "key2")
            assert result == 2

    def test_delete_empty_keys(self) -> None:
        client = KacheClient(port=12345)
        result = client.delete()
        assert result == 0


class TestKacheClientExists:
    def test_exists_returns_count(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(1))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            result = client.exists("user:1")
            assert result == 1

    def test_exists_empty_keys(self) -> None:
        client = KacheClient(port=12345)
        result = client.exists()
        assert result == 0


class TestKacheClientExtendedCommands:
    def test_mset(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("OK"))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.mset({"k1": "v1", "k2": "v2"}) is True

    def test_incr_decr(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(1), resp_integer(11), resp_integer(10))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.incr("counter") == 1
            assert client.incrby("counter", 10) == 11
            assert client.decr("counter") == 10

    def test_append_strlen(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(11), resp_integer(11))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.append("msg", "hello world") == 11
            assert client.strlen("msg") == 11

    def test_ttl_and_persist(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(1), resp_integer(60), resp_integer(1))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.expire("temp", 60) is True
            assert client.ttl("temp") == 60
            assert client.persist("temp") is True

    def test_expireat_and_pexpireat(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(1), resp_integer(1), resp_integer(1))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.pexpire("temp", 60000) is True
            assert client.expireat("temp", 1893456000) is True
            assert client.pexpireat("temp", 1893456000000) is True

    def test_info(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_bulk_string(b"# Server\r\nkachedb_version:0.1.0\r\n"))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            info = client.info()
            assert "kachedb_version:0.1.0" in info
