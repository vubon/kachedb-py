"""Unit tests for the synchronous KacheClient using mock server."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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

    def test_dbsize(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(42))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.dbsize() == 42

    def test_type(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_simple_string("string"),
            resp_simple_string("vector"),
            resp_simple_string("none"),
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.type("str_key") == "string"
            assert client.type("vec_key") == "vector"
            assert client.type("missing") == "none"

    def test_flushdb_and_flushall(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("OK"), resp_simple_string("OK"))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.flushdb() is True
            assert client.flushall() is True


class TestKacheClientAdvanced:
    def test_client_per_call_and_disconnect_all(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_simple_string("PONG"),
            resp_simple_string("PONG"),  # For pool health check on re-checkout
            resp_bulk_string(b"hello"),
        )
        port = mock_server.start()

        client = KacheClient(port=port, decode_responses=True)
        # Call without connect() or with client:
        assert client.ping() == "PONG"
        assert client.ping(message="hello") == "hello"
        client.disconnect_all()

    def test_client_execute_error_discards_conn(self) -> None:
        client = KacheClient(port=6379)
        mock_conn = MagicMock()
        mock_conn.send_command.side_effect = RuntimeError("socket error")
        client._conn = mock_conn

        with pytest.raises(RuntimeError, match="socket error"):
            client._execute("PING")

        mock_conn.disconnect.assert_called_once()

    def test_client_mset_empty_and_numeric_branches(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_integer(8),
            resp_integer(5),
            resp_integer(50000),
            resp_bulk_string(b"# Server\r\nversion:0.1.0"),
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.mset({}) is True
            assert client.decr("counter", 2) == 8
            assert client.decrby("counter", 3) == 5
            assert client.pttl("temp") == 50000
            info = client.info(section="server")
            assert "version:0.1.0" in info

    def test_client_batch_vectors(self, mock_server: MockKacheDBServer) -> None:
        import struct

        mock_server.program_responses(
            resp_integer(2),
            resp_array(
                resp_array(
                    resp_array(
                        resp_bulk_string(b"doc1"),
                        resp_bulk_string(b"0.99"),
                        resp_bulk_string(b"res1"),
                    )
                )
            ),
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.vadd_batch("idx", []) == 0
            assert client.vsearch_batch("idx", []) == []

            with pytest.raises(TypeError, match="Unsupported vector type"):
                client.vadd_batch("idx", [("id1", 12345, "payload")])  # type: ignore[list-item]

            with pytest.raises(TypeError, match="Unsupported vector type"):
                client.vsearch_batch("idx", [12345])  # type: ignore[list-item]

            v_bytes = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
            items = [
                ("doc1", [0.1, 0.2, 0.3, 0.4], "res1"),
                ("doc2", v_bytes, None),
            ]
            added = client.vadd_batch("idx", items, ex=60)
            assert added == 2

            search_res = client.vsearch_batch("idx", [[0.1, 0.2, 0.3, 0.4], v_bytes])
            assert len(search_res) == 1
            assert search_res[0][0][0] == b"doc1"

    def test_client_vector_types_and_errors(self, mock_server: MockKacheDBServer) -> None:
        import struct

        mock_server.program_responses(
            resp_integer(1),
            resp_array(
                resp_array(
                    resp_bulk_string(b"doc1"),
                    resp_bulk_string(b"0.95"),
                    resp_bulk_string(b"p1"),
                )
            ),
            resp_bulk_string(b"ERROR_NOT_A_LIST"),
            resp_bulk_string(b"ERROR_NOT_A_LIST"),
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            with pytest.raises(TypeError, match="Unsupported vector type"):
                client.vadd("idx", "doc1", "invalid_type")  # type: ignore[arg-type]

            with pytest.raises(TypeError, match="Unsupported vector type"):
                client.vsearch("idx", "invalid_type")  # type: ignore[arg-type]

            raw_v = struct.pack("<2f", 1.0, 2.0)
            assert client.vadd("idx", "doc1", raw_v) is True

            res = client.vsearch("idx", raw_v)
            assert len(res) == 1
            assert res[0][0] == b"doc1"

            assert client.vstats("idx") is None
            assert client.vindex_info("idx") is None

    def test_client_auth_and_admin(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_simple_string("OK"),
            resp_simple_string("OK"),
            resp_bulk_string(b"Background append only file rewriting started"),
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.auth("secret") is True
            assert client.auth("secret", username="user1") is True
            bg = client.bgrewriteaof()
            assert "rewriting started" in bg

    def test_client_vector_and_index_lifecycle(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_simple_string("OK"),
            resp_array(
                resp_array(
                    resp_bulk_string(b"doc1"),
                    resp_bulk_string(b"0.98"),
                    resp_bulk_string(b"payload"),
                )
            ),
            resp_array(
                resp_bulk_string(b"dimension"),
                resp_integer(128),
                resp_bulk_string(b"total_vectors"),
                resp_integer(1),
            ),
            resp_integer(1),
            resp_simple_string("OK"),
            resp_array(
                resp_bulk_string(b"dimension"),
                resp_integer(128),
                resp_bulk_string(b"metric"),
                resp_bulk_string(b"cosine"),
            ),
            resp_integer(1),
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.vadd("idx", "doc1", [0.1, 0.2], payload="payload", ex=3600) is True
            matches = client.vsearch("idx", [0.1, 0.2])
            assert len(matches) == 1
            assert matches[0][0] == b"doc1"

            st = client.vstats("idx")
            assert st == {"dimension": 128, "total_vectors": 1}

            assert client.vdel("idx", "doc1") is True

            assert client.vindex_create("vidx", 128) is True
            info = client.vindex_info("vidx")
            assert info == {"dimension": 128, "metric": "cosine"}
            assert client.vindex_drop("vidx") is True
