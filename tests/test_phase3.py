"""Unit tests for Phase 3 Python SDK enhancements.

Tests SQ8 quantization, VINDEX commands, BGREWRITEAOF, and AUTH methods
for both synchronous and asynchronous clients.
"""

from __future__ import annotations

import math

from kachedb import (
    AsyncKacheClient,
    KacheClient,
    sq8_decode,
    sq8_encode,
)
from tests.conftest import (
    MockKacheDBServer,
    resp_array,
    resp_bulk_string,
    resp_integer,
    resp_simple_string,
)


class TestSQ8Quantizer:
    def test_empty_vector(self) -> None:
        raw, min_v, max_v = sq8_encode([])
        assert raw == b""
        assert min_v == 0.0
        assert max_v == 0.0
        assert sq8_decode(raw, min_v, max_v) == []

    def test_constant_vector(self) -> None:
        vec = [3.5, 3.5, 3.5, 3.5]
        raw, min_v, max_v = sq8_encode(vec)
        assert len(raw) == 4
        assert min_v == 3.5
        assert max_v == 3.5
        decoded = sq8_decode(raw, min_v, max_v)
        assert len(decoded) == 4
        for x in decoded:
            assert math.isclose(x, 3.5, rel_tol=1e-5)

    def test_quantization_accuracy(self) -> None:
        vec = [-1.0, -0.5, 0.0, 0.5, 1.0]
        raw, min_v, max_v = sq8_encode(vec)
        assert len(raw) == len(vec)
        assert raw[0] == 0
        assert raw[-1] == 255

        decoded = sq8_decode(raw, min_v, max_v)
        assert len(decoded) == len(vec)
        for orig, recon in zip(vec, decoded, strict=True):
            # Max error for 256 buckets across range 2.0 is 2.0 / 255 / 2 ~ 0.004
            assert abs(orig - recon) < 0.02


class TestSyncPhase3Commands:
    def test_auth(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("OK"))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            assert client.auth("supersecret") is True

    def test_bgrewriteaof(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_simple_string("Background append only file rewriting started")
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            res = client.bgrewriteaof()
            assert "started" in res

    def test_vindex_create(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_simple_string("OK"))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            res = client.vindex_create(
                "hnsw_idx",
                dim=128,
                m=16,
                ef_construction=200,
                ef_search=50,
                metric="COSINE",
                quantization="SQ8",
            )
            assert res is True

    def test_vindex_drop(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(resp_integer(1))
        port = mock_server.start()

        with KacheClient(port=port) as client:
            res = client.vindex_drop("hnsw_idx")
            assert res is True

    def test_vindex_info(self, mock_server: MockKacheDBServer) -> None:
        mock_server.program_responses(
            resp_array(
                resp_bulk_string(b"name"),
                resp_bulk_string(b"hnsw_idx"),
                resp_bulk_string(b"type"),
                resp_bulk_string(b"hnsw"),
                resp_bulk_string(b"dimension"),
                resp_integer(128),
            )
        )
        port = mock_server.start()

        with KacheClient(port=port) as client:
            info = client.vindex_info("hnsw_idx")
            assert info is not None
            assert info["name"] == "hnsw_idx"
            assert info["type"] == "hnsw"
            assert info["dimension"] == 128


class TestClientSSLConfig:
    def test_client_init_ssl_options(self) -> None:
        client = KacheClient(
            password="secretpassword",
            ssl=True,
            ssl_keyfile="/tmp/key.pem",
            ssl_certfile="/tmp/cert.pem",
            ssl_ca_certs="/tmp/ca.pem",
            ssl_check_hostname=False,
        )
        assert client.password == "secretpassword"
        assert client.ssl is True
        assert client._pool.password == "secretpassword"
        assert client._pool.ssl is True
        assert client._pool.ssl_keyfile == "/tmp/key.pem"
        assert client._pool.ssl_certfile == "/tmp/cert.pem"
        assert client._pool.ssl_ca_certs == "/tmp/ca.pem"
        assert client._pool.ssl_check_hostname is False

    def test_async_client_init_ssl_options(self) -> None:
        async_client = AsyncKacheClient(
            password="secretpassword",
            ssl=True,
            ssl_keyfile="/tmp/key.pem",
            ssl_certfile="/tmp/cert.pem",
            ssl_ca_certs="/tmp/ca.pem",
            ssl_check_hostname=False,
        )
        assert async_client.password == "secretpassword"
        assert async_client.ssl is True
        assert async_client._pool.password == "secretpassword"
        assert async_client._pool.ssl is True
        assert async_client._pool.ssl_keyfile == "/tmp/key.pem"
        assert async_client._pool.ssl_certfile == "/tmp/cert.pem"
        assert async_client._pool.ssl_ca_certs == "/tmp/ca.pem"
        assert async_client._pool.ssl_check_hostname is False
