"""Shared test fixtures and helpers for kachedb test suite."""

from __future__ import annotations

import contextlib
import socket
import threading
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


class MockKacheDBServer:
    """Minimal in-process mock that speaks RESP for unit testing.

    Accepts one connection at a time and echoes back pre-programmed responses.
    """

    def __init__(self) -> None:
        self._server_sock: socket.socket | None = None
        self._client_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._responses: list[bytes] = []
        self._received_commands: list[bytes] = []
        self.host = "127.0.0.1"
        self.port = 0  # OS assigns free port

    def program_responses(self, *responses: bytes) -> None:
        """Queue raw RESP response bytes to send back."""
        self._responses.extend(responses)

    def start(self) -> int:
        """Start the mock server and return the assigned port."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, 0))
        self.port = self._server_sock.getsockname()[1]
        self._server_sock.listen(1)
        self._server_sock.settimeout(5.0)

        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self.port

    def _serve(self) -> None:
        assert self._server_sock is not None
        try:
            conn, _ = self._server_sock.accept()
            self._client_sock = conn
            conn.settimeout(5.0)

            # Read incoming data (commands from client).
            try:
                while True:
                    data = conn.recv(65536)
                    if not data:
                        break
                    self._received_commands.append(data)

                    # Send programmed responses.
                    if self._responses:
                        response = self._responses.pop(0)
                        conn.sendall(response)
            except (OSError, TimeoutError):
                pass
        except (OSError, TimeoutError):
            pass

    def stop(self) -> None:
        """Shut down the mock server."""
        if self._client_sock:
            with contextlib.suppress(OSError):
                self._client_sock.close()
        if self._server_sock:
            with contextlib.suppress(OSError):
                self._server_sock.close()
        if self._thread:
            self._thread.join(timeout=2.0)


@pytest.fixture
def mock_server() -> Generator[MockKacheDBServer, None, None]:
    """Fixture that provides a fresh mock KacheDB server per test."""
    server = MockKacheDBServer()
    yield server
    server.stop()


# ── RESP Response Helpers ─────────────────────────────────────────────────


def resp_simple_string(s: str) -> bytes:
    """Encode a RESP simple string response."""
    return f"+{s}\r\n".encode()


def resp_error(msg: str) -> bytes:
    """Encode a RESP error response."""
    return f"-{msg}\r\n".encode()


def resp_integer(n: int) -> bytes:
    """Encode a RESP integer response."""
    return f":{n}\r\n".encode()


def resp_bulk_string(data: bytes) -> bytes:
    """Encode a RESP bulk string response."""
    return f"${len(data)}\r\n".encode() + data + b"\r\n"


def resp_null() -> bytes:
    """Encode a RESP null bulk string."""
    return b"$-1\r\n"


def resp_array(*elements: bytes) -> bytes:
    """Encode a RESP array from pre-encoded elements."""
    header = f"*{len(elements)}\r\n".encode()
    return header + b"".join(elements)
