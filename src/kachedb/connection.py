"""
Low-level TCP connection to a KacheDB server.

Manages the socket lifecycle, RESP encoding, and buffered response reading.
Used internally by :class:`~kachedb.client.KacheClient` and the connection pool.
"""

from __future__ import annotations

import builtins
import contextlib
import socket

from .exceptions import ConnectionError, TimeoutError
from .resp import RespReader, RespValue, encode_command


class Connection:
    """A single TCP connection to a KacheDB daemon.

    Parameters
    ----------
    host : str
        Server hostname or IP address.
    port : int
        Server TCP port.
    socket_timeout : float | None
        Socket timeout in seconds.  ``None`` for blocking without timeout.
    decode_responses : bool
        If ``True``, decode byte responses to UTF-8 strings.
    """

    __slots__ = (
        "_reader",
        "_sock",
        "decode_responses",
        "host",
        "port",
        "socket_timeout",
    )

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        *,
        socket_timeout: float | None = 5.0,
        decode_responses: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.socket_timeout = socket_timeout
        self.decode_responses = decode_responses
        self._sock: socket.socket | None = None
        self._reader: RespReader | None = None

    @property
    def is_connected(self) -> bool:
        """Return ``True`` if the socket is open."""
        return self._sock is not None

    def connect(self) -> None:
        """Establish TCP connection to the KacheDB server."""
        if self._sock is not None:
            return

        try:
            sock = socket.create_connection(
                (self.host, self.port),
                timeout=self.socket_timeout,
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = sock
            self._reader = RespReader(sock)
        except OSError as exc:
            raise ConnectionError(
                f"Failed to connect to KacheDB at {self.host}:{self.port}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        """Close the TCP connection."""
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None
            self._reader = None

    def send_command(self, *args: str | bytes) -> None:
        """Encode and send a single RESP command."""
        if self._sock is None:
            raise ConnectionError("Not connected to KacheDB")

        data = encode_command(list(args))
        try:
            self._sock.sendall(data)
        except builtins.TimeoutError as exc:
            self.disconnect()
            raise TimeoutError(f"Timeout sending command to KacheDB: {exc}") from exc
        except OSError as exc:
            self.disconnect()
            raise ConnectionError(f"Error sending command to KacheDB: {exc}") from exc

    def send_packed(self, data: bytes) -> None:
        """Send pre-encoded bytes (used by pipeline)."""
        if self._sock is None:
            raise ConnectionError("Not connected to KacheDB")

        try:
            self._sock.sendall(data)
        except builtins.TimeoutError as exc:
            self.disconnect()
            raise TimeoutError(f"Timeout sending data to KacheDB: {exc}") from exc
        except OSError as exc:
            self.disconnect()
            raise ConnectionError(f"Error sending data to KacheDB: {exc}") from exc

    def read_response(self) -> RespValue:
        """Read and decode one RESP response."""
        if self._reader is None:
            raise ConnectionError("Not connected to KacheDB")

        try:
            response = self._reader.read_response()
        except builtins.TimeoutError as exc:
            self.disconnect()
            raise TimeoutError(f"Timeout reading response from KacheDB: {exc}") from exc
        except OSError as exc:
            self.disconnect()
            raise ConnectionError(f"Error reading response from KacheDB: {exc}") from exc

        if self.decode_responses and isinstance(response, bytes):
            return response.decode("utf-8")

        return response

    def check_health(self) -> bool:
        """Send a PING and verify PONG response.  Returns ``False`` on any error."""
        try:
            self.send_command("PING")
            response = self.read_response()
            return response == "PONG"
        except Exception:
            return False
