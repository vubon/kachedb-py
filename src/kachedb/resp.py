"""
RESP2/RESP3 wire protocol encoder and streaming decoder.

Matches the KacheDB server's ``kachedb-proto-resp`` crate for byte-level
compatibility.  All encoding produces standard RESP2 arrays of bulk strings.
The decoder handles: simple strings, errors, integers, bulk strings, null,
and arrays (recursive).

Performance note: The ``RespReader`` reads in 64 KB chunks (matching the
server's ``READ_BUF_SIZE``), not byte-by-byte.
"""

from __future__ import annotations

from typing import Union

from .exceptions import ProtocolError, ResponseError

# Type alias for decoded RESP values.
RespValue = Union[str, bytes, int, list["RespValue"], None]  # noqa: UP007

# Default socket read buffer size — matches KacheDB server ``READ_BUF_SIZE``.
_READ_CHUNK_SIZE = 64 * 1024


def encode_command(args: list[str | bytes]) -> bytes:
    """Encode a command as a RESP array of bulk strings.

    >>> encode_command(["SET", "key", "value"])
    b'*3\\r\\n$3\\r\\nSET\\r\\n$3\\r\\nkey\\r\\n$5\\r\\nvalue\\r\\n'
    """
    parts: list[bytes] = [f"*{len(args)}\r\n".encode()]
    for arg in args:
        raw = arg.encode("utf-8") if isinstance(arg, str) else arg
        parts.append(f"${len(raw)}\r\n".encode())
        parts.append(raw)
        parts.append(b"\r\n")
    return b"".join(parts)


def encode_commands(command_list: list[list[str | bytes]]) -> bytes:
    """Encode multiple commands into a single pipelined byte buffer.

    This is used by the ``Pipeline`` class to batch-send commands.
    """
    return b"".join(encode_command(args) for args in command_list)


class RespReader:
    """Buffered streaming RESP2/RESP3 decoder.

    Reads from a socket-like object in large chunks and parses complete
    frames.  Handles partial frames gracefully by retaining unconsumed
    bytes across ``read_response`` calls.

    Parameters
    ----------
    sock : socket.socket
        Connected TCP socket to read from.
    chunk_size : int
        Number of bytes to request per ``recv`` call.  Defaults to 64 KB.
    """

    __slots__ = ("_buf", "_chunk_size", "_pos", "_sock")

    def __init__(self, sock: socket.socket, chunk_size: int = _READ_CHUNK_SIZE) -> None:  # noqa: F821
        self._sock = sock
        self._buf = bytearray()
        self._pos = 0
        self._chunk_size = chunk_size

    # ── Public API ────────────────────────────────────────────────────────

    def read_response(self) -> RespValue:
        """Read and decode one complete RESP frame from the socket."""
        return self._parse_frame()

    # ── Internal Parser ───────────────────────────────────────────────────

    def _ensure_data(self, needed: int = 1) -> None:
        """Ensure at least *needed* bytes are available after ``_pos``."""
        while (len(self._buf) - self._pos) < needed:
            chunk = self._sock.recv(self._chunk_size)
            if not chunk:
                raise ProtocolError("Connection closed while reading RESP frame")
            self._buf.extend(chunk)

    def _read_line(self) -> bytes:
        """Read bytes until ``\\r\\n``, returning the line *without* the CRLF."""
        while True:
            idx = self._buf.find(b"\r\n", self._pos)
            if idx != -1:
                line = bytes(self._buf[self._pos : idx])
                self._pos = idx + 2
                self._compact()
                return line
            # Need more data.
            chunk = self._sock.recv(self._chunk_size)
            if not chunk:
                raise ProtocolError("Connection closed while reading RESP line")
            self._buf.extend(chunk)

    def _read_exact(self, n: int) -> bytes:
        """Read exactly *n* bytes from the buffer, fetching more if needed."""
        self._ensure_data(n)
        data = bytes(self._buf[self._pos : self._pos + n])
        self._pos += n
        self._compact()
        return data

    def _compact(self) -> None:
        """Discard consumed bytes when the cursor has advanced far enough."""
        if self._pos > 4096:
            del self._buf[: self._pos]
            self._pos = 0

    def _parse_frame(self) -> RespValue:
        """Parse a single RESP frame (recursive for arrays)."""
        self._ensure_data(1)
        marker = self._buf[self._pos : self._pos + 1]
        self._pos += 1

        if marker == b"+":
            # Simple string
            return self._read_line().decode("utf-8")

        if marker == b"-":
            # Error
            msg = self._read_line().decode("utf-8")
            raise ResponseError(msg)

        if marker == b":":
            # Integer
            return int(self._read_line())

        if marker == b"$":
            # Bulk string
            length = int(self._read_line())
            if length == -1:
                return None
            data = self._read_exact(length)
            self._read_exact(2)  # consume trailing \r\n
            return data

        if marker == b"*":
            # Array
            count = int(self._read_line())
            if count == -1:
                return None
            return [self._parse_frame() for _ in range(count)]

        if marker == b"_":
            # RESP3 null
            self._read_line()  # consume \r\n
            return None

        raise ProtocolError(f"Unknown RESP type marker: {marker!r}")


class AsyncRespReader:
    """Async buffered RESP decoder for ``asyncio.StreamReader``.

    Mirrors :class:`RespReader` but uses ``await reader.read()`` instead
    of ``socket.recv()``.
    """

    __slots__ = ("_buf", "_chunk_size", "_pos", "_reader")

    def __init__(
        self,
        reader: asyncio.StreamReader,  # noqa: F821
        chunk_size: int = _READ_CHUNK_SIZE,
    ) -> None:
        self._reader = reader
        self._buf = bytearray()
        self._pos = 0
        self._chunk_size = chunk_size

    async def read_response(self) -> RespValue:
        """Read and decode one complete RESP frame."""
        return await self._parse_frame()

    async def _ensure_data(self, needed: int = 1) -> None:
        while (len(self._buf) - self._pos) < needed:
            chunk = await self._reader.read(self._chunk_size)
            if not chunk:
                raise ProtocolError("Connection closed while reading RESP frame")
            self._buf.extend(chunk)

    async def _read_line(self) -> bytes:
        while True:
            idx = self._buf.find(b"\r\n", self._pos)
            if idx != -1:
                line = bytes(self._buf[self._pos : idx])
                self._pos = idx + 2
                self._compact()
                return line
            chunk = await self._reader.read(self._chunk_size)
            if not chunk:
                raise ProtocolError("Connection closed while reading RESP line")
            self._buf.extend(chunk)

    async def _read_exact(self, n: int) -> bytes:
        await self._ensure_data(n)
        data = bytes(self._buf[self._pos : self._pos + n])
        self._pos += n
        self._compact()
        return data

    def _compact(self) -> None:
        if self._pos > 4096:
            del self._buf[: self._pos]
            self._pos = 0

    async def _parse_frame(self) -> RespValue:
        await self._ensure_data(1)
        marker = self._buf[self._pos : self._pos + 1]
        self._pos += 1

        if marker == b"+":
            return (await self._read_line()).decode("utf-8")

        if marker == b"-":
            msg = (await self._read_line()).decode("utf-8")
            raise ResponseError(msg)

        if marker == b":":
            return int(await self._read_line())

        if marker == b"$":
            length = int(await self._read_line())
            if length == -1:
                return None
            data = await self._read_exact(length)
            await self._read_exact(2)
            return data

        if marker == b"*":
            count = int(await self._read_line())
            if count == -1:
                return None
            return [await self._parse_frame() for _ in range(count)]

        if marker == b"_":
            await self._read_line()
            return None

        raise ProtocolError(f"Unknown RESP type marker: {marker!r}")
