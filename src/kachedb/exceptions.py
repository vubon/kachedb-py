"""KacheDB-specific exception hierarchy."""


class KacheDBError(Exception):
    """Base exception for all KacheDB client errors."""


class ConnectionError(KacheDBError):
    """Raised when the client cannot connect or loses connection to the server."""


class TimeoutError(KacheDBError):
    """Raised when a socket operation times out."""


class ResponseError(KacheDBError):
    """Raised when the server returns a RESP error response (`-ERR ...`)."""


class ProtocolError(KacheDBError):
    """Raised when the RESP wire protocol stream is malformed or unexpected."""


class PoolExhaustedError(KacheDBError):
    """Raised when all connections in the pool are in use and none can be acquired."""
