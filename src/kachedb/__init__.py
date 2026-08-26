"""
KacheDB — Python client for the KacheDB zero-copy storage engine.

Install::

    pip install kachedb
    pip install kachedb[torch]  # for PyTorch tensor support

Quickstart::

    from kachedb import KacheClient

    with KacheClient(host="127.0.0.1", port=6379) as client:
        client.set("user:1", "alice", ex=3600)
        print(client.get("user:1"))  # b"alice"

Async::

    from kachedb import AsyncKacheClient

    async with AsyncKacheClient() as client:
        await client.set("key", "value")
        print(await client.get("key"))
"""

from ._version import __version__
from .async_client import AsyncKacheClient
from .client import KacheClient
from .connection import Connection
from .descriptor import (
    TENSOR_DESCRIPTOR_MAGIC,
    TensorBlockDescriptor,
    TensorCodec,
    TensorDType,
)
from .dma import KacheDBMemoryManager
from .exceptions import (
    ConnectionError,
    KacheDBError,
    PoolExhaustedError,
    ProtocolError,
    ResponseError,
    TimeoutError,
)
from .pipeline import AsyncPipeline, Pipeline
from .pool import AsyncConnectionPool, ConnectionPool
from .sglang import KacheDBRadixAdapter, KacheDBSGLangConnector
from .tensor import attach_shm, detach_all, read_tensor, read_torch_tensor
from .vllm import KacheDBConnector, KacheDBPrefixCache

__all__ = [
    "TENSOR_DESCRIPTOR_MAGIC",
    "AsyncConnectionPool",
    "AsyncKacheClient",
    "AsyncPipeline",
    "Connection",
    "ConnectionError",
    "ConnectionPool",
    "KacheClient",
    "KacheDBConnector",
    "KacheDBError",
    "KacheDBMemoryManager",
    "KacheDBPrefixCache",
    "KacheDBRadixAdapter",
    "KacheDBSGLangConnector",
    "Pipeline",
    "PoolExhaustedError",
    "ProtocolError",
    "ResponseError",
    "TensorBlockDescriptor",
    "TensorCodec",
    "TensorDType",
    "TimeoutError",
    "__version__",
    "attach_shm",
    "detach_all",
    "read_tensor",
    "read_torch_tensor",
]
