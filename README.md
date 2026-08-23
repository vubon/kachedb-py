# kachedb

<p align="center">
  <strong>High-Performance Python Client for KacheDB — The Zero-Copy Redis-Compatible & LLM KV-Cache Storage Engine</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/kachedb/"><img src="https://img.shields.io/pypi/v/kachedb.svg" alt="PyPI Version"/></a>
  <a href="https://pypi.org/project/kachedb/"><img src="https://img.shields.io/pypi/pyversions/kachedb.svg" alt="Python Versions"/></a>
  <a href="https://github.com/vubon/kachedb-py/actions"><img src="https://github.com/vubon/kachedb-py/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://github.com/vubon/kachedb-py/blob/main/LICENSE-MIT"><img src="https://img.shields.io/badge/license-Apache--2.0%20%2F%20MIT-blue.svg" alt="License"/></a>
</p>

---

## ⚡ Installation

```bash
pip install kachedb
```

With PyTorch tensor support:

```bash
pip install kachedb[torch]
```

## 🚀 Quickstart

### Synchronous Client

```python
from kachedb import KacheClient

with KacheClient(host="127.0.0.1", port=6379) as client:
    # Standard Redis-compatible operations
    client.set("user:1", "alice", ex=3600)     # SET with 1-hour TTL
    print(client.get("user:1"))                 # b"alice"

    # Batch operations
    client.set("user:2", "bob")
    result = client.mget("user:1", "user:2")    # [b"alice", b"bob"]

    # Check existence
    print(client.exists("user:1"))              # 1

    # Delete
    client.delete("user:1", "user:2")
```

### Async Client

```python
import asyncio
from kachedb import AsyncKacheClient

async def main():
    async with AsyncKacheClient(host="127.0.0.1", port=6379) as client:
        await client.set("key", "value", ex=60)
        result = await client.get("key")
        print(result)  # b"value"

asyncio.run(main())
```

### Pipeline Batching

Reduce network round-trips by batching multiple commands:

```python
from kachedb import KacheClient

with KacheClient() as client:
    pipe = client.pipeline()
    pipe.set("a", "1")
    pipe.set("b", "2")
    pipe.set("c", "3")
    pipe.get("a")
    pipe.get("b")
    pipe.get("c")

    results = pipe.execute()
    # ["OK", "OK", "OK", b"1", b"2", b"3"]
```

### Async Pipeline

```python
from kachedb import AsyncKacheClient

async def main():
    async with AsyncKacheClient() as client:
        pipe = client.pipeline()
        pipe.set("x", "10")
        pipe.get("x")
        results = await pipe.execute()
        # ["OK", b"10"]
```

### Zero-Copy Tensor Access (LLM KV-Cache)

Read KV-cache tensors directly from KacheDB's shared memory with **zero data copying**:

```python
from kachedb import read_tensor, read_torch_tensor

# Read as numpy array (zero-copy via /dev/shm)
np_tensor = read_tensor(core_id=0, byte_offset=0)
print(np_tensor.shape, np_tensor.dtype)

# Read as PyTorch tensor (requires: pip install kachedb[torch])
torch_tensor = read_torch_tensor(core_id=0, byte_offset=0)
print(torch_tensor.shape, torch_tensor.dtype)
```

## 📋 Supported Commands

All commands follow the [KacheDB RESP2/RESP3 wire protocol](https://github.com/vubon/kachedb):

| Command | Method | Description |
| :--- | :--- | :--- |
| `PING` | `client.ping()` | Test server liveness |
| `SET` | `client.set(key, value, ex=, px=)` | Store value with optional TTL |
| `GET` | `client.get(key)` | Retrieve value |
| `MGET` | `client.mget(*keys)` | Batch retrieve multiple keys |
| `DEL` | `client.delete(*keys)` | Delete keys |
| `EXISTS` | `client.exists(*keys)` | Count existing keys |

## 🏗️ Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                     Your Python App                      │
│             (vLLM / SGLang / FastAPI / etc.)             │
├──────────────────────────────────────────────────────────┤
│                   kachedb Python SDK                     │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ KacheClient  │  │  AsyncKache  │  │    Pipeline    │  │
│  │  (sync TCP)  │  │    Client    │  │    Batching    │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬────────┘  │
│         │                 │                  │           │
│  ┌──────┴─────────────────┴──────────────────┴────────┐  │
│  │            RESP2/RESP3 Protocol Engine             │  │
│  │         (64KB buffered encoder + decoder)          │  │
│  └──────────────────────┬─────────────────────────────┘  │
│                         │                                │
│  ┌──────────────────────┴─────────────────────────────┐  │
│  │        ConnectionPool / AsyncConnectionPool        │  │
│  │    (Thread-safe / asyncio.Queue, health checks)    │  │
│  └──────────────────────┬─────────────────────────────┘  │
├─────────────────────────┼────────────────────────────────┤
│                   TCP + /dev/shm                         │
├──────────────────────────────────────────────────────────┤
│                  KacheDB Server (Rust)                   │
│         io_uring / kqueue │ POSIX SHM │ Megaslab         │
└──────────────────────────────────────────────────────────┘
```

## 🔧 Connection Pool

The client automatically manages a connection pool:

```python
from kachedb import KacheClient

# Pool with up to 20 connections
client = KacheClient(
    host="127.0.0.1",
    port=6379,
    max_connections=20,
    socket_timeout=5.0,
)
```

## 🧪 Development

```bash
# Clone
git clone https://github.com/vubon/kachedb-py.git
cd kachedb-py

# Install in dev mode
pip install -e ".[dev]"

# Run unit tests
pytest tests/ -v --ignore=tests/test_integration.py

# Run integration tests (requires running KacheDB server)
pytest tests/test_integration.py -v

# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Type check
mypy src/kachedb/
```

## 🔗 Related Projects

- **[KacheDB Server](https://github.com/vubon/kachedb)** — The Rust storage engine

## 📄 License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
