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

### Installation Extras:

```bash
# PyTorch tensor zero-copy support (FP16, BF16, FP32, INT8)
pip install "kachedb[torch]"

# vLLM PagedAttention KV-transfer plugin
pip install "kachedb[vllm]"

# SGLang RadixAttention KV-cache plugin
pip install "kachedb[sglang]"

# Install all plugins & dependencies
pip install "kachedb[all]"
```

---

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

---

## 🧠 LLM KV-Cache Acceleration (vLLM & SGLang)

KacheDB serves as a high-speed, zero-copy **L1/L2 KV-Cache Tier** for AI inference engines, bypassing quadratic transformer attention prefill passes via POSIX shared memory (`/dev/shm`):

### 1. 🔌 vLLM PagedAttention Integration

Launch vLLM with the KacheDB KV connector:

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --kv-transfer-config '{"kv_connector": "kachedb.vllm.KacheDBConnector", "kv_role": "kv_both"}'
```

Programmatic Usage:
```python
from kachedb.vllm import KacheDBConnector

connector = KacheDBConnector(rank=0, local_rank=0, block_size=16)

# Restore prefix blocks directly into GPU PagedAttention buffers
matched_states, is_hit = connector.recv_kv_caches_and_hidden_states(
    model_executable=model,
    model_input=model_input,
    kv_caches=gpu_kv_caches,
)
```
👉 *Read the full [vLLM Production Integration Guide](docs/guides/vllm_integration_guide.md).*

---

### 2. 🌳 SGLang RadixAttention Integration

Use `KacheDBSGLangConnector` to offload and restore dynamic Radix tree branches:

```python
from kachedb.sglang import KacheDBSGLangConnector

connector = KacheDBSGLangConnector(rank=0, local_rank=0, pool_size_mb=2048)

# 1. Offload an evicted Radix tree node (Variable-length token slice)
desc = connector.offload_node(
    node_id=node.id,
    token_ids=node.token_ids,
    k_tensors=node_k_tensors,
    v_tensors=node_v_tensors,
    parent_hash=parent_hash,
)

# 2. Restore cached prefix subtree directly into target GPU/CPU memory
matched_tokens, is_hit = connector.restore_prefix(
    prompt_tokens=incoming_prompt_token_ids,
    target_k_buffers=target_k_buffers,
    target_v_buffers=target_v_buffers,
)
```
👉 *Read the full [SGLang Production Integration Guide](docs/guides/sglang_integration_guide.md).*

---

## ⚡ Master Proof-of-Speed Benchmarks

Evaluated on **Meta-Llama-3-8B Topology** (32 Layers, 8 KV Heads, FP16) connected to the live KacheDB storage engine:

| Context Length | KV Cache Size | 🔴 Cold GPU Recompute | 🟢 SGLang + KacheDB | ⚡ Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **512 tokens** | 64.0 MB | `2,429.8 ms` | **`7.99 ms`** | **`304.1×`** ⚡ |
| **1,024 tokens** | 128.0 MB | `607.9 ms` | **`7.17 ms`** | **`84.8×`** ⚡ |
| **2,048 tokens** | 256.0 MB | `1,425.1 ms` | **`9.61 ms`** | **`148.3×`** ⚡ |
| **4,096 tokens** | 512.0 MB | `3,164.2 ms` | **`10.14 ms`** | **`312.2×`** ⚡ |
| **8,192 tokens** | 1,024.0 MB | `5,541.6 ms` | **`9.25 ms`** | **`599.1×`** ⚡ |
| **16,384 tokens** | 2,048.0 MB (2GB) | `26,081.1 ms` (26.1s) | **`18.71 ms`** | **`1,393.9×`** ⚡ |

---

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

---

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

---

## 🧪 Development & Quality Gates

```bash
# Clone
git clone https://github.com/vubon/kachedb-py.git
cd kachedb-py

# Install in editable dev mode with all extras
pip install -e ".[all,dev]"

# Run full unit test suite (85 tests)
pytest tests/ -v

# Code formatting & linting
ruff check src/ tests/
ruff format --check src/ tests/

# Strict type checking
mypy src/
```

---

## 🔗 Documentation & Guides

- 📘 [vLLM Production Integration Guide](docs/guides/vllm_integration_guide.md)
- 📗 [SGLang Production Integration Guide](docs/guides/sglang_integration_guide.md)
- 🏆 [Master Proof-of-Speed Benchmarks](https://github.com/vubon/database/blob/main/experiments/master_sglang_kachedb_speed_benchmark.md)
- 🦀 [KacheDB Rust Server Engine](https://github.com/vubon/kachedb)

---

## 📄 License

Dual-licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
at your option.
