# Changelog

All notable changes to the `kachedb` Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0a3] — 2026-08-25

### Added
- **vLLM Distributed KV-Transfer Connector (`kachedb.vllm`):**
  - Added `KacheDBConnector` subclassing vLLM's `KVConnectorBase` for zero-copy PagedAttention block offloading and restoration.
  - Added `KacheDBPrefixCache` for 64-bit deterministic chained token chunk hashing ($H_k = \text{hash}(H_{k-1} + \text{tokens})$).
  - Added `KacheDBMemoryManager` for microsecond-tier PagedAttention block swapping via POSIX Shared Memory (`/dev/shm`).
- **Unit Test Suite:** Added `tests/test_vllm_connector.py` covering block hash determinism, hash chaining, longest prefix search, and multi-layer PagedAttention tensor restoration (76/76 tests passing).
- **Packaging Dependencies:** Added `vllm = ["torch>=2.0", "vllm>=0.6.0"]` and `all = [...]` optional extras in `pyproject.toml`.
- **Documentation:** Added vLLM CLI launch guide (`vllm serve ... --kv-transfer-config`) and programmatic connector usage in `README.md`.

---

## [0.1.0a2] — 2026-08-25

### Added
- **TensorBlockDescriptor Serialization:**
  - Added `.to_bytes()` method to serialize 64-byte C-ABI descriptor headers to binary buffers.
- **LLM Precision Support:**
  - Expanded `TensorDType` enum to cover `FP32`, `FP16`, `BF16` (bfloat16), `FP8E4M3`, `FP8E5M2`, `INT8`, and `INT4` precisions.

---

## [0.1.0a1] — 2026-08-23

### Added
- **Initial Alpha Release of `kachedb` Python SDK:**
  - Synchronous `KacheClient` supporting `PING`, `GET`, `SET` (with `EX`/`PX` TTL), `MGET`, `DEL`, and `EXISTS`.
  - Asynchronous `AsyncKacheClient` with non-blocking `asyncio` networking.
  - Thread-safe `ConnectionPool` and `AsyncConnectionPool` with automatic health checks.
  - `Pipeline` and `AsyncPipeline` for batching multiple commands in a single TCP round-trip.
  - Zero-allocation RESP2/RESP3 streaming protocol reader (`RespReader` and `AsyncRespReader`).
  - Zero-copy tensor memory mapping utilities (`attach_shm`, `read_tensor`, `read_torch_tensor`) via Linux `/dev/shm`.
  - Full PEP 561 type safety annotations with `py.typed` marker.
