# Changelog

All notable changes to the `kachedb` Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0b1] — 2026-08-30

### Added
- **Complete Redis TTL Lifecycle (`kachedb.KacheClient` & `kachedb.AsyncKacheClient`):**
  - Added `expire`, `pexpire`, `expireat`, `pexpireat`, `ttl`, `pttl`, and `persist`.
- **Extended Redis Command Primitives:**
  - Added atomic operations: `mset`, `incr`, `decr`, `incrby`, `decrby`, `append`, `strlen`, and `info`.
- **Async Vector & Semantic Operations (`AsyncKacheClient`):**
  - Added asynchronous vector search and index operations: `vadd`, `vsearch`, `vdel`, and `vstats`.
- **Asynchronous Semantic Cache (`kachedb.semantic.AsyncSemanticCache`):**
  - Added high-level async intent matching and LLM response caching engine.
- **Full Pipeline Support (`Pipeline` & `AsyncPipeline`):**
  - Added support for all extended Redis primitives and TTL lifecycle commands in batch pipelines.

---

## [0.1.0a5] — 2026-08-28

### Added
- **In-Memory Semantic Cache Engine (`kachedb.semantic`):**
  - Added `SemanticCache` class for intent-based LLM query caching matching by cosine similarity ($< 50\ \mu\text{s}$ cache resolution) with automatic TTL eviction and threshold tuning.
  - Added `SearchResult` data class encapsulating matched key, similarity score, and cached response payload.
- **Pluggable Embedding Providers (`kachedb.semantic.embedders`):**
  - `TransformersEmbedder`: Native HuggingFace `AutoModel` with mean pooling over token embeddings.
  - `FastEmbedAdapter`: Lightweight ONNX Runtime embedding generator via `fastembed`.
  - `SentenceTransformersAdapter`: HuggingFace `SentenceTransformer` adapter.
  - `OpenAIAdapter`: Remote embedding integration with OpenAI (`text-embedding-3-small`).
  - `CallableAdapter`: Wrapper for any custom user embedding function (`Callable[[str], list[float]]`).
  - `MockEmbedder`: Deterministic, zero-dependency token and semantic-cluster pseudo-embedder for local tests and simulations.
- **Native Vector Client Operations (`kachedb.KacheClient`):**
  - `client.vadd(index, item_id, vector, payload, ex)`: Store vector embedding with optional metadata and TTL.
  - `client.vsearch(index, query_vector, top_k, threshold)`: Nearest-neighbor search returning `[(item_id, similarity, payload), ...]`.
  - `client.vdel(index, item_id)`: Remove vector from named index.
  - `client.vstats(index)`: Retrieve index dimension, active vector count, and RAM footprint.
- **Test Suite & Type Checking:** Added `tests/test_semantic_cache.py` and `tests/test_semantic_standalone.py` with full type annotations, mypy compliance, and ruff formatting (102/102 tests passing).

---

## [0.1.0a4] — 2026-08-26

### Added
- **SGLang (RadixAttention) KV-Cache Connector (`kachedb.sglang`):**
  - Added `KacheDBSGLangConnector` for microsecond-tier tree node offloading and prefix restoration via POSIX Shared Memory (`/dev/shm`).
  - Added `KacheDBRadixAdapter` with chained Blake2b hashing for variable-length token slices ($H_k = \text{Blake2b}(H_{\text{parent}} \,\|\, \text{len} \,\|\, \text{tokens})$).
  - Added `RadixNodeDescriptor` for tracking tree node IDs, layer offsets, parent hashes, and metadata.
- **Universal Multi-Precision DMA (`kachedb.dma` & `TensorCodec`):**
  - Added `TensorCodec` in `kachedb.descriptor` with dynamic registry mapping for `torch.bfloat16` (BF16), `torch.float16` (FP16), `torch.float32` (FP32), and `torch.int8` (INT8).
  - Refactored `KacheDBMemoryManager` into top-level `kachedb.dma`, universally shared across `kachedb.vllm` and `kachedb.sglang` with 100% backward compatibility.
- **Production Integration Guides:**
  - Added `docs/guides/vllm_integration_guide.md` with multi-GPU CLI flags, programmatic loops, and production `docker-compose.yml`.
  - Added `docs/guides/sglang_integration_guide.md` with RadixAttention tree-branching and multi-precision configurations.
- **Unit Test Suite:** Added `tests/test_sglang_connector.py` covering variable-length hashing, single/multi-node offloads, branching tree resolution, and multi-precision FP16/BF16/FP32/INT8 bit-exact roundtrips (85/85 tests passing).
- **Packaging:** Added `sglang = ["torch>=2.0", "sglang>=0.3.0"]` extra to `pyproject.toml`.

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
