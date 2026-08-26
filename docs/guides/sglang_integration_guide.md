# 📗 SGLang (RadixAttention) Production Integration Guide with KacheDB

> **Overview:** How to integrate, configure, and benchmark **SGLang** with KacheDB's `kachedb.sglang` connector for dynamic Radix tree subtree offloading, cross-request prefix caching, and multi-turn prefill acceleration.

---

## 1. Architecture Overview

**SGLang** organizes KV caches as a dynamic **Radix Tree (RadixAttention)** where each node holds a variable-length slice of token IDs.

`KacheDBSGLangConnector` hooks into SGLang's `RadixCache` eviction and lookup lifecycles:
* **Offload:** When GPU memory is constrained, evicted tree nodes are serialized with 64-byte `TensorBlockDescriptor` headers and streamed into KacheDB's `/dev/shm` shared memory pool in microseconds.
* **Restore:** Incoming prompt prefixes traverse the cached Radix tree; matching subtrees are restored into GPU buffers via asynchronous DMA, bypassing full-sequence attention prefill passes.

```text
┌────────────────────────────────────────────────────────┐
│                   SGLang Inference Server              │
│                (RadixAttention Engine Core)            │
├───────────────────────────┬────────────────────────────┤
│     GPU Radix Buffers     │  kachedb.sglang.Connector  │
└─────────────┬─────────────┴─────────────┬──────────────┘
              │                           │
              ▼                           ▼
      cudaMemcpyAsync              Chained Blake2b
              │                           │
              └─────────────► ◄───────────┘
               Zero-Copy POSIX Shared Memory
                    (/dev/shm/kachedb_*)
```

---

## 2. Installation

Install `kachedb` with SGLang and PyTorch extras:

```bash
pip install "kachedb[sglang,torch]"
```

---

## 3. Integration & Usage

### 3.1 Basic Usage

```python
import torch
from kachedb.sglang import KacheDBSGLangConnector

# Initialize connector
connector = KacheDBSGLangConnector(
    rank=0,
    local_rank=0,
    pool_size_mb=2048,  # 2GB zero-copy shared memory
)

# Offload an evicted Radix node (e.g. 32 tokens across all model layers)
node_token_ids = [101, 2054, 2003, 1037, 7592, ...]  # Variable length
desc = connector.offload_node(
    node_id=1,
    token_ids=node_token_ids,
    k_tensors=layer_k_tensors,
    v_tensors=layer_v_tensors,
    parent_hash=0,  # Root node
)

# Restore matching prefix for incoming prompt
matched_tokens, is_hit = connector.restore_prefix(
    prompt_tokens=incoming_prompt_token_ids,
    target_k_buffers=target_k_buffers,
    target_v_buffers=target_v_buffers,
)

if is_hit:
    print(f"⚡ Restored {matched_tokens} tokens from KacheDB in sub-millisecond RAM!")
```

---

## 4. Multi-Turn Dialogue Tree Branching

In multi-turn chat and agent interactions, subtrees branch from common root prompts (e.g. system instructions):

```text
[Node 1: System Prompt (2,048 tok)] ──► Hash: 0x8F4A... (Stored in KacheDB)
        │
        ├──► [Node 2: Turn 1 User Query (512 tok)] ──► Hash: 0x3B19...
        │
        └──► [Node 3: Turn 2 Branch (768 tok)] ──────► Hash: 0x9E21...
```

`KacheDBRadixAdapter` automatically handles chained Blake2b variable-length token hashing:
$$H_{\text{node}} = \text{Blake2b}(H_{\text{parent}} \,\|\, \text{len} \,\|\, \text{tokens})$$

Ensuring that shared system prompts are stored **only once** and shared across all conversation branches.

---

## 5. Multi-Precision Support (BF16, FP16, FP32, INT8)

KacheDB's `TensorCodec` automatically handles all precision formats without configuration changes:
* **`torch.bfloat16`** (Default for LLaMA 3, Gemma, Mistral)
* **`torch.float16`** (Standard half precision)
* **`torch.float32`** (CPU verification)
* **`torch.int8`** (Quantized KV-cache)

---

## 6. Benchmarking & Verification

Run the master benchmark suite to measure your local hardware speedup:

```bash
python experiments/benchmark_sglang_kachedb.py
```
