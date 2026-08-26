# 📘 vLLM Production Integration Guide with KacheDB

> **Overview:** How to configure, deploy, and scale **vLLM** with KacheDB's zero-copy POSIX shared memory (`/dev/shm`) KV-cache connector for disaggregated prefill/decode and sub-millisecond prefix caching.

---

## 1. Architecture Overview

In standard vLLM deployments, when GPU High Bandwidth Memory (HBM) fills up, older KV cache blocks are discarded, forcing expensive GPU prefill recomputation on repeat queries or multi-turn chats.

With **KacheDB**:
* Evicted PagedAttention blocks stream into `/dev/shm` in **`1.1 µs`** via asynchronous DMA.
* When matching prompts arrive, KacheDB performs 64-bit Chained Blake2b prefix matching and restores blocks directly into GPU VRAM at memory bus speeds (**`222+ GB/s`**).

```text
┌────────────────────────────────────────────────────────┐
│                   vLLM Inference Server                │
│            (Engine Core / PagedAttention)              │
├───────────────────────────┬────────────────────────────┤
│   GPU HBM (PagedBlocks)   │   kachedb.vllm.Connector   │
└─────────────┬─────────────┴─────────────┬──────────────┘
              │                           │
              ▼                           ▼
      cudaMemcpyAsync             Chained Blake2b Hash
              │                           │
              └─────────────► ◄───────────┘
               Zero-Copy POSIX Shared Memory
                    (/dev/shm/kachedb_*)
```

---

## 2. Installation

Install `kachedb` with vLLM and PyTorch extras:

```bash
pip install "kachedb[vllm,torch]"
```

---

## 3. Deployment Modes

### Mode A: vLLM CLI / Server (`vllm serve`)

To enable KacheDB KV transfer in the official vLLM OpenAI-compatible server:

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --kv-transfer-config '{
    "kv_connector": "kachedb.vllm.KacheDBConnector",
    "kv_role": "kv_both",
    "kv_buffer_size": 2048
  }'
```

#### Configuration Parameters:
* **`kv_connector`**: Set to `"kachedb.vllm.KacheDBConnector"`.
* **`kv_role`**:
  * `"kv_both"`: Node both saves computed KV blocks and retrieves cached blocks (standard setup).
  * `"kv_producer"`: Dedicated prefill disaggregated worker.
  * `"kv_consumer"`: Dedicated decode disaggregated worker.

---

### Mode B: Programmatic Python Usage

For custom serving architectures or specialized inference loops:

```python
import torch
from kachedb.vllm import KacheDBConnector

# 1. Initialize connector for local worker rank
connector = KacheDBConnector(
    rank=0,
    local_rank=0,
    block_size=16,
    pool_size_mb=2048,  # Pre-allocate 2GB /dev/shm pool
)

# 2. Check and restore prefix blocks during prefill
prompt_tokens = [1, 1024, 7592, 2005, 394]
matched_states, is_hit = connector.recv_kv_caches_and_hidden_states(
    model_executable=model,
    model_input=model_input,
    kv_caches=gpu_paged_kv_cache,
)

if is_hit:
    print(f"⚡ Bypassed GPU prefill! Restored {len(matched_states)} blocks from KacheDB.")
else:
    # Compute on GPU and offload to KacheDB for future queries
    connector.send_kv_caches_and_hidden_states(
        model_executable=model,
        model_input=model_input,
        kv_caches=gpu_paged_kv_cache,
    )
```

---

## 4. Docker Deployment (`docker-compose.yml`)

Run KacheDB alongside vLLM in a co-located container stack sharing `/dev/shm`:

```yaml
version: "3.8"

services:
  kachedb:
    image: ghcr.io/vubon/kachedb:latest
    container_name: kachedb
    ports:
      - "6379:6379"
    volumes:
      - /dev/shm:/dev/shm
    ipc: host
    restart: unless-stopped

  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm_server
    ports:
      - "8000:8000"
    ipc: host
    volumes:
      - /dev/shm:/dev/shm
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      --model meta-llama/Meta-Llama-3-8B-Instruct
      --kv-transfer-config '{"kv_connector": "kachedb.vllm.KacheDBConnector", "kv_role": "kv_both"}'
    depends_on:
      - kachedb
```

---

## 5. Performance Tuning & Verification

1. **Verify Shared Memory Mounting:**
   Ensure `/dev/shm` is mounted with sufficient capacity:
   ```bash
   df -h /dev/shm
   ```
2. **Inspect Active Connections:**
   ```bash
   redis-cli -p 6379 PING
   # Returns: PONG
   ```
