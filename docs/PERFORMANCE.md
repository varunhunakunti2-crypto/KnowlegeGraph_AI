# Local Performance and Memory

This document records **measured** behavior on the development machine and
explains how the memory-related settings trade speed for headroom. The target
hardware is a laptop with an NVIDIA RTX 4050 Laptop GPU and 16 GB of RAM, but
the numbers below are specific to this machine, its driver, its background load,
and the exact models listed. They are not guarantees for other systems.

## Hardware and software

| Item | Value |
| --- | --- |
| CPU / platform | Windows 11, Python 3.14.5 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB VRAM |
| GPU driver | 591.91 |
| System RAM | 16,202 MB total |
| LLM | `qwen3:4b` (Q4_K_M) |
| Embedding model | `nomic-embed-text` (768 dimensions) |
| Ollama | 0.40.12 with the `cuda_v12` runtime restored from the official 0.40.13 Windows archive |

The GPU measurements come from `nvidia-smi`. Whether a model is on the GPU comes
from Ollama's `/api/ps` (`size_vram > 0`). The application does **not** assume
that CUDA is active just because an NVIDIA GPU is present.

## Reproduce the measurements

From `backend/`, with Ollama running and both models pulled:

```powershell
# Default profile: MODEL_CONTEXT_LENGTH=4096, OLLAMA_NUM_BATCH=512
.\.venv\Scripts\python.exe -m scripts.benchmark_local --json benchmark-report.json

# Low-memory profile: smaller context and prompt batch
.\.venv\Scripts\python.exe -m scripts.benchmark_local `
    --context 2048 --num-batch 128 --json benchmark-low-memory.json
```

The script unloads this application's models before and after the run, samples
`nvidia-smi` and host RAM while requests are in flight, and writes a JSON report.
The console output labels measured values and derived estimates separately.
`models_loaded_after` in the JSON is sampled just before the final cleanup
unload.

The benchmark is intentionally small: one cold load, two short generation
requests, and three embedding batch sizes. It measures local model throughput,
not full-PDF ingestion time or answer quality. A real document will also spend
time on PDF parsing, chunking, database writes, and graph extraction.

## Measured results

### Default profile

`MODEL_CONTEXT_LENGTH=4096`, `OLLAMA_NUM_BATCH=512`, two generation runs of 64
output tokens. Host RAM available before the run was 3,313 MB.

| Measurement | Result |
| --- | --- |
| Cold model load | 3.14 s reported, 3.35 s wall |
| Cold-load output rate | 51.3 tok/s (8 tokens) |
| Generation median | **41.5 tok/s** (43.2 and 39.7 tok/s) |
| Prompt processing | 679 and 422 tok/s |
| GPU utilization during generation | 89.2% average, 96% max |
| Peak VRAM during generation | **4,368 MB** |
| Cold-load peak VRAM | 4,368 MB |
| Embeddings, batch 1 | 31.98 texts/s |
| Embeddings, batch 8 | **98.6 texts/s** |
| Embeddings, batch 16 | 85.45 texts/s |
| Derived estimate | 5,916 chunks/min at the best measured embedding rate |
| Derived estimate | 2,490 output tokens/min at the median generation rate |

The embedding figures are throughput for short synthetic texts. The `batch 8`
result was the best measured point; `batch 16` was slower in that run, so do not
assume that larger is always faster. Indexing defaults to small request batches
to bound memory, not to maximize a single request's throughput.

### Low-memory profile

`MODEL_CONTEXT_LENGTH=2048`, `OLLAMA_NUM_BATCH=128`, two generation runs of 64
output tokens. Host RAM available before the run was 4,012 MB.

| Measurement | Result |
| --- | --- |
| Cold model load | 3.475 s reported, 3.66 s wall |
| Cold-load output rate | 62.7 tok/s (8 tokens) |
| Generation median | **45.8 tok/s** (47.2 and 44.3 tok/s) |
| Prompt processing | 728 and 606 tok/s |
| GPU utilization during generation | 83.6% average, 95% max |
| Peak VRAM during generation | **4,001 MB** |
| Cold-load peak VRAM | 4,002 MB |
| Embeddings, batch 1 | 23.17 texts/s |
| Embeddings, batch 8 | **105.54 texts/s** |
| Embeddings, batch 16 | 120.48 texts/s |
| Derived estimate | 7,228.8 chunks/min at the best measured embedding rate |
| Derived estimate | 2,748 output tokens/min at the median generation rate |

On this run the lower-memory configuration was slightly faster and used about
370 MB less peak VRAM. That is a real observation, but not a controlled
benchmark: short generations, warm/cold model state, background GPU load, and
run-to-run variance all matter. The practical reason to use the low-memory
profile is a smaller context and request footprint, not a guaranteed speed gain.

### CPU fallback baseline

Before the local Ollama CUDA runtime was repaired, the same GPU was present but
Ollama reported the model as `100% CPU`. A run in that state produced:

| Measurement | CPU-only result |
| --- | --- |
| Cold model load | 8.275 s |
| Generation median | 7.3 tok/s |
| Embeddings, batch 8 | 18.39 texts/s |
| Peak sampled VRAM | about 1,226 MB |

That is a useful fallback baseline, not a controlled comparison: the model was
being partially loaded/paged differently, and the machine had very little free
RAM during the run. It demonstrates why the application reports the actual
processor from Ollama instead of assuming GPU acceleration.

## What the application changes

### Low-memory profile

Set `MEMORY_PROFILE=low-memory` in `.env`, or merge
`.env.low-memory.example` into the existing file. The profile changes only fields
that were not explicitly set. Explicit values in `.env` always win, and a fresh
copy of `.env.example` sets most of these fields, so comment out or delete the
profile-sensitive lines if you want the profile defaults to apply.

Profile defaults:

| Setting | Default | Low-memory |
| --- | --- | --- |
| `MODEL_CONTEXT_LENGTH` | 4096 | 2048 |
| `OLLAMA_MAX_TOKENS` | 2048 | 384 |
| `OLLAMA_NUM_BATCH` | 512 | 128 |
| `OLLAMA_KEEP_ALIVE` | `5m` | `2m` |
| `OLLAMA_IDLE_UNLOAD_SECONDS` | 900 | 120 |
| `OLLAMA_SINGLE_MODEL_RESIDENT` | `false` | `true` |
| `MAX_CONTEXT_TOKENS` | 4096 | 1536 |
| `EMBEDDING_BATCH_SIZE` | 32 | 8 |
| `BATCH_SIZE` | 16 | 8 |
| `INDEX_BATCH_SIZE` | 8 | 4 |
| `MAX_INDEX_BATCH_FAILURES` | 3 | 2 |
| `MAX_CONCURRENT_REQUESTS` | 2 | 1 |
| `MAX_CONCURRENT_UPLOADS` | 2 | 1 |
| `TOP_K_CHUNKS` | 10 | 5 |
| `MAX_EVIDENCE_ITEMS` | 20 | 8 |
| `GRAPH_TRAVERSAL_DEPTH` | 2 | 1 |
| `MAX_GRAPH_EXTRACTION_CHUNKS` | 100 | 25 |

This keeps the LLM, embedding model, browser, and IDE from competing for the same
6 GB of VRAM. It also lowers retrieval breadth and graph-extraction work, so
answers may use less evidence than in the default profile.

### Model lifecycle

- `OLLAMA_KEEP_ALIVE` controls how long Ollama keeps a model after a request.
- `OLLAMA_IDLE_UNLOAD_SECONDS` unloads this application's models after idle
  time. `0` disables idle unloading.
- `OLLAMA_SINGLE_MODEL_RESIDENT=true` evicts the other application model before
  loading a new one, so `qwen3:4b` and `nomic-embed-text` do not compete for VRAM.
- `MEMORY_GUARD_ENABLED=true` checks RAM/VRAM before a model call. Under
  pressure it unloads this application's models and retries the measurement.
- `OLLAMA_UNLOAD_ON_MEMORY_PRESSURE=true` unloads after a request when memory is
  still tight.
- `MIN_FREE_MEMORY_MB` and `MIN_FREE_VRAM_MB` define the pressure thresholds.
  The guard only unloads the app's models, never unrelated Ollama models.

When the host is critically low on RAM before a request, the API returns HTTP
503 with an actionable message rather than allowing the machine to thrash.
Ollama/CUDA allocation failures are mapped to the same class of error with the
current free RAM/VRAM in the message.

### Resumable indexing

`POST /api/v1/documents/{id}/index` processes `INDEX_BATCH_SIZE` chunks at a
time. Each successful batch is committed before the next starts. The query uses
keyset pagination (`chunk_index > last_index`), so the whole document is never
loaded into memory.

If a batch fails:

- the failed chunks are marked `failed` for a later retry;
- already indexed chunks remain committed;
- indexing stops after `MAX_INDEX_BATCH_FAILURES` consecutive failures;
- an out-of-memory error stops immediately and returns HTTP 503, preserving
  completed work.

Calling the same endpoint again resumes from the remaining chunks. Use
`GET /api/v1/documents/{id}/index/status` to see total, indexed, pending, and
failed counts.

### Bounded retrieval and graph work

`TOP_K_CHUNKS`, `TOP_K_ENTITIES`, `MAX_CONTEXT_TOKENS`, `MAX_EVIDENCE_ITEMS`,
`GRAPH_TRAVERSAL_DEPTH`, `MAX_GRAPH_TRAVERSAL_DEPTH`, `MAX_GRAPH_LIMIT`, and
`MAX_GRAPH_EXTRACTION_CHUNKS` bound retrieval and graph work. The document and
graph read paths avoid loading full page JSON or the whole graph when only
existence or a limited result set is needed.

## Tradeoff guide

| Change | Memory effect | Speed/quality effect |
| --- | --- | --- |
| Lower `MODEL_CONTEXT_LENGTH` | Less KV cache and peak VRAM | Faster prompt handling; less room for evidence |
| Lower `MAX_CONTEXT_TOKENS` | Less prompt memory | Lower retrieval breadth; may reduce answer completeness |
| Lower `OLLAMA_NUM_BATCH` | Lower prompt-processing peak VRAM | Can be slower for long prompts; may be neutral or faster for short ones |
| Lower `MAX_CONCURRENT_REQUESTS` | One model request at a time | Less contention; queued requests wait |
| `OLLAMA_SINGLE_MODEL_RESIDENT=true` | Avoids two resident models | More model reloads when switching chat/embedding/graph work |
| Lower `INDEX_BATCH_SIZE` | Smaller DB/embedding batch footprint | More round trips; easier resume |
| Lower `EMBEDDING_BATCH_SIZE` / `BATCH_SIZE` | Smaller embedding request | More requests; throughput may drop |
| Lower `TOP_K_CHUNKS` / `MAX_EVIDENCE_ITEMS` | Smaller answer context | Lower recall; less evidence in answers |
| Lower graph depth/limits | Smaller Neo4j result sets | May miss multi-hop facts |
| `OLLAMA_KEEP_ALIVE=0` or short idle unload | Frees VRAM/RAM quickly | Every request may pay a cold-load cost |

There is no single "best" value. The default profile is a good starting point on
this machine; use low-memory when the browser and IDE are also running or when
VRAM/RAM pressure is observed.

## Verifying GPU use and memory

Use all three checks; the presence of an NVIDIA GPU alone is not evidence that
Ollama is using it:

```powershell
ollama ps
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
Invoke-RestMethod http://127.0.0.1:8000/api/v1/system/status
Invoke-RestMethod http://127.0.0.1:8000/api/v1/system/gpu
```

`ollama ps` should show `100% GPU` (or a GPU/CPU split) while a model is loaded.
`/api/v1/system/status` returns measured RAM, VRAM, loaded models, pressure
reasons, and the active limits. `/api/v1/system/unload` unloads this
application's models immediately.

### If Ollama reports 100% CPU

1. Check `ollama --version` and `ollama ps`.
2. Check the Ollama installation's `lib/ollama/cuda_v12` directory. A working
   Windows CUDA runtime includes `ggml-cuda.dll`, `cudart64_12.dll`,
   `cublas64_12.dll`, `cublasLt64_12.dll`, and the Visual C++ runtime DLLs. A
   partially extracted installation can leave only a temporary file and silently
   fall back to CPU.
3. Repair or reinstall Ollama using the official Windows package, or restore the
   complete `cuda_v12` directory from the matching official Ollama Windows
   archive. Do not mix arbitrary CUDA DLL versions.
4. Restart the Ollama application/server, then confirm `ollama ps` says
   `100% GPU` and re-run the benchmark.

On this machine, fixing an incomplete `cuda_v12` installation changed the
measured generation rate from 7.3 tok/s (CPU) to 41.5 tok/s (GPU) with the
default profile. That is an environment repair, not an application optimization;
the application still works when Ollama falls back to CPU, just more slowly.

## Reproducibility caveats

- The machine had a browser, editor, and other applications running. Free RAM
  varied between runs.
- The benchmark uses short prompts and 64 output tokens. Long-context prompts
  will show a larger benefit from prompt batching and may have different peak
  VRAM.
- The first request after a model unload includes load time; warm runs do not.
- The low-memory run happened after the default run and had more free RAM at the
  start. Treat the small speed difference as noise unless you repeat it with the
  same warm-up and background load.
- The derived "chunks/min" and "tokens/min" figures are arithmetic projections
  from measured rates. They are not sustained end-to-end ingestion or generation
  measurements.
- Re-run the benchmark after changing models, drivers, Ollama versions, or
  `MODEL_CONTEXT_LENGTH`; do not reuse these numbers as a capacity plan.
