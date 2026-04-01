# Codeflow

## Entry Points

- CLI: `llm-serve` -> [`src/llm_serve/main.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/main.py)
- ASGI app factory: [`src/llm_serve/app.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/app.py)

## GPU Configuration

- `CUDA_VISIBLE_DEVICES` limits which physical GPUs the service can see.
- `VLLM_GPU_COUNT` sets `tensor_parallel_size` for vLLM, so one model can be spread across multiple visible GPUs.
- Validation rejects `VLLM_GPU_COUNT` values larger than the number of configured visible devices.
- Apply these settings before starting the server; they are startup/runtime initialization settings, not request-level controls.

## Interactive Request Flow

1. FastAPI receives either `/v1/*` or `/api/*` input and validates it with the request schemas.
2. The route normalizes the public request into one internal `InferenceRequest`.
3. The runtime manager checks the configured allowlists, token caps, and active-model state.
4. If the requested model is not active, the runtime manager starts an async switch and the route returns `202`.
5. If the model is active, the runtime manager enforces the configured concurrency lane and dispatches to the active backend.
6. The backend returns text or text chunks, and the route re-encodes them into OpenAI SSE or Ollama NDJSON/output JSON.

Relevant files:

- [`src/llm_serve/app.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/app.py)
- [`src/llm_serve/runtime/manager.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/runtime/manager.py)
- [`src/llm_serve/runtime/vllm_backend.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/runtime/vllm_backend.py)

## Model Switching Flow

1. The active model is tracked by the runtime manager.
2. A request or `/api/pull` against another allowlisted model triggers a background switch task.
3. The switch waits for in-flight backend usage to drain, starts the new backend, swaps the active model, and shuts down the old backend.
4. During the switch, requests for the target model receive `202`; conflicting switch requests receive `409`.

## Batch Flow

1. `/v1/files` stores the uploaded JSONL file on disk.
2. `/v1/batches` creates a persisted batch record plus output and error artifact files.
3. The batch manager restores queued work on startup and runs each JSONL line through the same runtime manager used for interactive requests.
4. Each batch item writes either a success line into the output file or an error line into the error file.
5. `/v1/batches/{id}` and `/v1/files/{id}/content` expose status and generated artifacts.

Relevant files:

- [`src/llm_serve/batch.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/batch.py)
- [`src/llm_serve/storage.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/storage.py)
