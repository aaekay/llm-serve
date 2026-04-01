# Codeflow

## Entry Points

- CLI: `llm-serve` -> [`src/llm_serve/main.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/main.py)
- ASGI app factory: [`src/llm_serve/app.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/app.py)

## GPU Configuration

- `VLLM_GPU_AUTO_SELECT=true` makes the first vLLM model load inspect all host GPUs with `nvidia-smi`.
- The backend ranks GPUs by current free memory, chooses the best `VLLM_GPU_COUNT` GPUs, and exports `CUDA_VISIBLE_DEVICES` for the vLLM process.
- `VLLM_GPU_MEMORY_UTILIZATION` is treated as the preferred cap. If the chosen GPUs do not have enough free memory for that cap, the backend derives a lower safe utilization using `VLLM_GPU_MEMORY_RESERVE_FRACTION`.
- If the derived utilization would fall below `VLLM_GPU_MEMORY_UTILIZATION_MIN`, startup fails with an insufficient-memory error instead of overcommitting.
- When `VLLM_GPU_AUTO_SELECT=false`, the backend uses the configured `CUDA_VISIBLE_DEVICES` directly and keeps `VLLM_GPU_MEMORY_UTILIZATION` fixed.
- `VLLM_GPU_COUNT` still sets vLLM `tensor_parallel_size`, so model compatibility constraints still apply.

## Model Cache

- `MODEL_CACHE_DIR` defaults to the repo-local `data/models/`.
- Before vLLM is initialized, the backend creates that directory and exports `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` under it.
- As a result, model weights and Hugging Face cache artifacts stay inside the repo-local cache tree unless the config is changed.

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
4. During the first backend load only, adaptive GPU selection may inspect host GPUs and derive a safe vLLM memory-utilization ratio before the engine starts.
5. During the switch, requests for the target model receive `202`; conflicting switch requests receive `409`.

## Batch Flow

1. `/v1/files` stores the uploaded JSONL file on disk.
2. `/v1/batches` creates a persisted batch record plus output and error artifact files.
3. The batch manager restores queued work on startup and runs each JSONL line through the same runtime manager used for interactive requests.
4. Each batch item writes either a success line into the output file or an error line into the error file.
5. `/v1/batches/{id}` and `/v1/files/{id}/content` expose status and generated artifacts.

Relevant files:

- [`src/llm_serve/batch.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/batch.py)
- [`src/llm_serve/storage.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/storage.py)
