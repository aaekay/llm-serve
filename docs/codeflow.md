# Codeflow

## Entry Points

- CLI: `llm-serve` -> [`src/llm_serve/main.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/main.py)
- ASGI app factory: [`src/llm_serve/app.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/app.py)

## Inference Modes

- `INFERENCE_BACKEND=vllm` serves Hugging Face models directly through the local vLLM runtime.
- `INFERENCE_BACKEND=ollama` treats Ollama as an already-running upstream service and proxies inference, optional pulls, and model tags to `OLLAMA_BASE_URL`.
- `INFERENCE_BACKEND=mock` exists only for tests and lightweight local verification.

## GPU Configuration

- These settings only apply in `vllm` mode.
- `VLLM_GPU_AUTO_SELECT=true` makes the first vLLM model load inspect all host GPUs with `nvidia-smi`.
- The backend ranks GPUs by current free memory, chooses the best `VLLM_GPU_COUNT` GPUs, and exports `CUDA_VISIBLE_DEVICES` for the vLLM process.
- `VLLM_GPU_MEMORY_UTILIZATION` is treated as the preferred cap. If the chosen GPUs do not have enough free memory for that cap, the backend derives a lower safe utilization using `VLLM_GPU_MEMORY_RESERVE_FRACTION`.
- If the derived utilization would fall below `VLLM_GPU_MEMORY_UTILIZATION_MIN`, startup fails with an insufficient-memory error instead of overcommitting.
- When `VLLM_GPU_AUTO_SELECT=false`, the backend uses the configured `CUDA_VISIBLE_DEVICES` directly and keeps `VLLM_GPU_MEMORY_UTILIZATION` fixed.
- `VLLM_GPU_COUNT` still sets vLLM `tensor_parallel_size`, so model compatibility constraints still apply.

## Model Cache

- This cache is only used in `vllm` mode.
- `MODEL_CACHE_DIR` defaults to the repo-local `data/models/`.
- Before vLLM is initialized, the backend creates that directory and exports `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` under it.
- As a result, model weights and Hugging Face cache artifacts stay inside the repo-local cache tree unless the config is changed.

## Startup Self-Test

- After the default model is loaded, the runtime can run a configured startup prompt to verify real generation.
- The default prompt is a thousand-word poem request so generation health is tested with a non-trivial output.
- By default the self-test is launched in the background, so FastAPI startup completes after model load and `/healthz` reports `queued`, `running`, `passed`, or `failed`.
- The runtime records completion tokens, latency, and tokens-per-second and exposes them via `/healthz`.
- The runtime also logs default-model load, self-test queue/start transitions, and the final pass/fail result so background failures are visible without polling `/healthz`.
- When `STARTUP_SELF_TEST_BLOCKING=true`, startup instead waits for the self-test to finish and fails immediately if generation fails.

## Interactive Request Flow

1. FastAPI receives either `/v1/*` or `/api/*` input and validates it with the request schemas.
2. The route normalizes the public request into one internal `InferenceRequest`.
3. The runtime manager checks the configured allowlists, token caps, and active-model state.
4. If the requested model is not active, the runtime manager starts an async switch and the route returns `202`.
5. If the model is active, the runtime manager enforces the configured concurrency lane and dispatches to the active backend.
6. In `ollama` mode, direct non-batch requests use an upstream timeout policy with one longer retry before returning a timeout.
7. The backend either runs vLLM locally or proxies to Ollama, then returns text or text chunks for the route to re-encode into OpenAI SSE or Ollama NDJSON/output JSON.

Relevant files:

- [`src/llm_serve/app.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/app.py)
- [`src/llm_serve/runtime/manager.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/runtime/manager.py)
- [`src/llm_serve/runtime/vllm_backend.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/runtime/vllm_backend.py)
- [`src/llm_serve/runtime/ollama_backend.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/runtime/ollama_backend.py)

## Model Switching Flow

1. The active model is tracked by the runtime manager.
2. A request or `/api/pull` against another allowlisted model triggers a background switch task.
3. The switch waits for in-flight backend usage to drain, starts the new backend, swaps the active model, and shuts down the old backend.
4. In `ollama` mode, `/api/pull` is only an optional convenience proxy to the upstream Ollama daemon; normal inference still assumes the model may already be available there.
5. In `vllm` mode, the first backend load may inspect host GPUs and derive a safe vLLM memory-utilization ratio before the engine starts.
6. During the switch, requests for the target model receive `202`; conflicting switch requests receive `409`.

## Batch Flow

1. `/v1/files` stores the uploaded JSONL file on disk.
2. `/v1/batches` creates a persisted batch record plus output and error artifact files.
3. The batch manager restores queued work on startup and runs each JSONL line through the same runtime manager used for interactive requests.
4. In `ollama` mode, a timed-out item is held back from the error artifact during the first pass, then retried once with a larger upstream timeout and larger internal output-token limit.
5. After the retry pass, each item writes exactly one final success line into the output file or one final error line into the error file.
6. `/v1/batches/{id}` and `/v1/files/{id}/content` expose status and generated artifacts.

Relevant files:

- [`src/llm_serve/batch.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/batch.py)
- [`src/llm_serve/storage.py`](/Users/aaekay/Documents/projects/llm-serve/src/llm_serve/storage.py)
