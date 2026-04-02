# llm-serve

`llm-serve` is a local developer gateway for text-generation models. It exposes both OpenAI-compatible (`/v1/*`) and Ollama-compatible (`/api/*`) APIs on `127.0.0.1:11424`, enforces a model allowlist, supports one active model at a time with warmup/switch semantics, and includes disk-backed batch processing. Inference can run either through local Hugging Face models served by `vLLM` or through an upstream Ollama server.

See [api_usage.md](api_usage.md) for the external API contract and [docs/codeflow.md](docs/codeflow.md) for the internal request flow.

## Quickstart

1. Copy `.env.example` to `.env` and adjust `INFERENCE_BACKEND`, `DEFAULT_MODEL_ID`, and the allowlisted model IDs you want to serve.
2. Install the project:
   - vLLM runtime plus tests: `uv sync --extra runtime --extra dev`
   - Ollama or mock runtime plus tests: `uv sync --extra dev`
3. Start the server: `uv run llm-serve`

The default mock-safe code path exists to keep tests lightweight. For real model serving, use either `INFERENCE_BACKEND=vllm` for direct local serving or `INFERENCE_BACKEND=ollama` to proxy an Ollama daemon running at `OLLAMA_BASE_URL`.

## Notes

- The server binds to `127.0.0.1:11424` by default.
- Interactive requests are capped by `PROMPT_MAX_PARALLEL`, which defaults to `8`.
- Batch uploads, metadata, and generated outputs are stored under `STORAGE_ROOT`.
- In `vllm` mode, model downloads and Hugging Face cache files are stored under `MODEL_CACHE_DIR`, which defaults to the repo-local `data/models/`.
- In `ollama` mode, `llm-serve` treats Ollama as an already-running upstream service. It proxies inference, model tags, and optional `/api/pull` calls to `OLLAMA_BASE_URL` and does not host models or use Hugging Face/vLLM GPU settings locally.
- Direct non-batch Ollama requests use `OLLAMA_REQUEST_TIMEOUT_SECONDS` and can retry once with `OLLAMA_REQUEST_TIMEOUT_RETRY_MULTIPLIER` before returning a timeout.
- Timed-out Ollama batch items do not fail the whole batch immediately. `llm-serve` retries them once after the first pass using `OLLAMA_BATCH_TIMEOUT_RETRY_MULTIPLIER` and `OLLAMA_BATCH_RETRY_OUTPUT_TOKENS_MULTIPLIER`, then writes the final consolidated result into the original batch artifacts.
- On startup, the service can run a self-test prompt such as a thousand-word poem after the default model loads. By default this runs in the background so the API becomes ready after model load, and `/healthz` reports `queued`, `running`, `passed`, or `failed` along with completion tokens, latency, and tokens-per-second.
- Startup logs also show when the default model load begins, when the self-test is queued or started, and whether it passed or failed. If `DEFAULT_MODEL_ID` is misconfigured, startup fails with the bad model ID and the current allowlist in the error text.
- Set `STARTUP_SELF_TEST_BLOCKING=true` if you want startup readiness to wait for that generation to finish and fail fast on a broken generation path.
- `VLLM_GPU_AUTO_SELECT=true` makes the first vLLM model load inspect all host GPUs with `nvidia-smi`, choose the best `VLLM_GPU_COUNT` GPUs by free memory, and set `CUDA_VISIBLE_DEVICES` automatically.
- `VLLM_GPU_MEMORY_UTILIZATION` is a preferred cap, not a fixed requirement. If the selected GPUs cannot safely support that value, the server derives a lower `gpu_memory_utilization` from current free memory, `VLLM_GPU_MEMORY_RESERVE_FRACTION`, and `VLLM_GPU_MEMORY_UTILIZATION_MIN`.
- If you disable auto-selection, `CUDA_VISIBLE_DEVICES` is used directly and `VLLM_GPU_MEMORY_UTILIZATION` stays fixed.
- `VLLM_GPU_COUNT` still maps to vLLM `tensor_parallel_size`. It must be compatible with the model; for example, a setting of `3` can fail on models whose tensor-parallel dimensions are not divisible by `3`.
