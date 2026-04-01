# llm-serve

`llm-serve` is a local developer gateway for Hugging Face text-generation models served through `vLLM`. It exposes both OpenAI-compatible (`/v1/*`) and Ollama-compatible (`/api/*`) APIs on `127.0.0.1:11424`, enforces a model allowlist, supports one active model at a time with warmup/switch semantics, and includes disk-backed batch processing.

See [api_usage.md](api_usage.md) for the external API contract and [docs/codeflow.md](docs/codeflow.md) for the internal request flow.

## Quickstart

1. Copy `.env.example` to `.env` and adjust the allowlisted Hugging Face model IDs you want to serve.
2. Install the project:
   - Runtime plus tests: `uv sync --extra runtime --extra dev`
   - Minimal API/test stack without `vLLM`: `uv sync --extra dev`
3. Start the server: `uv run llm-serve`

The default mock-safe code path exists to keep tests lightweight, but actual model serving should run with `INFERENCE_BACKEND=vllm`.

## Notes

- The server binds to `127.0.0.1:11424` by default.
- Interactive requests are capped by `PROMPT_MAX_PARALLEL`, which defaults to `8`.
- Batch uploads, metadata, and generated outputs are stored under `STORAGE_ROOT`.
- Model downloads and Hugging Face cache files are stored under `MODEL_CACHE_DIR`, which defaults to the repo-local `data/models/`.
- On startup, the service can run a self-test prompt such as a thousand-word poem after the default model loads. By default this runs in the background so the API becomes ready after model load, and `/healthz` reports `queued`, `running`, `passed`, or `failed` along with completion tokens, latency, and tokens-per-second.
- Set `STARTUP_SELF_TEST_BLOCKING=true` if you want startup readiness to wait for that generation to finish and fail fast on a broken generation path.
- `VLLM_GPU_AUTO_SELECT=true` makes the first vLLM model load inspect all host GPUs with `nvidia-smi`, choose the best `VLLM_GPU_COUNT` GPUs by free memory, and set `CUDA_VISIBLE_DEVICES` automatically.
- `VLLM_GPU_MEMORY_UTILIZATION` is a preferred cap, not a fixed requirement. If the selected GPUs cannot safely support that value, the server derives a lower `gpu_memory_utilization` from current free memory, `VLLM_GPU_MEMORY_RESERVE_FRACTION`, and `VLLM_GPU_MEMORY_UTILIZATION_MIN`.
- If you disable auto-selection, `CUDA_VISIBLE_DEVICES` is used directly and `VLLM_GPU_MEMORY_UTILIZATION` stays fixed.
- `VLLM_GPU_COUNT` still maps to vLLM `tensor_parallel_size`. It must be compatible with the model; for example, a setting of `3` can fail on models whose tensor-parallel dimensions are not divisible by `3`.
