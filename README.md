# llm-serve

`llm-serve` is a local developer gateway for text-generation models. It exposes both OpenAI-compatible (`/v1/*`) and Ollama-compatible (`/api/*`) APIs on `127.0.0.1:11424`, enforces a model allowlist, supports one active model at a time with warmup/switch semantics, and includes disk-backed batch processing. Inference can run either through local Hugging Face models served by `vLLM` or through an upstream Ollama server.

See [api_usage.md](api_usage.md) for the external API contract and [docs/codeflow.md](docs/codeflow.md) for the internal request flow.

## Quickstart

1. Copy `.env.example` to `.env` and adjust `INFERENCE_BACKEND`, the default model env vars for your backend, and the allowlisted model IDs you want to serve.
2. Install the project:
   - vLLM runtime plus tests: `uv sync --extra runtime --extra dev`
   - Ollama or mock runtime plus tests: `uv sync --extra dev`
3. Start the server: `./run-server.sh`

The default mock-safe code path exists to keep tests lightweight. For real model serving, use either `INFERENCE_BACKEND=vllm` for direct local serving or `INFERENCE_BACKEND=ollama` to proxy an Ollama daemon running at `OLLAMA_BASE_URL`.

## Notes

- The server binds to `127.0.0.1:11424` by default.
- Interactive requests are capped by `PROMPT_MAX_PARALLEL`, which defaults to `8`. Batch items are capped by `BATCH_MAX_PARALLEL`, which defaults to `4`.
- `/healthz` reports `foreground_active`/`foreground_capacity` and `batch_active`/`batch_capacity`, so you can see whether work is actively overlapping or only queueing.
- Batch uploads, metadata, and generated outputs are stored under `STORAGE_ROOT`.
- In `vllm` mode, model downloads and Hugging Face cache files are stored under `MODEL_CACHE_DIR`, which defaults to the repo-local `data/models/`.
- `run-server.sh` is the canonical API entrypoint for `/v1/*`, `/api/*`, files, batches, and health. It is the shell wrapper around `llm-serve`.
- `run-qwen35-27b.sh` is only for direct raw `vllm serve` debugging or benchmarking. It does not expose the full `llm-serve` API surface.
- `run-server.sh` does not auto-install the vLLM optional dependency set. If you run `INFERENCE_BACKEND=vllm`, install it first with `uv sync --extra runtime`.
- In `ollama` mode, `llm-serve` treats Ollama as an already-running upstream service. It proxies inference, model tags, and optional `/api/pull` calls to `OLLAMA_BASE_URL` and does not host models or use Hugging Face/vLLM GPU settings locally.
- For the Qwen/vLLM path, see [docs/qwen-vllm.md](docs/qwen-vllm.md) for the canonical model ID, reasoning behavior, and 2-GPU tuning notes.
- With large Ollama-backed models such as `gpt-oss:120b`, higher `PROMPT_MAX_PARALLEL` usually buys less aggregate throughput than it costs in single-request token rate. If responsiveness matters more than bulk throughput, lower `PROMPT_MAX_PARALLEL` and/or choose a smaller model.
- `ENABLE_THINKING` (default `false`) is a global toggle for Qwen thinking mode on the vLLM backend. When `false`, the chat template always receives `enable_thinking=False` regardless of `reasoning_effort` in the request. When `true`, requests that include `reasoning_effort` activate thinking mode. The model must still be in `REASONING_MODEL_ALLOWLIST` for `reasoning_effort` to be accepted. The Ollama backend does not consult this toggle — it passes reasoning controls through to upstream.
- `VLLM_DEFAULT_MODEL_ID` and `OLLAMA_DEFAULT_MODEL_ID` let each backend use its own startup/request default model. If a backend-specific value is unset, `DEFAULT_MODEL_ID` is used as the fallback.
- Direct non-batch Ollama requests use `OLLAMA_REQUEST_TIMEOUT_SECONDS` and can retry once with `OLLAMA_REQUEST_TIMEOUT_RETRY_MULTIPLIER` before returning a timeout.
- Timed-out Ollama batch items do not fail the whole batch immediately. `llm-serve` retries them once after the first pass using `OLLAMA_BATCH_TIMEOUT_RETRY_MULTIPLIER` and `OLLAMA_BATCH_RETRY_OUTPUT_TOKENS_MULTIPLIER`, then writes the final consolidated result into the original batch artifacts.
- On startup, the service can run a self-test prompt such as a thousand-word poem after the default model loads. By default this runs in the background so the API becomes ready after model load, and `/healthz` reports `queued`, `running`, `passed`, or `failed` along with completion tokens, latency, and tokens-per-second.
- Startup logs also show when the effective default model load begins, when the self-test is queued or started, and whether it passed or failed. If the selected default-model env var is misconfigured, startup fails with the bad model ID and the current allowlist in the error text.
- Set `STARTUP_SELF_TEST_BLOCKING=true` if you want startup readiness to wait for that generation to finish and fail fast on a broken generation path.
- `VLLM_GPU_AUTO_SELECT=true` makes the first vLLM model load inspect all host GPUs with `nvidia-smi`, choose the best `VLLM_GPU_COUNT` GPUs by free memory, and set `CUDA_VISIBLE_DEVICES` automatically.
- `VLLM_GPU_MEMORY_UTILIZATION` is a preferred cap, not a fixed requirement. If the selected GPUs cannot safely support that value, the server derives a lower `gpu_memory_utilization` from current free memory, `VLLM_GPU_MEMORY_RESERVE_FRACTION`, and `VLLM_GPU_MEMORY_UTILIZATION_MIN`.
- If you disable auto-selection, `CUDA_VISIBLE_DEVICES` is used directly and `VLLM_GPU_MEMORY_UTILIZATION` stays fixed.
- `VLLM_GPU_COUNT` still maps to vLLM `tensor_parallel_size`. It must be compatible with the model; for example, a setting of `3` can fail on models whose tensor-parallel dimensions are not divisible by `3`.
- `VLLM_DTYPE` defaults to `bfloat16`, which is optimal for Qwen3.5-27B inference speed on modern GPUs.
