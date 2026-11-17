# llm-serve

`llm-serve` is a local text-generation gateway with OpenAI-compatible (`/v1/*`)
and Ollama-compatible (`/api/*`) APIs. It supports mock, upstream Ollama,
upstream llama.cpp, and local vLLM inference, plus disk-backed batch processing.
The vLLM and llama.cpp paths support agent tool calling; the supervised vLLM
path also supports the stateful OpenAI Responses API.

## Start

1. Create the local configuration:

   ```bash
   cp .env.example .env
   ```

2. Install with `uv`:

   ```bash
   # Mock, Ollama, or an upstream llama.cpp server
   uv sync --extra dev

   # Local vLLM
   uv sync --extra runtime --extra dev
   ```

3. Start the API:

   ```bash
   ./run-server.sh
   ```

The default address is `http://127.0.0.1:11424`.

## Runtime

- `INFERENCE_BACKEND=mock` provides lightweight local development and tests.
- `INFERENCE_BACKEND=ollama` proxies an existing server at `OLLAMA_BASE_URL`.
- `INFERENCE_BACKEND=llamacpp` proxies an existing OpenAI-compatible
  `llama-server` at `LLAMACPP_BASE_URL`.
- `INFERENCE_BACKEND=vllm` starts supervised local vLLM workers.
- `VLLM_GPU_MODE=manual|fixed|auto` changes GPU topology selection; every mode
  uses the same worker lifecycle, request path, and cleanup behavior.
- Models and Hugging Face cache files live under `MODEL_CACHE_DIR` (`models/`).
- Uploads, batches, outputs, and tuning profiles live under `STORAGE_ROOT`
  (`data/runtime/`).

`run-qwen35-27b.sh` starts raw `vllm serve` for debugging and benchmarking. It
does not expose the complete `llm-serve` API.

`run-vllm-batch.sh <preset>` starts raw `vllm serve` on one GPU, tuned for offline
batch throughput: 64 concurrent sequences, prefix caching, 16k context, offline
model cache, bound to 127.0.0.1, with Prometheus `/metrics`. Use it when a client
sends thousands of independent requests and does its own load balancing. The
gateway (`run-server.sh`) is built for interactive use and serialises requests.
Presets: `gemma4-31b`, `qwen36-35b-a3b-fp8`, `qwen36-35b-a3b`, `qwen36-27b-fp8`, `qwen38-27b-fp8`.
Models are served offline from `models/hub` (Hugging Face cache layout). Download a new one once, with internet, e.g.
`HF_HOME=models uvx --from huggingface-hub hf download Qwen/Qwen3.8-27B-FP8` (about 29 GB).

`run-vllm-batch-file.sh <preset> [prefix]` runs offline batches with `vllm run-batch`
(no HTTP server, no tunnel): every `batch_io/in/<prefix>*.jsonl` (OpenAI batch format)
becomes `batch_io/out/<name>.out.jsonl`; finished files are skipped on a re-run.
`batch_io/` is git-ignored; inputs may contain PHI, so copy them with scp/rsync,
never git, and delete them after the results are copied back.

`run-batch-jobs.sh` runs a small batch job server on 127.0.0.1 (default port 8000):
clients `POST /jobs?preset=<preset>&name=<label>` with an OpenAI batch JSONL body,
poll `GET /jobs/<id>`, download `GET /jobs/<id>/output` and `DELETE /jobs/<id>`.
Jobs run one at a time through `run-vllm-batch-file.sh`, so the tunnel is only
needed while uploading and downloading, and the GPU is free between jobs.

## Documentation

- [API usage](api_usage.md)
- [Hermes, OpenCode, and Pi provider setup](docs/agent-providers.md)
- [Request and startup flow](docs/codeflow.md)
- [Qwen with vLLM](docs/qwen-vllm.md)
- [Gemma 4 autotuning](docs/gemma4-autotune.md)
- [Muse Glimmer 30B on H100 with llama.cpp](docs/muse-glimmer-llamacpp.md)
- [Batch progress](docs/batch-progress.md)
- [Version-control policy](docs/version-control.md)

## Verify

```bash
uv run pytest -q
uv run ruff check src tests
bash -n run-server.sh
bash -n run-qwen35-27b.sh
uv run python scripts/agent_provider_smoke.py  # requires a running vLLM server
```
