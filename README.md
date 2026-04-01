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
- Use `CUDA_VISIBLE_DEVICES` to choose which GPUs the process may see, and `VLLM_GPU_COUNT` to set how many of those visible GPUs vLLM should shard the model across.
