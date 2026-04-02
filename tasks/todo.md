# Implementation Plan

- [x] Translate the approved plan into concrete build tasks.
- [x] Scaffold the Python package, CLI entrypoint, dependency metadata, and environment config.
- [x] Implement configuration loading, allowlist enforcement, token limits, and shared error handling.
- [x] Implement runtime management for one active model, warmup/model switching, and health state.
- [x] Implement `/v1/*` and `/api/*` endpoints with streaming and non-stream responses.
- [x] Implement disk-backed files and batch jobs with startup restoration behavior.
- [x] Add unit and API tests covering runtime, token limits, streaming, and batch flows.
- [x] Add developer docs and usage instructions.
- [x] Run verification and record results.
- [x] Add explicit multi-GPU vLLM configuration for visible CUDA devices and GPU shard count.
- [x] Add a repo-local model cache directory for Hugging Face and vLLM downloads.
- [x] Surface actionable startup errors for invalid `VLLM_GPU_COUNT` tensor-parallel settings.
- [x] Add adaptive startup GPU selection and dynamic vLLM memory utilization based on current free GPU memory.
- [x] Add a startup generation self-test prompt with tokens-per-second reporting.
- [x] Make the startup self-test non-blocking so API readiness does not wait on a long warmup generation.
- [x] Add `INFERENCE_BACKEND=ollama` support that uses a local Ollama server instead of vLLM/Hugging Face while preserving existing `/v1/*`, `/api/*`, and batch APIs.
- [x] Proxy Ollama model pull/list/generate behavior through the runtime manager with strict allowlist enforcement and one-active-model semantics.
- [x] Add Ollama backend tests plus docs and env configuration updates, then run verification.
- [x] Add Ollama-specific upstream timeout controls with one longer retry for normal non-batch requests.
- [x] Add Ollama batch timeout handling that defers timed-out items to a second retry pass with larger timeout and output-token limits.
- [x] Consolidate Ollama batch retry outcomes into the original batch artifacts, update docs/env comments, and rerun verification.

## Review

- Installed the project with `uv sync --extra dev` and verified the package wiring inside `.venv`.
- Added explicit multi-GPU configuration support via `CUDA_VISIBLE_DEVICES` and `VLLM_GPU_COUNT`.
- Added a repo-local model cache root via `MODEL_CACHE_DIR`, exported to Hugging Face cache environment variables before vLLM startup.
- Added actionable startup errors when `VLLM_GPU_COUNT` is incompatible with a model's tensor-parallel requirements.
- Added startup-only adaptive GPU selection using `nvidia-smi`, with best-GPU selection by free memory and derived `gpu_memory_utilization` when the preferred cap cannot be used.
- Added a startup self-test prompt and health-reporting for completion tokens, latency, and tokens-per-second.
- Moved the startup self-test off the blocking startup path by default, added opt-in blocking mode, and kept self-test progress visible in health data without consuming developer request slots.
- Added an Ollama runtime backend that preserves the current developer-facing API while switching upstream inference from vLLM/Hugging Face to Ollama on port `11434`.
- Added strict allowlist filtering for upstream Ollama model tags, real upstream `/api/pull` proxying through the runtime switch path, and Ollama-backed batch compatibility.
- Updated dependency metadata so `httpx` is part of the runtime install, refreshed `uv.lock`, and documented the new backend-specific config behavior.
- Refined Ollama mode to behave purely as an upstream proxy with explicit timeout retry policy for direct requests and a second-pass retry flow for timed-out batch items.
- Added dedicated Ollama timeout and retry config controls, including separate request retry and batch retry multipliers plus an internal batch retry output-token cap.
- Changed Ollama batch processing so timed-out first-pass items are retried once after the batch pass completes and only their final consolidated outcome is written to the original batch output/error artifacts.
- Updated docs and env comments to clarify that Ollama is already hosted externally and that llm-serve's Ollama responsibilities are upstream proxying, timeout handling, and batch orchestration.
- Ran `uv run pytest -q`.
- Result: `42 passed in 1.28s`.

## GitHub Repo Prep

- [x] Audit current generated artifacts and the placeholder ignore rules.
- [x] Replace the root `.gitignore` with project-specific Python, packaging, env, editor, and runtime-data rules.
- [x] Initialize a local Git repository and verify ignored paths without excluding source, docs, or lockfiles.

- Verification:
- `git init` created the local repository metadata under `.git/`.
- `git status --short --ignored` shows only expected local/generated paths ignored: `.pytest_cache/`, `.venv/`, `src/llm_serve.egg-info/`, `src/**/__pycache__/`, and `tests/__pycache__/`.
- `git check-ignore -v --stdin -n` confirmed `.venv/`, `.env`, `data/runtime/`, `*.egg-info/`, and `__pycache__/` are ignored, while `.env.example`, `uv.lock`, `pyproject.toml`, and `docs/version-control.md` are not ignored.

## Batch Progress

- [x] Add server-side batch progress reporting with live `tqdm` bars for interactive terminals.
- [x] Fall back to concise batch lifecycle logs when the server is not attached to a TTY.
- [x] Add coverage for progress updates on success/failure, distinct positions for concurrent batches, and non-TTY fallback logging.

- Verification:
- Added `tqdm` to dependency metadata and refreshed `uv.lock`.
- Ran `uv run pytest -q tests/test_batch_progress.py tests/test_app.py`.
- Result: `9 passed in 0.37s`.
