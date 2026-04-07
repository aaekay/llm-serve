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

## Startup Diagnostics

- [x] Make the `DEFAULT_MODEL_ID` validation error include the bad value and current allowlist.
- [x] Log startup self-test queue/start/pass/fail transitions so background failures are visible in server logs.
- [x] Add regression tests covering the config error message and background self-test failure logging.

- Verification:
- Ran `uv run pytest -q tests/test_config.py tests/test_runtime_manager.py tests/test_app.py`.
- Result: `18 passed in 1.12s`.
- Ran `uv run pytest -q`.
- Result: `43 passed in 1.15s`.

## Backend-Specific Default Models

- [x] Add `VLLM_DEFAULT_MODEL_ID` and `OLLAMA_DEFAULT_MODEL_ID` with `DEFAULT_MODEL_ID` kept as a compatibility fallback.
- [x] Make startup loading, omitted-model request fallback, and startup self-test use the effective backend-specific default model.
- [x] Update env/docs comments and add config/runtime/app coverage for backend-specific default selection and validation.

- Verification:
- Ran `uv run pytest -q tests/test_config.py tests/test_runtime_manager.py tests/test_ollama_app.py tests/test_app.py`.
- Result: `28 passed in 1.32s`.
- Ran `uv run pytest -q`.
- Result: `49 passed in 1.47s`.

## Launcher Compatibility

- [x] Reproduce the `run-server.sh` failure and confirm whether it comes from the wrapper or the app.
- [x] Remove the forced `uv run --extra runtime` path from the shell launcher so non-vLLM backends do not resolve incompatible vLLM dependencies at startup.
- [x] Verify the launcher still starts successfully in a backend mode that does not require the vLLM extra.
- [x] Update launcher documentation and record the verification result.

- Verification:
- Reproduced the old launcher failure with `./run-server.sh --help`, which failed during `uv` resolution on macOS arm64 because the forced `runtime` extra pulled `nvidia-cudnn-frontend==1.18.0`, a package without a compatible wheel for this platform.
- Verified the updated launcher with `PORT=11425 INFERENCE_BACKEND=mock STARTUP_LOAD_DEFAULT_MODEL=false ./run-server.sh`; the server started successfully and shut down cleanly on interrupt.
- Ran `bash -n run-server.sh`.
- Result: shell syntax check passed.

## Throughput Diagnostics

- [x] Reproduce current generation speed through both direct Ollama and the llm-serve proxy.
- [x] Check whether concurrent requests overlap in practice or serialize completely.
- [x] Add runtime health visibility for active foreground and batch slots.
- [x] Run focused tests for the new health snapshot fields and record the results.
- [x] Update docs with guidance on interpreting concurrency metrics and tuning large Ollama-backed models.

- Verification:
- Ran `uv run pytest -q tests/test_runtime_manager.py tests/test_app.py tests/test_ollama_app.py`.
- Result: `19 passed in 1.66s`.
- Benchmarked direct Ollama vs the llm-serve proxy against `gpt-oss:120b` with a 256-token `/api/generate` request.
- Direct single request: `2.706s`, `94.61` aggregate tokens/sec.
- Proxy single request: `2.712s`, `94.41` aggregate tokens/sec.
- Direct parallel 3 requests: `7.028s` wall clock, `109.28` aggregate tokens/sec.
- Proxy parallel 3 requests: `6.858s` wall clock, `111.98` aggregate tokens/sec.
- Verified the new live concurrency fields on the real server: during three overlapping requests, `/healthz` reported `foreground_active=3`, `foreground_capacity=8`, `queue_depth=0`, `batch_active=0`.

## Qwen vLLM API Alignment

- [x] Align the canonical Qwen 27B model ID across config, docs, and launchers.
- [x] Make `run-server.sh` the documented API entrypoint and keep `run-qwen35-27b.sh` as direct vLLM debugging only.
- [x] Implement model-native chat templating for vLLM-backed chat requests instead of flattening messages into a plain prompt.
- [x] Make vLLM reasoning behavior explicit and API-compatible for Qwen requests, including hidden reasoning by default and optional extraction in non-stream responses.
- [x] Add `Retry-After` headers to `202` spin-up responses and tighten schema/API validation drift.
- [x] Update docs and `.env.example` for the 2-GPU Qwen vLLM path, concurrency tuning, and backend-specific reasoning behavior.
- [x] Run focused tests plus full verification and record the results.

- Verification:
- Switched the vLLM chat path to tokenizer-backed chat templating and reasoning extraction, with streamed output stripped down to answer content only.
- Propagated chat messages through batch inference requests so the vLLM backend can apply the same chat-template path outside direct API calls.
- Added `Retry-After` headers to `202` spin-up responses and tightened request validation so `top_p=0` and `tool` messages are rejected consistently with the docs.
- Aligned the canonical Qwen model ID to `Qwen/Qwen3.5-27B` across `.env.example`, the direct Qwen launcher, docs, and examples.
- Changed the vLLM text-only default to `VLLM_LANGUAGE_MODEL_ONLY=true` and documented the 2-GPU Qwen baseline config, including `VLLM_MAX_MODEL_LEN=8192` in `.env.example`.
- Updated the user-facing docs to make `run-server.sh` the canonical API entrypoint and `run-qwen35-27b.sh` the direct vLLM debug launcher.
- Added a dedicated [`docs/qwen-vllm.md`](../docs/qwen-vllm.md) reference for 2-GPU Qwen startup and tuning notes.
- Ran `uv run pytest -q tests/test_vllm_backend.py tests/test_app.py tests/test_ollama_app.py tests/test_runtime_manager.py tests/test_config.py`.
- Result: `44 passed in 1.67s`.
- Ran `uv run pytest -q`.
- Result: `58 passed in 1.67s`.
- Ran `bash -n run-server.sh` and `bash -n run-qwen35-27b.sh`.
- Result: both shell syntax checks passed.
- Ran `git diff --check`.
- Result: no diff formatting issues.
- Verified the launcher smoke test with `PORT=18425 INFERENCE_BACKEND=mock STARTUP_LOAD_DEFAULT_MODEL=false ./run-server.sh`.
- Result: the server started successfully, reached application startup complete, and shut down cleanly on interrupt.

## Ctrl+C GPU Cleanup

- [x] Reproduce and explain why `run-server.sh` can leave a GPU process behind after Ctrl+C.
- [x] Fix runtime shutdown so an interrupted vLLM startup cleans up partially started worker descendants.
- [x] Harden `run-server.sh` so the launcher supervises the server process group and forwards shutdown signals.
- [x] Add regression coverage plus docs updates, then run verification and record the results.

- Review:
- `run-server.sh` previously `exec`'d straight into the server, so there was no supervising parent shell left once the process tree was running. At the same time, the vLLM backend only tracked direct child PIDs after a successful startup snapshot, which left a hole for interrupted startup and deeper worker descendants.
- Updated the runtime manager so cancelling a model switch now shuts down the partially started backend instead of dropping it during shutdown.
- Updated the vLLM backend to snapshot descendant PIDs, keep tracked root PIDs for late-spawned descendants, and reuse that data in both normal shutdown and `atexit` cleanup.
- Updated `run-server.sh` to keep a small shell supervisor alive, prefer the repo-local `.venv` Python when available, launch the server in its own bash job-control process group, and escalate from `TERM` to `KILL` if shutdown stalls.
- Updated `README.md` and `docs/codeflow.md` to document the new Ctrl+C cleanup behavior and the launcher supervision model.
- Ran `uv run pytest -q tests/test_runtime_manager.py tests/test_vllm_backend.py`.
- Result: `26 passed in 2.52s`.
- Ran `uv run pytest -q`.
- Result: `72 passed in 2.93s`.
- Ran `bash -n run-server.sh`.
- Result: shell syntax check passed.
- Ran `git diff --check`.
- Result: no diff formatting issues.
- Attempted a local `run-server.sh` smoke test with `INFERENCE_BACKEND=mock`, but this sandbox blocked binding `127.0.0.1:18525`, so full end-to-end interrupt verification could not be completed here.
