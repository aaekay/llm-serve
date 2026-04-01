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

## Review

- Installed the project with `uv sync --extra dev` and verified the package wiring inside `.venv`.
- Added explicit multi-GPU configuration support via `CUDA_VISIBLE_DEVICES` and `VLLM_GPU_COUNT`.
- Added a repo-local model cache root via `MODEL_CACHE_DIR`, exported to Hugging Face cache environment variables before vLLM startup.
- Ran `uv run pytest -q`.
- Result: `12 passed in 0.54s`.

## GitHub Repo Prep

- [x] Audit current generated artifacts and the placeholder ignore rules.
- [x] Replace the root `.gitignore` with project-specific Python, packaging, env, editor, and runtime-data rules.
- [x] Initialize a local Git repository and verify ignored paths without excluding source, docs, or lockfiles.

- Verification:
- `git init` created the local repository metadata under `.git/`.
- `git status --short --ignored` shows only expected local/generated paths ignored: `.pytest_cache/`, `.venv/`, `src/llm_serve.egg-info/`, `src/**/__pycache__/`, and `tests/__pycache__/`.
- `git check-ignore -v --stdin -n` confirmed `.venv/`, `.env`, `data/runtime/`, `*.egg-info/`, and `__pycache__/` are ignored, while `.env.example`, `uv.lock`, `pyproject.toml`, and `docs/version-control.md` are not ignored.
