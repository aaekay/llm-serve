#!/usr/bin/env bash
# Canonical entry point: full llm-serve API (OpenAI + Ollama routes, /v1/batches, etc.)
# per api_usage.md. Configure via .env in this directory; requires vLLM optional deps.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

exec uv run --extra runtime llm-serve "$@"
