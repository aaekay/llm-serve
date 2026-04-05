#!/usr/bin/env bash
# Canonical entry point: full llm-serve API (OpenAI + Ollama routes, /v1/batches, etc.)
# per api_usage.md. Configure via .env in this directory.
# Do not force the vLLM runtime extra here: ollama/mock backends do not need it,
# and forcing resolution can fail on unsupported platforms before the server starts.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

exec uv run llm-serve "$@"
