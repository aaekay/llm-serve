#!/usr/bin/env bash
# Canonical entry point: full llm-serve API (OpenAI + Ollama routes, /v1/batches, etc.)
# per api_usage.md. Configure via .env in this directory.
# Do not force the vLLM runtime extra here: ollama/mock backends do not need it,
# and forcing resolution can fail on unsupported platforms before the server starts.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Run the child as a separate bash job-control process group so Ctrl+C can
# stop the entire server tree without changing the login/session environment.
set -m

SERVER_PID=""
SERVER_PGID=""
LAUNCH_CMD=()

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  LAUNCH_CMD=("$ROOT_DIR/.venv/bin/python" "-m" "llm_serve")
else
  LAUNCH_CMD=("uv" "run" "llm-serve")
fi

forward_signal() {
  local signal_name="$1"
  if [[ -n "$SERVER_PGID" ]]; then
    kill "-$signal_name" -- "-$SERVER_PGID" 2>/dev/null || true
    return
  fi
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "-$signal_name" "$SERVER_PID" 2>/dev/null || true
  fi
}

cleanup() {
  local exit_status=$?
  local deadline=0

  trap - EXIT INT TERM HUP

  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    forward_signal TERM
    deadline=$((SECONDS + 10))
    while kill -0 "$SERVER_PID" 2>/dev/null; do
      if (( SECONDS >= deadline )); then
        forward_signal KILL
        break
      fi
      sleep 1
    done
    wait "$SERVER_PID" 2>/dev/null || true
  fi

  return "$exit_status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

"${LAUNCH_CMD[@]}" "$@" &
SERVER_PID="$!"
SERVER_PGID="$SERVER_PID"

wait "$SERVER_PID"
