#!/usr/bin/env bash
# Direct vLLM debug launcher for Qwen/Qwen3.5-27B.
# This is not the canonical llm-serve API entrypoint; use ./run-server.sh for
# the full /v1/*, /api/*, /v1/files, /v1/batches, and /healthz surface.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$ROOT_DIR/data/models}"

dotenv_value() {
  python3 - "$ROOT_DIR/.env" "$1" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    raise SystemExit(0)

for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    env_key, value = line.split("=", 1)
    env_key = env_key.strip()
    if env_key.startswith("export "):
        env_key = env_key[len("export "):].strip()
    if env_key == key:
        print(value.strip().strip("'").strip('"'))
        break
PY
}

export HF_HOME="${HF_HOME:-$MODEL_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$MODEL_CACHE_DIR/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$MODEL_CACHE_DIR/transformers}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-27B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-11424}"
DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GDN_BACKEND="${GDN_BACKEND:-triton}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TEST_PROMPT="${TEST_PROMPT:-Write a thousand word poem about dawn breaking over mountains.}"
TEST_MAX_TOKENS="${TEST_MAX_TOKENS:-1400}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-900}"
ENABLE_THINKING="${ENABLE_THINKING:-$(dotenv_value ENABLE_THINKING)}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"

export MODEL_ID HOST PORT DTYPE TP_SIZE MAX_MODEL_LEN GDN_BACKEND REASONING_PARSER TEST_PROMPT TEST_MAX_TOKENS STARTUP_TIMEOUT_SECONDS ENABLE_THINKING

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

cd "$ROOT_DIR"

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

uv run vllm serve "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --language-model-only \
  --reasoning-parser "$REASONING_PARSER" \
  --gdn-prefill-backend "$GDN_BACKEND" \
  --disable-custom-all-reduce \
  "$@" &
SERVER_PID="$!"

echo "Waiting for server on http://$HOST:$PORT ..."

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
until curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server exited before becoming ready." >&2
    wait "$SERVER_PID"
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for server readiness." >&2
    exit 1
  fi
  sleep 2
done

echo
echo "Running startup test prompt..."

response_file="$(mktemp)"
elapsed_seconds="$(
  curl -sS \
    -o "$response_file" \
    -w "%{time_total}" \
    "http://$HOST:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 - <<'PY'
import json
import os

payload = {
    "model": os.environ["MODEL_ID"],
    "messages": [
        {
            "role": "user",
            "content": os.environ["TEST_PROMPT"],
        }
    ],
    "max_tokens": int(os.environ["TEST_MAX_TOKENS"]),
}
enable_thinking = os.environ.get("ENABLE_THINKING", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if not enable_thinking:
    payload["chat_template_kwargs"] = {"enable_thinking": False}
print(json.dumps(payload))
PY
)"
)"

python3 - "$response_file" "$elapsed_seconds" <<'PY'
import json
import sys

response_path, elapsed_raw = sys.argv[1], sys.argv[2]
with open(response_path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)

if "choices" not in payload:
    raise SystemExit(json.dumps(payload, indent=2))

choice = payload["choices"][0]["message"]["content"]
usage = payload.get("usage", {})
completion_tokens = usage.get("completion_tokens")
elapsed = float(elapsed_raw)
tokens_per_second = None
if completion_tokens is not None and elapsed > 0:
    tokens_per_second = completion_tokens / elapsed

print()
print("=== Startup Test Output ===")
print(choice)
print()
print("=== Startup Test Stats ===")
if completion_tokens is not None:
    print(f"Completion tokens: {completion_tokens}")
print(f"Elapsed seconds: {elapsed:.2f}")
if tokens_per_second is not None:
    print(f"Generation speed: {tokens_per_second:.2f} tokens/sec")
PY

rm -f "$response_file"

echo
echo "Server is still running on http://$HOST:$PORT"
wait "$SERVER_PID"
