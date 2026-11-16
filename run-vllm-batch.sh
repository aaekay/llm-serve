#!/usr/bin/env bash
# Raw `vllm serve` tuned for offline batch throughput (many concurrent requests, short JSON answers), one GPU.
#
# Use this instead of run-server.sh when a client sends thousands of independent requests and balances load itself:
# the llm-serve gateway is built for interactive use (PROMPT_MAX_PARALLEL=1, VLLM_MAX_NUM_SEQS=1, eager mode, model
# allowlist, no /metrics), while this exposes plain vLLM: OpenAI API at /v1/* plus Prometheus /metrics.
#
#   ./run-vllm-batch.sh gemma4-31b              # google/gemma-4-31B-it-qat-w4a16-ct
#   ./run-vllm-batch.sh qwen36-35b-a3b-fp8      # Qwen/Qwen3.6-35B-A3B-FP8
#   ./run-vllm-batch.sh qwen36-27b-fp8          # Qwen/Qwen3.6-27B-FP8
#   ./run-vllm-batch.sh qwen36-35b-a3b          # Qwen/Qwen3.6-35B-A3B (BF16)
#   GPU=1 PORT=8001 ./run-vllm-batch.sh gemma4-31b
#
# Binds to 127.0.0.1 only; reach it through an SSH tunnel. Runs offline (HF_HUB_OFFLINE=1): the model must already
# be in the local Hugging Face cache (models/hub or models/). The served model name equals the Hugging Face id
# without the organisation, e.g. "gemma-4-31B-it-qat-w4a16-ct"; clients must send exactly that as "model".
# Thinking/reasoning parsers are deliberately not enabled: clients send chat_template_kwargs.enable_thinking=false
# and expect the answer in message.content.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PRESET="${1:-}"
case "$PRESET" in
  gemma4-31b)          MODEL_ID="google/gemma-4-31B-it-qat-w4a16-ct" ;;
  qwen36-35b-a3b-fp8)  MODEL_ID="Qwen/Qwen3.6-35B-A3B-FP8" ;;
  qwen36-35b-a3b)      MODEL_ID="Qwen/Qwen3.6-35B-A3B" ;;
  qwen36-27b-fp8)      MODEL_ID="Qwen/Qwen3.6-27B-FP8" ;;
  "")                  echo "usage: $0 <gemma4-31b|qwen36-35b-a3b-fp8|qwen36-35b-a3b|qwen36-27b-fp8>  (or MODEL_ID=org/name $0 custom)" >&2; exit 2 ;;
  custom)              MODEL_ID="${MODEL_ID:?set MODEL_ID=org/name}" ;;
  *)                   echo "unknown preset: $PRESET" >&2; exit 2 ;;
esac
shift || true

GPU="${GPU:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"       # our prompts are <=12k tokens + <=1.2k output
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
SEED="${SEED:-42}"
SERVED_NAME="${SERVED_NAME:-${MODEL_ID#*/}}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$ROOT_DIR/models}"

# Locate the Hugging Face hub cache that holds the model (models/hub/models--org--name or models/models--org--name).
REPO_DIR="models--${MODEL_ID/\//--}"
if [[ -d "$MODEL_CACHE_DIR/hub/$REPO_DIR" ]]; then
  HUB="$MODEL_CACHE_DIR/hub"
elif [[ -d "$MODEL_CACHE_DIR/$REPO_DIR" ]]; then
  HUB="$MODEL_CACHE_DIR"
else
  echo "Model $MODEL_ID not found under $MODEL_CACHE_DIR/hub or $MODEL_CACHE_DIR (looked for $REPO_DIR)." >&2
  echo "Set MODEL_CACHE_DIR to the folder that contains (hub/)$REPO_DIR." >&2
  exit 1
fi
export HF_HOME="$MODEL_CACHE_DIR" HUGGINGFACE_HUB_CACHE="$HUB" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Serve the cached snapshot folder directly. Passing the repo id makes huggingface_hub verify the snapshot against the
# full repo file list and refuse to start offline when non-model files (.gitattributes, README.md) were never downloaded.
REV="${REVISION:-main}"
if [[ -f "$HUB/$REPO_DIR/refs/$REV" ]]; then
  COMMIT="$(tr -d '[:space:]' < "$HUB/$REPO_DIR/refs/$REV")"
else
  COMMIT="$(ls -1t "$HUB/$REPO_DIR/snapshots" 2>/dev/null | head -1)"
fi
SNAP="$HUB/$REPO_DIR/snapshots/$COMMIT"
if [[ -z "$COMMIT" || ! -f "$SNAP/config.json" ]]; then
  echo "No usable snapshot for $MODEL_ID under $HUB/$REPO_DIR/snapshots (need config.json)." >&2
  exit 1
fi
if ! ls "$SNAP"/*.safetensors >/dev/null 2>&1; then
  echo "Snapshot $SNAP has no *.safetensors weights; the download is incomplete." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ "$MODEL_ID" == google/gemma-4* ]] && ! command -v nvcc >/dev/null 2>&1 && [[ ! -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ]]; then
  echo "[warn] Gemma 4 may JIT-compile CUDA kernels and needs the CUDA toolkit (nvcc). See docs/gemma4-autotune.md / nvcc.sh." >&2
fi

mkdir -p "$ROOT_DIR/logs"
LOG="$ROOT_DIR/logs/vllm_batch_${SERVED_NAME}_gpu${GPU}_${PORT}.log"
echo "[run-vllm-batch] $MODEL_ID as '$SERVED_NAME' on GPU $GPU, http://$HOST:$PORT (snapshot: $SNAP), log: $LOG"

# Put the venv's bin first on PATH: FlashInfer and torch JIT-compile kernels with `ninja` (a runtime dependency that
# lives in .venv/bin); calling .venv/bin/vllm directly does not make it visible to their subprocesses.
if [[ -x "$ROOT_DIR/.venv/bin/vllm" ]]; then
  export PATH="$ROOT_DIR/.venv/bin:$PATH"; VLLM=("$ROOT_DIR/.venv/bin/vllm")
else
  VLLM=(uv run --extra runtime vllm)
fi
if ! command -v ninja >/dev/null 2>&1 && [[ "${VLLM[0]}" != uv ]]; then
  echo "ninja not found in $ROOT_DIR/.venv/bin; run: uv sync --extra runtime" >&2; exit 1
fi

"${VLLM[@]}" serve "$SNAP" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-prefix-caching \
  --language-model-only \
  --seed "$SEED" \
  "$@" 2>&1 | tee -a "$LOG"
