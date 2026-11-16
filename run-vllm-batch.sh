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

source "$ROOT_DIR/scripts/vllm_preset.sh" "${1:-}"
shift || true

LOG="$ROOT_DIR/logs/vllm_batch_${SERVED_NAME}_gpu${GPU}_${PORT}.log"
echo "[run-vllm-batch] $MODEL_ID as '$SERVED_NAME' on GPU $GPU, http://$HOST:$PORT (snapshot: $SNAP), log: $LOG"

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
