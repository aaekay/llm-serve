#!/usr/bin/env bash
# Offline batch inference with `vllm run-batch`: no HTTP server, no tunnel. One GPU.
#
#   ./run-vllm-batch-file.sh gemma4-31b                  # every batch_io/in/*.jsonl -> batch_io/out/<name>.out.jsonl
#   ./run-vllm-batch-file.sh gemma4-31b some_job_        # only input files whose name starts with some_job_
#   GPU=2 ./run-vllm-batch-file.sh qwen36-35b-a3b-fp8
#
# Input: OpenAI batch format, one request per line:
#   {"custom_id": "...", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "<served name>", ...}}
# "model" in each body must equal the served name (the Hugging Face id without the organisation).
# Each input file is run in turn; a file whose .out.jsonl already exists and has as many lines as the input is skipped,
# so re-running after a stop resumes at the next unfinished file (keep files to ~2,000 requests). A partial output is
# moved aside to .partial and the file is re-run whole.
#
# batch_io/ is git-ignored. Input files may contain PHI: copy them with scp/rsync only (never git), and delete both
# input and output here once the results have been copied back.
#
# Engine settings match run-vllm-batch.sh (16k context, 64 sequences, prefix caching, seed 42, offline model cache).
# Run it inside tmux or with nohup; progress goes to logs/vllm_runbatch_<served name>_gpu<N>.log.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/vllm_preset.sh" "${1:-}"
PREFIX="${2:-}"

IN_DIR="${BATCH_IN_DIR:-$ROOT_DIR/batch_io/in}"
OUT_DIR="${BATCH_OUT_DIR:-$ROOT_DIR/batch_io/out}"
mkdir -p "$IN_DIR" "$OUT_DIR"
LOG="$ROOT_DIR/logs/vllm_runbatch_${SERVED_NAME}_gpu${GPU}.log"

shopt -s nullglob
FILES=("$IN_DIR"/"$PREFIX"*.jsonl)
if (( ${#FILES[@]} == 0 )); then
  echo "No input files matching $IN_DIR/${PREFIX}*.jsonl" >&2; exit 1
fi
echo "[run-vllm-batch-file] $MODEL_ID as '$SERVED_NAME' on GPU $GPU; ${#FILES[@]} input file(s); log: $LOG" | tee -a "$LOG"

for IN in "${FILES[@]}"; do
  NAME="$(basename "$IN" .jsonl)"
  OUT="$OUT_DIR/$NAME.out.jsonl"
  N_IN=$(grep -c . "$IN" || true)
  if [[ -f "$OUT" ]]; then
    N_OUT=$(grep -c . "$OUT" || true)
    if (( N_OUT >= N_IN )); then
      echo "[skip] $NAME: already done ($N_OUT/$N_IN)" | tee -a "$LOG"; continue
    fi
    mv "$OUT" "$OUT.partial.$(date +%s)"
  fi
  echo "[run] $NAME: $N_IN requests  $(date)" | tee -a "$LOG"
  T0=$SECONDS
  "${VLLM[@]}" run-batch -i "$IN" -o "$OUT.tmp" \
    --model "$SNAP" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-prefix-caching \
    --language-model-only \
    --seed "$SEED" >> "$LOG" 2>&1
  mv "$OUT.tmp" "$OUT"
  echo "[done] $NAME: $(grep -c . "$OUT") results in $((SECONDS - T0)) s" | tee -a "$LOG"
done
echo "[run-vllm-batch-file] all files done $(date)" | tee -a "$LOG"
