#!/usr/bin/env bash
# Batch job server on 127.0.0.1 (default port 8000): clients upload OpenAI batch JSONL over HTTP, the server runs the
# jobs one at a time on one GPU with run-vllm-batch-file.sh (vllm run-batch), and clients download the results later.
# The GPU is used only while a job runs. See scripts/batch_job_server.py for the endpoints.
#
#   GPU=2 ./run-batch-jobs.sh            # keep it running in tmux; reach it through an SSH tunnel
#   GPU=2 PORT=8010 ./run-batch-jobs.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs batch_io/jobs
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then PY="$ROOT_DIR/.venv/bin/python"; else PY="python3"; fi
export GPU="${GPU:-0}" PORT="${PORT:-8000}" HOST=127.0.0.1
exec "$PY" scripts/batch_job_server.py 2>&1 | tee -a "logs/batch_jobs_gpu${GPU}_${PORT}.log"
