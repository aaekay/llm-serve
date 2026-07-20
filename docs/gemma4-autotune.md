# Gemma 4 12B throughput autotuning

Use the vLLM checkpoint `google/gemma-4-12B-it`; the Ollama-style tag `gemma4:12b` does not give this service control over GPU topology.

Gemma 4 12B Unified uses the project's supported vLLM 0.23.x runtime line for the native optimized implementation. If logs resolve `TransformersMultiModalForCausalLM` and report a Transformers fallback, upgrade the runtime before tuning:

```bash
uv sync --extra runtime --upgrade-package vllm
```

Set `HF_TOKEN` before the first run to avoid unauthenticated Hub rate limits while the model files are downloaded.

The native Gemma 4 path may JIT-compile CUDA components on startup. The host therefore needs the CUDA toolkit—not only an NVIDIA driver—and `CUDA_HOME/bin/nvcc` must exist. Check the CUDA version expected by the installed PyTorch build and verify the compiler before tuning:

```bash
python -c 'import torch; print(torch.version.cuda)'
command -v nvcc
echo "$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version
```

Load or install the matching toolkit, then export `CUDA_HOME`, prepend its `bin` directory to `PATH`, and add its `lib64` directory to `LD_LIBRARY_PATH`. On managed workstations this is often provided through a CUDA environment module; otherwise use the NVIDIA CUDA repository packages for the host distribution.

For four 24 GB NVIDIA GPUs and text-only BF16 serving, start with:

```dotenv
INFERENCE_BACKEND=vllm
DEFAULT_MODEL_ID=google/gemma-4-12B-it
VLLM_DEFAULT_MODEL_ID=google/gemma-4-12B-it
MODEL_ALLOWLIST=google/gemma-4-12B-it
REASONING_MODEL_ALLOWLIST=
VLLM_GPU_MODE=auto
VLLM_GPU_COUNT=4
VLLM_GPU_AUTO_SELECT=true
VLLM_DTYPE=bfloat16
VLLM_LANGUAGE_MODEL_ONLY=true
VLLM_MAX_MODEL_LEN=8192
VLLM_MAX_NUM_BATCHED_TOKENS=8192
VLLM_AUTOTUNE_ON_MISS=true
VLLM_AUTOTUNE_SLO_MULTIPLIER=2.0
VLLM_AUTOTUNE_MAX_CONCURRENCY=64
```

Run a planned tuning pass with:

```bash
uv run --extra runtime llm-serve tune
```

Use `--force` to ignore an existing matching profile. Normal server startup also tunes when the profile is missing or stale and `VLLM_AUTOTUNE_ON_MISS=true`; readiness remains blocked while candidates are loading and benchmarking.

Auto mode inspects the selected GPUs and their interconnects, rejects incompatible or memory-infeasible BF16 tensor-parallel widths, and benchmarks each remaining topology. On four 24 GB cards this normally compares two TP=2 replicas with one TP=4 replica. The result with the highest aggregate token throughput is selected only if every workload bucket remains within the configured p95 latency multiplier.

For a TP=2/two-replica candidate, replicas start sequentially to avoid simultaneous model-download and compilation spikes. Seeing only two GPUs active during the first replica's startup is expected; all four become active before that candidate is benchmarked.

The default workload is:

- 50% short: 256 input and 128 output tokens
- 30% medium: 1,024 input and 256 output tokens
- 20% long: 4,096 input and 512 output tokens

Override it with `VLLM_AUTOTUNE_WORKLOAD_FILE`. The file may contain a list directly or a `buckets` object:

```json
{
  "buckets": [
    {"name": "chat", "input_tokens": 512, "output_tokens": 192, "weight": 0.8},
    {"name": "analysis", "input_tokens": 4096, "output_tokens": 768, "weight": 0.2}
  ]
}
```

Profiles live under `data/runtime/tuning/` by default. They are invalidated when the model revision, vLLM version, GPU identity/topology, effective memory allocation, benchmark workload, latency target, context length, or scheduler limits change.

`GET /healthz` reports the selected tensor-parallel size, replica count, tuned foreground concurrency, measured throughput, latency ratio, and unhealthy worker count. Batch requests use otherwise-idle capacity and stop being admitted whenever foreground work is queued.

## Force a topology without tuning

If one complete model plus its KV cache fits comfortably on each GPU, force four independent one-GPU replicas with:

```dotenv
VLLM_GPU_MODE=fixed
VLLM_GPU_COUNT=4
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_REPLICA_COUNT=4
PROMPT_MAX_PARALLEL=32
```

Fixed mode skips topology and concurrency benchmarking. It selects four GPUs, starts one vLLM worker on each GPU, and load-balances requests across them. `VLLM_TENSOR_PARALLEL_SIZE * VLLM_REPLICA_COUNT` must equal `VLLM_GPU_COUNT`, so an invalid layout fails during configuration instead of silently leaving GPUs unused. `PROMPT_MAX_PARALLEL` remains the gateway concurrency limit because fixed mode has no measured concurrency profile.

Other four-GPU layouts are `VLLM_TENSOR_PARALLEL_SIZE=2` with `VLLM_REPLICA_COUNT=2`, or `VLLM_TENSOR_PARALLEL_SIZE=4` with `VLLM_REPLICA_COUNT=1`.
