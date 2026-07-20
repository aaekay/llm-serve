# Gemma 4 12B throughput autotuning

Use the vLLM checkpoint `google/gemma-4-12B-it`; the Ollama-style tag `gemma4:12b` does not give this service control over GPU topology.

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
