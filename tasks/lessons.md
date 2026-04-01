# Lessons

- When infrastructure requirements mention GPU placement or model fit, expose explicit environment controls for both visible CUDA devices and multi-GPU shard count instead of assuming single-GPU execution.
- When a user expects local model storage, do not rely on the default Hugging Face global cache; expose and wire a repo-local cache directory explicitly.
- When an env var maps directly to a backend-specific constraint like vLLM tensor parallelism, document that exact mapping and raise an actionable startup error instead of exposing the raw backend assertion.
- When GPU memory availability is dynamic because other applications may already occupy GPUs, do not treat memory-utilization config as a fixed constant; inspect current free memory and derive a safe startup value.
- When the user wants proof that startup generation actually works, add an explicit startup self-test and publish its result as structured health data instead of relying only on model-load success.
- When a startup self-test uses a real long-form generation, do not block API readiness on it by default; run it in the background and expose progress in health data, with blocking mode only as an opt-in.
