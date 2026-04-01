# Lessons

- When infrastructure requirements mention GPU placement or model fit, expose explicit environment controls for both visible CUDA devices and multi-GPU shard count instead of assuming single-GPU execution.
- When a user expects local model storage, do not rely on the default Hugging Face global cache; expose and wire a repo-local cache directory explicitly.
