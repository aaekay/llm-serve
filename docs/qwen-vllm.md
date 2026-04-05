# Qwen vLLM Notes

This repository treats `run-server.sh` as the canonical API entrypoint. It launches `llm-serve`, which exposes the OpenAI-compatible and Ollama-compatible HTTP surface used by clients.

Use `run-qwen35-27b.sh` only when you want a raw `vllm serve` process for debugging, benchmarking, or comparing bare engine behavior against `llm-serve`.

## Canonical Model

- The canonical Qwen model ID for the vLLM path is `Qwen/Qwen3.5-27B`.
- Keep the model ID exact and consistent across `DEFAULT_MODEL_ID`, `VLLM_DEFAULT_MODEL_ID`, and the allowlist.

## vLLM Request Behavior

- Chat requests in `vllm` mode should use the model tokenizer's chat template instead of a flattened role-prefixed prompt.
- That keeps Qwen formatting closer to the native model behavior and avoids treating chat as plain text completion.
- The same request path is used for both `/v1/chat/completions` and `/api/chat`.

## Reasoning Behavior

- In `vllm` mode, `reasoning_effort` enables thinking mode for `Qwen/Qwen3.5-27B`.
- The `low`, `medium`, and `high` labels currently share the same backend behavior. They are request intent markers, not separate tuned profiles.
- For non-stream responses, `include_reasoning=true` may surface extracted reasoning in the response body.
- Streaming remains answer-only. Reasoning content is not streamed.

## Spin-Up Responses

- When a request triggers model loading, the server returns `202` with `retry_after_seconds` in the body and a matching `Retry-After` header.
- Retry the same request after that delay.

## 2-GPU Tuning

- Set `VLLM_GPU_COUNT=2` for tensor parallelism across two GPUs.
- Leave `VLLM_GPU_AUTO_SELECT=true` if you want the server to choose the best GPUs by current free memory on startup.
- Set `CUDA_VISIBLE_DEVICES` explicitly if you want deterministic placement instead of adaptive selection.
- `VLLM_GPU_MEMORY_UTILIZATION` is a preferred cap, not a hard requirement. The server may derive a lower safe value from current free memory.
- `VLLM_ENFORCE_EAGER=true` and `VLLM_DISABLE_CUSTOM_ALL_REDUCE=true` are the conservative defaults for stable multi-GPU startup.

## Throughput Notes

- Increase `PROMPT_MAX_PARALLEL` only after measuring your actual Qwen workload.
- For large models, higher concurrency can improve aggregate throughput while reducing per-request token rate.
- The health endpoint exposes `foreground_active`, `foreground_capacity`, `batch_active`, and `batch_capacity` so you can observe whether work is truly overlapping.
