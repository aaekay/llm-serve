# API Usage Guide

Base URL:

- `http://127.0.0.1:11424`

This server supports both API styles:

- OpenAI-compatible (`/v1/*`)
- Ollama-compatible (`/api/*`)

## 1) Start The Server

```bash
llm-serve
```

Backend notes:

- `INFERENCE_BACKEND=vllm`: serves Hugging Face models directly through local vLLM.
- `INFERENCE_BACKEND=ollama`: treats Ollama as an already-running upstream service at `OLLAMA_BASE_URL` while keeping the same `llm-serve` API surface, batching, and allowlist behavior.
- In Ollama mode, direct non-batch requests can get one longer internal retry on upstream timeout before `llm-serve` returns an error.

## 2) OpenAI-Compatible Usage (`/v1/*`)

### Endpoints

- `POST /v1/chat/completions`
- `GET /v1/models`
- `POST /v1/files`
- `GET /v1/files/{file_id}`
- `GET /v1/files/{file_id}/content`
- `POST /v1/batches`
- `GET /v1/batches/{batch_id}`
- `POST /v1/batches/{batch_id}/cancel`

### `POST /v1/chat/completions` Request Fields

- `messages` (required): array of `{ "role": "...", "content": "..." }`
- `role` values: `system`, `user`, `assistant`
- `model` (optional): requested model ID
- `stream` (optional, default `false`): `true` for SSE output
- `temperature` (optional): `0.0` to `2.0`
- `top_p` (optional): `>0.0` and `<=1.0`
- `max_tokens` (optional): `1` to `8192`
- `max_completion_tokens` (optional): `1` to `8192` (takes precedence over `max_tokens`)
- `reasoning_effort` (optional): `low`, `medium`, or `high`
- `include_reasoning` (optional, default `false`): include extracted reasoning text in non-stream response

Allowlist policy:

- `reasoning_effort` set -> model must be in `REASONING_MODEL_ALLOWLIST`
- `reasoning_effort` not set -> model must be in `MODEL_ALLOWLIST`

### Reasoning Requests (OpenAI)

- Use `reasoning_effort` to request deeper reasoning:
  - `low`: simple extraction/classification/short transformations
  - `medium`: multi-step explanation and synthesis
  - `high`: difficult edge-case reasoning (higher latency/token usage)
- `include_reasoning` defaults to `false`.
- When `include_reasoning=false`, consume `choices[0].message.content` as the final answer.
- When `include_reasoning=true` (non-stream), response may include `choices[0].message.reasoning`.

Non-stream (reasoning enabled, reasoning hidden by default):

```bash
curl -s http://127.0.0.1:11424/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Compare two treatment options in 3 bullets."}],
    "reasoning_effort": "medium",
    "stream": false
  }'
```

Non-stream (reasoning enabled and included in output):

```bash
curl -s http://127.0.0.1:11424/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Solve this step by step and then give final answer."}],
    "reasoning_effort": "high",
    "include_reasoning": true,
    "stream": false
  }'
```

Read output as:

- `choices[0].message.content`: final answer (stable)
- `choices[0].message.reasoning`: optional reasoning trace (debug-style field)

Streaming (reasoning enabled):

```bash
curl -N http://127.0.0.1:11424/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Think carefully and answer in JSON."}],
    "reasoning_effort": "medium",
    "stream": true
  }'
```

Streaming consumer note:

- Parse `delta.content` chunks and concatenate for final answer.
- End of stream is `data: [DONE]`.
- With `include_reasoning=false`, stream is expected to contain final-answer content only.

### OpenAI Non-Streaming (`curl`)

```bash
curl -s http://127.0.0.1:11424/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Explain cataract surgery in one sentence."}],
    "stream": false
  }'
```

### OpenAI Streaming (`curl`)

```bash
curl -N http://127.0.0.1:11424/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Count from one to five."}],
    "stream": true
  }'
```

Streaming format is Server-Sent Events (`text/event-stream`) and ends with:

```text
data: [DONE]
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11424/v1",
    api_key="not-required",
)

resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Say hello"}],
    stream=False,
)

print(resp.choices[0].message.content)
```

### OpenAI Python SDK (Reasoning Example)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11424/v1",
    api_key="not-required",
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Analyze this and return 3 concise recommendations."}],
    stream=False,
    extra_body={
        "reasoning_effort": "medium",
        "include_reasoning": True,
    },
)

answer = resp.choices[0].message.content
reasoning = getattr(resp.choices[0].message, "reasoning", None)
print("answer:", answer)
print("reasoning:", reasoning)
```

### OpenAI Model Spin-Up (`202`)

With model switching enabled, first request for an unloaded model can return `202`:

```json
{
  "object": "model.load",
  "status": "spinning_up",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "current_model": "Qwen/Qwen2.5-7B-Instruct",
  "retry_after_seconds": 2,
  "eta_seconds": null
}
```

When this happens:

1. Wait for `retry_after_seconds` (or `Retry-After` header).
2. Retry the same request.

### OpenAI Batch Workflow (`/v1/files`, `/v1/batches`)

Upload batch input JSONL:

```bash
cat > /tmp/batch-input.jsonl <<'JSONL'
{"custom_id":"req-1","method":"POST","url":"/v1/chat/completions","body":{"messages":[{"role":"user","content":"Say hello"}],"stream":false}}
{"custom_id":"req-2","method":"POST","url":"/v1/chat/completions","body":{"messages":[{"role":"user","content":"Write one sentence about cataract surgery."}],"stream":false}}
JSONL

curl -s http://127.0.0.1:11424/v1/files \
  -F purpose=batch \
  -F file=@/tmp/batch-input.jsonl
```

Create batch:

```bash
curl -s http://127.0.0.1:11424/v1/batches \
  -H 'Content-Type: application/json' \
  -d '{
    "input_file_id": "file-abc123",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'
```

Batch timeout note for Ollama mode:

- If an item times out talking to Ollama during the first pass, `llm-serve` keeps the batch running.
- After the first pass finishes, timed-out items are retried once with a longer upstream timeout and a larger internal output-token limit.
- The final success or failure is written into the original batch output/error artifacts.

Poll status:

```bash
curl -s http://127.0.0.1:11424/v1/batches/batch-abc123
```

Download outputs:

```bash
curl -s http://127.0.0.1:11424/v1/files/file-output123/content
curl -s http://127.0.0.1:11424/v1/files/file-error123/content
```

Cancel batch:

```bash
curl -s -X POST http://127.0.0.1:11424/v1/batches/batch-abc123/cancel
```

Input JSONL rules:

- Each line must be an object with `custom_id`, `method`, `url`, and `body`.
- `method` must be `POST`.
- `url` must be `/v1/chat/completions`.
- `body.stream=true` is not supported in batch requests.
- Batch items are processed best-effort; one item failure does not fail the whole batch.

## 3) Ollama-Compatible Usage (`/api/*`)

### Endpoints

- `POST /api/chat`
- `POST /api/generate`
- `GET /api/tags`
- `POST /api/pull`

### `POST /api/chat` (Ollama-style)

Non-stream:

```bash
curl -s http://127.0.0.1:11424/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role":"user","content":"Say hello"}],
    "stream": false
  }'
```

Stream:

```bash
curl -N http://127.0.0.1:11424/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role":"user","content":"Count to five"}],
    "stream": true
  }'
```

Streaming response is NDJSON (`application/x-ndjson`) with one JSON object per line.

### `POST /api/generate`

```bash
curl -s http://127.0.0.1:11424/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Write one sentence about cataract surgery.",
    "stream": false,
    "options": {
      "num_predict": 64,
      "temperature": 0.2,
      "top_p": 0.9
    }
  }'
```

Supported option mappings:

- `options.num_predict` -> completion token limit
- `options.temperature` -> temperature
- `options.top_p` -> top-p
- `options.reasoning_effort` -> reasoning effort (`low|medium|high`)
- `options.include_reasoning` -> include extracted reasoning (non-stream response path)

Reasoning example (`/api/generate`):

```bash
curl -s http://127.0.0.1:11424/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "prompt": "Reason about this scenario and return the final recommendation.",
    "stream": false,
    "options": {
      "reasoning_effort": "medium",
      "include_reasoning": true,
      "num_predict": 256
    }
  }'
```

Ollama output usage:

- Use `response` as the final answer for downstream logic.
- Treat any reasoning-style metadata as optional and non-contractual.

### `GET /api/tags`

```bash
curl -s http://127.0.0.1:11424/api/tags
```

Notes:

- `ollama` backend: returns allowlisted models that are already installed in the upstream Ollama server, with `details.loaded` showing the model currently active in `llm-serve`.
- other backends: returns allowlisted models based on local runtime state.

### `POST /api/pull`

Triggers async model activation for allowlisted models.

- In `ollama` mode, this is only an optional convenience proxy to Ollama. Use it when the upstream Ollama process still needs to pull the model.
- In other backends, this is a warmup/model-switch operation only.

```bash
curl -s http://127.0.0.1:11424/api/pull \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "meta-llama/Llama-3.1-8B-Instruct"
  }'
```

Typical responses:

- `200`: model already ready
- `202`: warmup/switch triggered or in progress
- `400`: model is not available/allowed

## 4) Best Practices for Reasoning Models

Practical checklist:

- Choose `reasoning_effort` based on task complexity:
  - `low` for simple tasks
  - `medium` for multi-step but routine tasks
  - `high` for difficult tasks where quality matters more than speed
- Set explicit output format constraints in your prompt (for example: bullets, JSON keys, max length).
- Prefer `max_completion_tokens` on reasoning workloads to cap cost and latency.
- Keep `include_reasoning=false` in production by default.
- If `include_reasoning=true`, store reasoning separately from end-user answer fields.
- Track latency by effort tier (`low/medium/high`) and tune per endpoint.
- For single-model backends (`vllm`, `tensorrt_llm`), ensure the served `MODEL_ID` is in the allowlist that matches your request mode.

How to consume reasoning output safely:

- Stable field for app logic:
  - OpenAI: `choices[0].message.content`
  - Ollama: `response`
- Optional field:
  - OpenAI non-stream: `choices[0].message.reasoning` when requested
- Do not hard-code business logic against free-form reasoning text.
- Validate/parsing should target the final answer field, not reasoning trace text.

## 5) Shared Utility Endpoints

### Health

```bash
curl -s http://127.0.0.1:11424/healthz
```

Useful fields:

- `status`
- `loaded`
- `model_id`
- `inference_backend`
- `queue_depth`
- `batch_jobs_total`
- `batch_jobs_in_progress`
- `batch_queue_depth`
- `switch_in_progress`
- `switch_target_model`

## 6) Common Error

- `400`: invalid request, unknown model, prompt too long, or model mismatch
- `409`: runtime not accepting requests or model switch conflict
- `429`: queue full
- `503`: service/runtime not ready
- `504`: inference timeout
- `500`/`502`: internal/upstream runtime failures
