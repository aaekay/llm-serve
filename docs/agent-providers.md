# Coding Agent Providers

`llm-serve` can act as a local OpenAI-compatible provider for Hermes, OpenCode,
and Pi. The production agent path uses vLLM, `Qwen/Qwen3.5-27B`, a 65,536-token
context window, an 8,192-token output cap, streaming tool calls, and the Responses
API.

## Start and verify

```bash
cp .env.example .env
uv sync --extra runtime --extra dev
./run-server.sh
```

In another terminal:

```bash
curl -s http://127.0.0.1:11424/healthz
uv run python scripts/agent_provider_smoke.py
```

The first startup downloads all Hugging Face and vLLM files beneath `models/`.
The 64K preset intentionally fails during startup if the selected GPUs cannot
provide the requested context instead of advertising a false capability.

## Hermes

Run `hermes model`, select a custom/self-hosted endpoint, and enter:

- Base URL: `http://127.0.0.1:11424/v1`
- API mode: `chat_completions`
- Model: `Qwen/Qwen3.5-27B`
- API key: any placeholder value

Equivalent configuration:

```yaml
model:
  default: Qwen/Qwen3.5-27B
  provider: custom
  base_url: http://127.0.0.1:11424/v1
  api_key: local
```

## OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "llm-serve/Qwen/Qwen3.5-27B",
  "provider": {
    "llm-serve": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local llm-serve",
      "options": {
        "baseURL": "http://127.0.0.1:11424/v1",
        "apiKey": "local"
      },
      "models": {
        "Qwen/Qwen3.5-27B": {
          "name": "Qwen3.5 27B Local",
          "limit": {"context": 65536, "output": 8192}
        }
      }
    }
  }
}
```

## Pi

Add this provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "llm-serve": {
      "baseUrl": "http://127.0.0.1:11424/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsDeveloperRole": true,
        "supportsUsageInStreaming": true,
        "supportsStrictMode": true,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "Qwen/Qwen3.5-27B",
          "name": "Qwen3.5 27B Local",
          "reasoning": true,
          "contextWindow": 65536,
          "maxTokens": 8192,
          "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
        }
      ]
    }
  }
}
```

## Responses state

The vLLM backend supports `POST /v1/responses`, retrieval, cancellation, SSE,
and `previous_response_id`. Response state belongs to the worker that generated
it. `llm-serve` preserves replica affinity, but the state is lost when that
worker restarts or when another model is loaded. Clients should treat a 404 as
a signal to resend their complete history.

## Model switching

Requests for another allowlisted model wait for that model to load; they no
longer receive an agent-incompatible HTTP 202 response. Model switching drains
active streams and background Responses work first. A model can accept tools
only when it has an entry in `VLLM_TOOL_CALL_PARSERS`.
