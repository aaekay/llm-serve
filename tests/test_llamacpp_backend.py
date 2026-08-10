from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from llm_serve.errors import NotReadyError
from llm_serve.runtime.llamacpp_backend import LlamaCppAPIClient, LlamaCppModelBackend
from llm_serve.types import InferenceRequest

from .conftest import make_settings


MODEL_ID = "meta-models/Muse-Glimmer-30B-GGUF"


def _settings(tmp_path, **overrides):
    return make_settings(
        tmp_path,
        INFERENCE_BACKEND="llamacpp",
        DEFAULT_MODEL_ID=MODEL_ID,
        LLAMACPP_DEFAULT_MODEL_ID=MODEL_ID,
        MODEL_ALLOWLIST=MODEL_ID,
        REASONING_MODEL_ALLOWLIST=MODEL_ID,
        STARTUP_SELF_TEST_ENABLED="false",
        **overrides,
    )


def _request(*, stream=False):
    return InferenceRequest(
        model_id=MODEL_ID,
        prompt="hello",
        messages=[{"role": "user", "content": "hello"}],
        max_output_tokens=128,
        temperature=1.0,
        top_p=0.95,
        stream=stream,
        reasoning_effort="high",
    )


def test_llamacpp_backend_checks_alias_and_generates_chat(tmp_path):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL_ID}]})
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["chat_template_kwargs"] == {"reasoning_strength": "high"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ready",
                            "reasoning_content": "checked",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    settings = _settings(tmp_path, LLAMACPP_API_KEY="secret")
    client = LlamaCppAPIClient(settings, transport=httpx.MockTransport(handler))
    backend = LlamaCppModelBackend(MODEL_ID, settings, client=client)
    async def scenario():
        await backend.start()
        result = await backend.generate_chat(_request())
        await backend.shutdown()
        return result

    result = asyncio.run(scenario())

    assert result.text == "ready"
    assert result.reasoning == "checked"
    assert result.total_tokens == 3
    assert requests[0].headers["authorization"] == "Bearer secret"


def test_llamacpp_backend_streams_openai_events(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":" two"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    settings = _settings(tmp_path)
    client = LlamaCppAPIClient(settings, transport=httpx.MockTransport(handler))
    backend = LlamaCppModelBackend(MODEL_ID, settings, client=client)
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }

    async def scenario():
        events = [event async for event in backend.openai_chat_stream(payload)]
        await backend.shutdown()
        return events

    events = asyncio.run(scenario())

    assert events[0]["choices"][0]["delta"]["content"] == "one"
    assert events[1]["choices"][0]["delta"]["content"] == " two"


def test_llamacpp_backend_rejects_wrong_alias(tmp_path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "different-model"}]})

    settings = _settings(tmp_path)
    client = LlamaCppAPIClient(settings, transport=httpx.MockTransport(handler))
    backend = LlamaCppModelBackend(MODEL_ID, settings, client=client)

    async def scenario():
        try:
            await backend.start()
        finally:
            await backend.shutdown()

    with pytest.raises(NotReadyError, match="--alias"):
        asyncio.run(scenario())
