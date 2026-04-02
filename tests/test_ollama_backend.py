from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from llm_serve.errors import NotReadyError, UpstreamTimeoutError
from llm_serve.runtime.ollama_backend import OllamaAPIClient, OllamaModelBackend
from llm_serve.types import InferenceRequest

from .conftest import make_settings


def test_ollama_backend_start_generate_and_stream(tmp_path):
    captured = {"payloads": []}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "mock/default", "model": "mock/default", "size": 1, "digest": "sha256:abc"}
                    ]
                },
            )
        if request.url.path == "/api/generate":
            payload = json.loads(request.content.decode("utf-8"))
            captured["payloads"].append(payload)
            if payload["stream"] is True:
                content = "\n".join(
                    [
                        json.dumps({"model": "mock/default", "response": "hello ", "done": False}),
                        json.dumps({"model": "mock/default", "response": "world", "done": False}),
                        json.dumps({"model": "mock/default", "response": "", "done": True}),
                    ]
                )
                return httpx.Response(200, content=content.encode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "model": "mock/default",
                    "response": "hello world",
                    "thinking": "debug reasoning",
                    "prompt_eval_count": 11,
                    "eval_count": 2,
                    "done": True,
                },
            )
        raise AssertionError("Unexpected request: %s %s" % (request.method, request.url))

    settings = make_settings(tmp_path, INFERENCE_BACKEND="ollama")
    transport = httpx.MockTransport(handler)
    backend = OllamaModelBackend(
        "mock/default",
        settings,
        client_factory=lambda runtime_settings: OllamaAPIClient(runtime_settings, transport=transport),
    )

    async def scenario():
        await backend.start()
        request = InferenceRequest(
            model_id="mock/default",
            prompt="hello",
            max_output_tokens=64,
            temperature=0.2,
            top_p=0.95,
            stream=False,
            reasoning_effort="high",
            include_reasoning=True,
        )
        result = await backend.generate(request)
        stream_chunks = []
        async for chunk in backend.generate_stream(request):
            stream_chunks.append(chunk)
        await backend.shutdown()
        return result, "".join(stream_chunks)

    result, streamed_text = asyncio.run(scenario())

    assert result.text == "hello world"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 2
    assert result.reasoning == "debug reasoning"
    assert streamed_text == "hello world"
    assert captured["payloads"][0]["model"] == "mock/default"
    assert captured["payloads"][0]["options"]["num_predict"] == 64
    assert captured["payloads"][0]["think"] == "high"


def test_ollama_backend_rejects_missing_model(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": "mock/other", "model": "mock/other", "size": 1, "digest": "sha256:def"}]},
        )

    settings = make_settings(tmp_path, INFERENCE_BACKEND="ollama")
    transport = httpx.MockTransport(handler)
    backend = OllamaModelBackend(
        "mock/default",
        settings,
        client_factory=lambda runtime_settings: OllamaAPIClient(runtime_settings, transport=transport),
    )

    with pytest.raises(NotReadyError):
        asyncio.run(backend.start())


def test_ollama_api_client_pull_model_posts_expected_payload(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"status": "success"})

    settings = make_settings(tmp_path, INFERENCE_BACKEND="ollama")
    transport = httpx.MockTransport(handler)

    async def scenario():
        client = OllamaAPIClient(settings, transport=transport)
        try:
            return await client.pull_model("mock/reasoning")
        finally:
            await client.close()

    payload = asyncio.run(scenario())

    assert payload["status"] == "success"
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/pull"
    assert captured["payload"] == {"model": "mock/reasoning", "stream": False}


def test_ollama_backend_retries_timed_out_non_stream_request_once(tmp_path):
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "mock/default", "model": "mock/default", "size": 1, "digest": "sha256:abc"}]},
            )
        if request.url.path == "/api/generate":
            state["calls"] += 1
            if state["calls"] == 1:
                raise httpx.ReadTimeout("timed out")
            return httpx.Response(
                200,
                json={
                    "model": "mock/default",
                    "response": "retry success",
                    "prompt_eval_count": 4,
                    "eval_count": 2,
                    "done": True,
                },
            )
        raise AssertionError("Unexpected request: %s" % request.url.path)

    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="ollama",
        OLLAMA_REQUEST_TIMEOUT_SECONDS="1",
        OLLAMA_REQUEST_TIMEOUT_RETRY_MULTIPLIER="3",
    )
    transport = httpx.MockTransport(handler)
    backend = OllamaModelBackend(
        "mock/default",
        settings,
        client_factory=lambda runtime_settings: OllamaAPIClient(runtime_settings, transport=transport),
    )

    async def scenario():
        await backend.start()
        request = InferenceRequest(
            model_id="mock/default",
            prompt="hello",
            max_output_tokens=32,
            temperature=0.2,
            top_p=0.95,
            stream=False,
        )
        try:
            return await backend.generate(request)
        finally:
            await backend.shutdown()

    result = asyncio.run(scenario())

    assert result.text == "retry success"
    assert state["calls"] == 2


def test_ollama_backend_raises_timeout_after_retry_exhaustion(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "mock/default", "model": "mock/default", "size": 1, "digest": "sha256:abc"}]},
            )
        if request.url.path == "/api/generate":
            raise httpx.ReadTimeout("still timed out")
        raise AssertionError("Unexpected request: %s" % request.url.path)

    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="ollama",
        OLLAMA_REQUEST_TIMEOUT_SECONDS="1",
        OLLAMA_REQUEST_TIMEOUT_RETRY_MULTIPLIER="2",
    )
    transport = httpx.MockTransport(handler)
    backend = OllamaModelBackend(
        "mock/default",
        settings,
        client_factory=lambda runtime_settings: OllamaAPIClient(runtime_settings, transport=transport),
    )

    async def scenario():
        await backend.start()
        request = InferenceRequest(
            model_id="mock/default",
            prompt="hello",
            max_output_tokens=32,
            temperature=0.2,
            top_p=0.95,
            stream=False,
        )
        try:
            await backend.generate(request)
        finally:
            await backend.shutdown()

    with pytest.raises(UpstreamTimeoutError):
        asyncio.run(scenario())
