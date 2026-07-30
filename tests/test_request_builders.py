from __future__ import annotations

import json

import pytest

from llm_serve.batch import BatchManager
from llm_serve.errors import BadRequestError
from llm_serve.request_builders import (
    build_ollama_generate_request,
    build_openai_chat_request,
)
from llm_serve.runtime.manager import RuntimeManager
from llm_serve.schemas import OllamaGenerateRequest, OpenAIChatRequest
from llm_serve.storage import StorageManager

from .conftest import make_settings


def test_direct_and_batch_openai_requests_share_normalization(tmp_path):
    settings = make_settings(tmp_path)
    runtime = RuntimeManager(settings)
    payload = OpenAIChatRequest.model_validate(
        {
            "model": "mock/reasoning",
            "messages": [{"role": "user", "content": "think"}],
            "temperature": 0,
            "top_p": 0.8,
            "max_completion_tokens": 32,
            "reasoning_effort": "medium",
            "include_reasoning": True,
        }
    )
    model_id = runtime.resolve_model(payload.model, payload.reasoning_effort)
    direct = build_openai_chat_request(
        payload,
        model_id,
        settings,
        stream=False,
        timeout_retry_enabled=False,
    )
    manager = BatchManager(StorageManager(settings.storage_root), runtime)
    line = json.dumps(
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": payload.model_dump(),
        }
    )

    batch = manager._prepare_batch_item(line, timeout_retry_enabled=False).inference_request

    assert batch == direct


def test_ollama_generate_builder_applies_defaults_and_limits(tmp_path):
    settings = make_settings(
        tmp_path,
        MAX_OUTPUT_TOKENS=64,
        STARTUP_SELF_TEST_MAX_OUTPUT_TOKENS=64,
    )
    payload = OllamaGenerateRequest.model_validate(
        {
            "model": "mock/default",
            "prompt": "hello",
            "options": {"num_predict": 100, "temperature": 0, "top_p": 0.7},
        }
    )

    request = build_ollama_generate_request(payload, "mock/default", settings)

    assert request.max_output_tokens == 64
    assert request.temperature == 0
    assert request.top_p == 0.7


def test_shared_builder_rejects_oversized_direct_and_batch_prompts(tmp_path):
    settings = make_settings(tmp_path, MAX_INPUT_TOKENS=1)
    payload = OpenAIChatRequest.model_validate(
        {"messages": [{"role": "user", "content": "this prompt is too long"}]}
    )

    with pytest.raises(BadRequestError, match="Prompt too long"):
        build_openai_chat_request(payload, "mock/default", settings)
