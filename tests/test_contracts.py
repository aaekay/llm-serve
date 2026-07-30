from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from llm_serve.app import create_app

from .conftest import make_settings


PUBLIC_ROUTES = {
    ("GET", "/api/tags"),
    ("GET", "/healthz"),
    ("GET", "/v1/batches/{batch_id}"),
    ("GET", "/v1/files/{file_id}"),
    ("GET", "/v1/files/{file_id}/content"),
    ("GET", "/v1/models"),
    ("POST", "/api/chat"),
    ("POST", "/api/generate"),
    ("POST", "/api/pull"),
    ("POST", "/v1/batches"),
    ("POST", "/v1/batches/{batch_id}/cancel"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/files"),
}


def test_public_route_contract(tmp_path):
    app = create_app(make_settings(tmp_path, STARTUP_LOAD_DEFAULT_MODEL="false"))
    routes = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
        if route.path.startswith(("/api/", "/v1/", "/healthz"))
    }

    assert routes == PUBLIC_ROUTES


def test_openai_and_ollama_response_contracts(tmp_path):
    settings = make_settings(tmp_path, STARTUP_SELF_TEST_ENABLED="false")

    with TestClient(create_app(settings)) as client:
        openai = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        ollama = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        health = client.get("/healthz")

    assert openai.status_code == 200
    assert set(openai.json()) == {"id", "object", "created", "model", "choices", "usage"}
    assert set(openai.json()["choices"][0]) == {"index", "message", "finish_reason"}
    assert set(openai.json()["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}

    assert ollama.status_code == 200
    assert set(ollama.json()) == {
        "model",
        "created_at",
        "message",
        "done",
        "done_reason",
        "prompt_eval_count",
        "eval_count",
        "total_duration",
    }
    assert set(ollama.json()["message"]) == {"role", "content"}

    assert health.status_code == 200
    assert set(health.json()) == {
        "status",
        "inference_backend",
        "loaded",
        "model_id",
        "switch_in_progress",
        "switch_target_model",
        "queue_depth",
        "foreground_active",
        "foreground_capacity",
        "batch_active",
        "batch_capacity",
        "batch_queue_depth",
        "batch_jobs_total",
        "batch_jobs_in_progress",
        "startup_self_test",
        "throughput_tuning",
    }


def test_env_example_matches_supported_settings():
    root = Path(__file__).parents[1]
    config_source = (root / "src/llm_serve/config.py").read_text(encoding="utf-8")
    example_source = (root / ".env.example").read_text(encoding="utf-8")
    configured = set(re.findall(r'"([A-Z][A-Z0-9_]+)"', config_source))
    documented = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", example_source, re.MULTILINE))

    assert documented == configured
