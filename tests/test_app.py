from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from llm_serve.app import create_app

from .conftest import make_settings


def test_healthz_and_openai_chat_completion(tmp_path):
    settings = make_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["loaded"] is True
        assert health.json()["model_id"] == "mock/default"

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "mock/default"
    assert "mock-response[mock/default]" in payload["choices"][0]["message"]["content"]


def test_reasoning_allowlist_is_enforced(tmp_path):
    settings = make_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock/default",
                "messages": [{"role": "user", "content": "Think harder"}],
                "reasoning_effort": "medium",
            },
        )

    assert response.status_code == 400
    assert "REASONING_MODEL_ALLOWLIST" in response.json()["error"]["message"]


def test_request_to_unloaded_model_returns_202_then_succeeds(tmp_path):
    settings = make_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        first = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock/reasoning",
                "messages": [{"role": "user", "content": "Switch models"}],
            },
        )
        assert first.status_code == 202
        time.sleep(0.05)
        second = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock/reasoning",
                "messages": [{"role": "user", "content": "Switch models"}],
            },
        )

    assert second.status_code == 200
    assert second.json()["model"] == "mock/reasoning"


def test_openai_streaming_endpoint_emits_done_marker(tmp_path):
    settings = make_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Count to three"}],
                "stream": True,
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "data: [DONE]" in body
    assert "chat.completion.chunk" in body


def test_batch_file_upload_and_processing(tmp_path):
    settings = make_settings(tmp_path)
    batch_line = json.dumps(
        {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": [{"role": "user", "content": "Say hello from batch"}],
                "stream": False,
            },
        }
    ) + "\n"

    with TestClient(create_app(settings)) as client:
        upload = client.post(
            "/v1/files",
            data={"purpose": "batch"},
            files={"file": ("batch.jsonl", batch_line.encode("utf-8"), "application/jsonl")},
        )
        assert upload.status_code == 200
        file_id = upload.json()["id"]

        create_batch = client.post(
            "/v1/batches",
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        assert create_batch.status_code == 200
        batch_id = create_batch.json()["id"]

        batch = _wait_for_batch(client, batch_id)
        output_content = client.get("/v1/files/%s/content" % batch["output_file_id"])

    assert batch["status"] == "completed"
    assert batch["request_counts"]["completed"] == 1
    assert output_content.status_code == 200
    assert "req-1" in output_content.text


def _wait_for_batch(client: TestClient, batch_id: str):
    deadline = time.time() + 3.0
    last_payload = None
    while time.time() < deadline:
        response = client.get("/v1/batches/%s" % batch_id)
        response.raise_for_status()
        last_payload = response.json()
        if last_payload["status"] in {"completed", "failed", "cancelled"}:
            return last_payload
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for batch completion: %s" % last_payload)
