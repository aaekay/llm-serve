from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from llm_serve.app import create_app
from llm_serve.errors import NotReadyError, UpstreamTimeoutError
from llm_serve.tokenization import estimate_text_tokens
from llm_serve.types import InferenceResult
import llm_serve.runtime.manager as runtime_manager_module
import llm_serve.runtime.ollama_backend as ollama_backend_module

from .conftest import make_settings


class FakeOllamaAPIClient:
    installed_models = {"mock/default"}
    pulled_models = []
    generated_requests = []

    def __init__(self, settings):
        self.settings = settings

    async def close(self) -> None:
        return None

    @staticmethod
    def extract_model_name(model):
        return model.get("model") or model.get("name") or ""

    async def list_models(self):
        return [
            {
                "name": model_id,
                "model": model_id,
                "size": len(model_id),
                "digest": "sha256:%s" % model_id.replace("/", "-"),
                "details": {"family": "fake"},
            }
            for model_id in sorted(self.installed_models)
        ]

    async def assert_model_installed(self, model_id: str):
        if model_id not in self.installed_models:
            raise NotReadyError("Model '%s' is not installed" % model_id)
        return {"model": model_id}

    async def pull_model(self, model_id: str):
        self.installed_models.add(model_id)
        self.pulled_models.append(model_id)
        return {"status": "success"}

    async def generate(self, request):
        self.generated_requests.append(
            {
                "prompt": request.prompt,
                "timeout": request.upstream_timeout_seconds,
                "max_output_tokens": request.max_output_tokens,
                "timeout_retry_enabled": request.timeout_retry_enabled,
            }
        )
        if "always timeout" in request.prompt:
            raise UpstreamTimeoutError("Timed out while generating from Ollama at %s." % self.settings.ollama_base_url)
        if "retry after timeout" in request.prompt:
            timeout = request.upstream_timeout_seconds or self.settings.ollama_request_timeout_seconds
            if timeout <= self.settings.ollama_request_timeout_seconds:
                raise UpstreamTimeoutError("Timed out while generating from Ollama at %s." % self.settings.ollama_base_url)
        text = "ollama-response[%s]: %s" % (request.model_id, request.prompt)
        return InferenceResult(
            model_id=request.model_id,
            text=text,
            prompt_tokens=estimate_text_tokens(request.prompt),
            completion_tokens=estimate_text_tokens(text),
            reasoning="fake reasoning" if request.include_reasoning else None,
        )

    async def chat(self, request):
        return await self.generate(request)

    async def chat_stream(self, request):
        yield "ollama "
        yield "stream"

    async def generate_stream(self, request):
        yield "ollama "
        yield "stream"


def test_ollama_backend_preserves_chat_tags_and_pull_api(tmp_path, monkeypatch):
    _patch_fake_ollama(monkeypatch)
    FakeOllamaAPIClient.installed_models = {"mock/default"}
    FakeOllamaAPIClient.pulled_models = []
    FakeOllamaAPIClient.generated_requests = []
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="ollama",
        STARTUP_SELF_TEST_ENABLED="false",
    )

    with TestClient(create_app(settings)) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["inference_backend"] == "ollama"
        assert health.json()["model_id"] == "mock/default"

        tags = client.get("/api/tags")
        assert tags.status_code == 200
        assert [model["model"] for model in tags.json()["models"]] == ["mock/default"]
        assert tags.json()["models"][0]["details"]["loaded"] is True

        completion = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": False,
            },
        )
        assert completion.status_code == 200
        assert "ollama-response[mock/default]" in completion.json()["choices"][0]["message"]["content"]

        pull = client.post("/api/pull", json={"name": "mock/reasoning"})
        assert pull.status_code == 202
        assert pull.headers["Retry-After"] == "2"
        assert pull.json()["retry_after_seconds"] == 2
        _wait_for_openai_model(client, "mock/reasoning")
        pulled_completion = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock/reasoning",
                "messages": [{"role": "user", "content": "Switch models"}],
                "stream": False,
            },
        )
        assert pulled_completion.status_code == 200
        assert pulled_completion.json()["model"] == "mock/reasoning"

        refreshed_tags = client.get("/api/tags")

    assert FakeOllamaAPIClient.pulled_models == ["mock/reasoning"]
    assert [model["model"] for model in refreshed_tags.json()["models"]] == ["mock/default", "mock/reasoning"]


def test_ollama_backend_uses_ollama_default_model_override(tmp_path, monkeypatch):
    _patch_fake_ollama(monkeypatch)
    FakeOllamaAPIClient.installed_models = {"mock/default", "mock/reasoning"}
    FakeOllamaAPIClient.pulled_models = []
    FakeOllamaAPIClient.generated_requests = []
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="ollama",
        DEFAULT_MODEL_ID="mock/default",
        OLLAMA_DEFAULT_MODEL_ID="mock/reasoning",
        STARTUP_SELF_TEST_ENABLED="false",
    )

    with TestClient(create_app(settings)) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["model_id"] == "mock/reasoning"

        completion = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Use the ollama default"}],
                "stream": False,
            },
        )

    assert completion.status_code == 200
    assert completion.json()["model"] == "mock/reasoning"
    assert "ollama-response[mock/reasoning]" in completion.json()["choices"][0]["message"]["content"]


def test_ollama_backend_supports_batch_jobs(tmp_path, monkeypatch):
    _patch_fake_ollama(monkeypatch)
    FakeOllamaAPIClient.installed_models = {"mock/default"}
    FakeOllamaAPIClient.pulled_models = []
    FakeOllamaAPIClient.generated_requests = []
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="ollama",
        STARTUP_SELF_TEST_ENABLED="false",
    )
    batch_line = json.dumps(
        {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": [{"role": "user", "content": "Say hello from Ollama batch"}],
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
        create_batch = client.post(
            "/v1/batches",
            json={
                "input_file_id": upload.json()["id"],
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        assert create_batch.status_code == 200
        batch = _wait_for_batch(client, create_batch.json()["id"])
        output = client.get("/v1/files/%s/content" % batch["output_file_id"])

    assert batch["status"] == "completed"
    assert output.status_code == 200
    assert "ollama-response[mock/default]" in output.text


def test_ollama_batch_retries_timeouts_with_larger_limits(tmp_path, monkeypatch):
    _patch_fake_ollama(monkeypatch)
    FakeOllamaAPIClient.installed_models = {"mock/default"}
    FakeOllamaAPIClient.pulled_models = []
    FakeOllamaAPIClient.generated_requests = []
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="ollama",
        STARTUP_SELF_TEST_ENABLED="false",
        OLLAMA_REQUEST_TIMEOUT_SECONDS="3",
        OLLAMA_BATCH_TIMEOUT_RETRY_MULTIPLIER="4",
        OLLAMA_BATCH_RETRY_OUTPUT_TOKENS_MULTIPLIER="2",
        OLLAMA_BATCH_RETRY_MAX_OUTPUT_TOKENS="256",
    )
    batch_line = _build_batch_line(
        "req-timeout",
        "retry after timeout",
        max_completion_tokens=40,
    )

    with TestClient(create_app(settings)) as client:
        upload = client.post(
            "/v1/files",
            data={"purpose": "batch"},
            files={"file": ("batch.jsonl", batch_line.encode("utf-8"), "application/jsonl")},
        )
        create_batch = client.post(
            "/v1/batches",
            json={
                "input_file_id": upload.json()["id"],
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        batch = _wait_for_batch(client, create_batch.json()["id"])
        output = client.get("/v1/files/%s/content" % batch["output_file_id"])
        error = client.get("/v1/files/%s/content" % batch["error_file_id"])

    assert batch["status"] == "completed"
    assert batch["request_counts"]["completed"] == 1
    assert batch["request_counts"]["failed"] == 0
    assert batch["metadata"]["ollama_timeout_retry"] == {"scheduled": 1, "succeeded": 1, "failed": 0}
    assert "retry after timeout" in output.text
    assert error.text == ""
    matching_requests = [
        request for request in FakeOllamaAPIClient.generated_requests if "retry after timeout" in request["prompt"]
    ]
    assert len(matching_requests) == 2
    assert matching_requests[0]["timeout"] is None
    assert matching_requests[0]["max_output_tokens"] == 40
    assert matching_requests[1]["timeout"] == 12
    assert matching_requests[1]["max_output_tokens"] == 80


def test_ollama_batch_records_final_failure_after_retry(tmp_path, monkeypatch):
    _patch_fake_ollama(monkeypatch)
    FakeOllamaAPIClient.installed_models = {"mock/default"}
    FakeOllamaAPIClient.pulled_models = []
    FakeOllamaAPIClient.generated_requests = []
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="ollama",
        STARTUP_SELF_TEST_ENABLED="false",
        OLLAMA_REQUEST_TIMEOUT_SECONDS="2",
        OLLAMA_BATCH_TIMEOUT_RETRY_MULTIPLIER="3",
        OLLAMA_BATCH_RETRY_OUTPUT_TOKENS_MULTIPLIER="2",
        OLLAMA_BATCH_RETRY_MAX_OUTPUT_TOKENS="256",
    )
    batch_line = _build_batch_line("req-fail", "always timeout", max_completion_tokens=24)

    with TestClient(create_app(settings)) as client:
        upload = client.post(
            "/v1/files",
            data={"purpose": "batch"},
            files={"file": ("batch.jsonl", batch_line.encode("utf-8"), "application/jsonl")},
        )
        create_batch = client.post(
            "/v1/batches",
            json={
                "input_file_id": upload.json()["id"],
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        batch = _wait_for_batch(client, create_batch.json()["id"])
        output = client.get("/v1/files/%s/content" % batch["output_file_id"])
        error = client.get("/v1/files/%s/content" % batch["error_file_id"])

    assert batch["status"] == "failed"
    assert batch["request_counts"]["completed"] == 0
    assert batch["request_counts"]["failed"] == 1
    assert batch["metadata"]["ollama_timeout_retry"] == {"scheduled": 1, "succeeded": 0, "failed": 1}
    assert output.text == ""
    assert error.text.count("req-fail") == 1
    matching_requests = [request for request in FakeOllamaAPIClient.generated_requests if "always timeout" in request["prompt"]]
    assert len(matching_requests) == 2
    assert matching_requests[1]["timeout"] == 6
    assert matching_requests[1]["max_output_tokens"] == 48


def _patch_fake_ollama(monkeypatch) -> None:
    monkeypatch.setattr(runtime_manager_module, "OllamaAPIClient", FakeOllamaAPIClient)
    monkeypatch.setattr(ollama_backend_module, "OllamaAPIClient", FakeOllamaAPIClient)


def _wait_for_openai_model(client: TestClient, model_id: str):
    deadline = time.time() + 3.0
    last_response = None
    while time.time() < deadline:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "ready?"}],
                "stream": False,
            },
        )
        last_response = response
        if response.status_code == 200:
            return response
        assert response.status_code == 202
        assert response.headers["Retry-After"] == "2"
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for model load: %s" % last_response.json())


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


def _build_batch_line(custom_id: str, content: str, max_completion_tokens: int) -> str:
    return json.dumps(
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": [{"role": "user", "content": content}],
                "stream": False,
                "max_completion_tokens": max_completion_tokens,
            },
        }
    ) + "\n"
