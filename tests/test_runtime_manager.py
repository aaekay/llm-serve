from __future__ import annotations

import asyncio

import pytest

from llm_serve.errors import TooManyRequestsError
from llm_serve.runtime.base import ModelBackend
from llm_serve.runtime.manager import RuntimeManager
from llm_serve.tokenization import estimate_text_tokens
from llm_serve.types import InferenceRequest, InferenceResult

from .conftest import make_settings


class TrackingBackend(ModelBackend):
    def __init__(self, model_id: str, delay: float = 0.05) -> None:
        super().__init__(model_id)
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._lock = asyncio.Lock()

    async def shutdown(self) -> None:
        return None

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        async with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            return InferenceResult(
                model_id=self.model_id,
                text="ok",
                prompt_tokens=estimate_text_tokens(request.prompt),
                completion_tokens=estimate_text_tokens("ok"),
            )
        finally:
            async with self._lock:
                self.active -= 1

    async def generate_stream(self, request: InferenceRequest):
        yield "ok"


class FailingSelfTestBackend(ModelBackend):
    async def shutdown(self) -> None:
        return None

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        raise RuntimeError("startup self-test exploded")

    async def generate_stream(self, request: InferenceRequest):
        if False:
            yield ""


def test_runtime_manager_respects_parallel_limit(tmp_path):
    settings = make_settings(
        tmp_path,
        PROMPT_MAX_PARALLEL=2,
        FOREGROUND_QUEUE_LIMIT=10,
        STARTUP_SELF_TEST_ENABLED="false",
    )
    created = {}

    async def factory(model_id: str) -> ModelBackend:
        backend = TrackingBackend(model_id)
        created["backend"] = backend
        return backend

    async def scenario() -> int:
        runtime = RuntimeManager(settings, backend_factory=factory)
        await runtime.startup()
        request = InferenceRequest(
            model_id="mock/default",
            prompt="hello",
            max_output_tokens=16,
            temperature=0.2,
            top_p=0.95,
            stream=False,
        )
        await asyncio.gather(*[runtime.run_foreground(request) for _ in range(5)])
        backend = created["backend"]
        await runtime.shutdown()
        return backend.max_active

    assert asyncio.run(scenario()) == 2


def test_runtime_manager_rejects_when_foreground_queue_is_full(tmp_path):
    settings = make_settings(
        tmp_path,
        PROMPT_MAX_PARALLEL=1,
        FOREGROUND_QUEUE_LIMIT=0,
        STARTUP_SELF_TEST_ENABLED="false",
    )

    async def factory(model_id: str) -> ModelBackend:
        return TrackingBackend(model_id, delay=0.1)

    async def scenario():
        runtime = RuntimeManager(settings, backend_factory=factory)
        await runtime.startup()
        request = InferenceRequest(
            model_id="mock/default",
            prompt="hello",
            max_output_tokens=16,
            temperature=0.2,
            top_p=0.95,
            stream=False,
        )
        first = asyncio.create_task(runtime.run_foreground(request))
        await asyncio.sleep(0.01)
        with pytest.raises(TooManyRequestsError):
            await runtime.run_foreground(request)
        await first
        await runtime.shutdown()

    asyncio.run(scenario())


def test_runtime_manager_health_snapshot_reports_active_and_queued_foreground_work(tmp_path):
    settings = make_settings(
        tmp_path,
        PROMPT_MAX_PARALLEL=1,
        FOREGROUND_QUEUE_LIMIT=1,
        STARTUP_SELF_TEST_ENABLED="false",
    )

    async def factory(model_id: str) -> ModelBackend:
        return TrackingBackend(model_id, delay=0.1)

    async def scenario():
        runtime = RuntimeManager(settings, backend_factory=factory)
        await runtime.startup()
        request = InferenceRequest(
            model_id="mock/default",
            prompt="hello",
            max_output_tokens=16,
            temperature=0.2,
            top_p=0.95,
            stream=False,
        )
        first = asyncio.create_task(runtime.run_foreground(request))
        second = asyncio.create_task(runtime.run_foreground(request))
        await asyncio.sleep(0.02)
        snapshot = runtime.health_snapshot()
        await asyncio.gather(first, second)
        await runtime.shutdown()
        return snapshot

    snapshot = asyncio.run(scenario())
    assert snapshot["foreground_active"] == 1
    assert snapshot["foreground_capacity"] == 1
    assert snapshot["queue_depth"] == 1
    assert snapshot["batch_active"] == 0
    assert snapshot["batch_capacity"] == 2


def test_runtime_manager_startup_does_not_wait_for_background_self_test(tmp_path):
    settings = make_settings(tmp_path, MOCK_RESPONSE_DELAY_SECONDS=0.2)

    async def factory(model_id: str) -> ModelBackend:
        return TrackingBackend(model_id, delay=0.2)

    async def scenario():
        runtime = RuntimeManager(settings, backend_factory=factory)
        started = asyncio.get_running_loop().time()
        await runtime.startup()
        elapsed = asyncio.get_running_loop().time() - started
        initial_status = runtime.health_snapshot()["startup_self_test"]["status"]
        await asyncio.sleep(0.25)
        final_status = runtime.health_snapshot()["startup_self_test"]["status"]
        await runtime.shutdown()
        return elapsed, initial_status, final_status

    elapsed, initial_status, final_status = asyncio.run(scenario())
    assert elapsed < 0.15
    assert initial_status in {"queued", "running"}
    assert final_status == "passed"


def test_runtime_manager_background_self_test_does_not_consume_foreground_slots(tmp_path):
    settings = make_settings(
        tmp_path,
        PROMPT_MAX_PARALLEL=1,
        FOREGROUND_QUEUE_LIMIT=0,
        MOCK_RESPONSE_DELAY_SECONDS=0.2,
    )

    async def factory(model_id: str) -> ModelBackend:
        return TrackingBackend(model_id, delay=0.2)

    async def scenario():
        runtime = RuntimeManager(settings, backend_factory=factory)
        await runtime.startup()
        await asyncio.sleep(0.01)
        initial_status = runtime.health_snapshot()["startup_self_test"]["status"]
        request = InferenceRequest(
            model_id="mock/default",
            prompt="hello",
            max_output_tokens=16,
            temperature=0.2,
            top_p=0.95,
            stream=False,
        )
        result = await runtime.run_foreground(request)
        await runtime.shutdown()
        return initial_status, result.text

    initial_status, text = asyncio.run(scenario())
    assert initial_status in {"queued", "running"}
    assert text == "ok"


def test_runtime_manager_uses_backend_specific_default_model(tmp_path):
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        DEFAULT_MODEL_ID="mock/default",
        VLLM_DEFAULT_MODEL_ID="mock/reasoning",
    )

    async def factory(model_id: str) -> ModelBackend:
        return TrackingBackend(model_id, delay=0.01)

    async def scenario():
        runtime = RuntimeManager(settings, backend_factory=factory)
        await runtime.startup()
        selected_model = runtime.resolve_model(None)
        loaded_model = runtime.health_snapshot()["model_id"]
        await asyncio.sleep(0.05)
        self_test = runtime.health_snapshot()["startup_self_test"]
        await runtime.shutdown()
        return selected_model, loaded_model, self_test

    selected_model, loaded_model, self_test = asyncio.run(scenario())
    assert selected_model == "mock/reasoning"
    assert loaded_model == "mock/reasoning"
    assert self_test["model_id"] == "mock/reasoning"
    assert self_test["status"] == "passed"


def test_runtime_manager_logs_background_self_test_failure(tmp_path, caplog):
    settings = make_settings(tmp_path)

    async def factory(model_id: str) -> ModelBackend:
        return FailingSelfTestBackend(model_id)

    async def scenario():
        runtime = RuntimeManager(settings, backend_factory=factory)
        await runtime.startup()
        await asyncio.sleep(0.05)
        snapshot = runtime.health_snapshot()["startup_self_test"]
        await runtime.shutdown()
        return snapshot

    with caplog.at_level("INFO"):
        snapshot = asyncio.run(scenario())

    assert snapshot["status"] == "failed"
    assert snapshot["error"] == "startup self-test exploded"
    assert "Loading default model 'mock/default' during startup." in caplog.text
    assert "Startup self-test queued for model 'mock/default'" in caplog.text
    assert "Startup self-test started for model 'mock/default'" in caplog.text
    assert "Startup self-test failed for model 'mock/default'" in caplog.text
