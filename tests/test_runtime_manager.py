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


def test_runtime_manager_respects_parallel_limit(tmp_path):
    settings = make_settings(tmp_path, PROMPT_MAX_PARALLEL=2, FOREGROUND_QUEUE_LIMIT=10)
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
    settings = make_settings(tmp_path, PROMPT_MAX_PARALLEL=1, FOREGROUND_QUEUE_LIMIT=0)

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
