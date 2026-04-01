from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable, Dict, List, Optional

from llm_serve.config import Settings
from llm_serve.errors import (
    BadRequestError,
    ConflictError,
    GatewayTimeoutError,
    TooManyRequestsError,
)
from llm_serve.types import InferenceRequest, InferenceResult, LoadStatus

from .base import ModelBackend
from .mock import MockModelBackend
from .vllm_backend import VLLMModelBackend


BackendFactory = Callable[[str], Awaitable[ModelBackend]]


class RuntimeManager:
    def __init__(self, settings: Settings, backend_factory: Optional[BackendFactory] = None) -> None:
        self.settings = settings
        self._backend_factory = backend_factory or self._default_factory
        self._active_backend: Optional[ModelBackend] = None
        self._active_model_id: Optional[str] = None
        self._switch_lock = asyncio.Lock()
        self._usage_lock = asyncio.Lock()
        self._usage_idle = asyncio.Event()
        self._usage_idle.set()
        self._active_usages = 0
        self._switch_task: Optional["asyncio.Task[None]"] = None
        self._switch_target_model: Optional[str] = None
        self._last_switch_error: Optional[str] = None
        self._foreground_pending = 0
        self._batch_pending = 0
        self._foreground_semaphore = asyncio.Semaphore(settings.prompt_max_parallel)
        self._batch_semaphore = asyncio.Semaphore(settings.batch_max_parallel)

    async def startup(self) -> None:
        if self.settings.startup_load_default_model:
            await self.ensure_loaded(self.settings.default_model_id)

    async def shutdown(self) -> None:
        if self._switch_task and not self._switch_task.done():
            self._switch_task.cancel()
            try:
                await self._switch_task
            except asyncio.CancelledError:
                pass
        backend = self._active_backend
        self._active_backend = None
        self._active_model_id = None
        if backend is not None:
            await backend.shutdown()

    def list_models(self) -> List[Dict[str, object]]:
        models = []
        for model_id in self.settings.model_allowlist:
            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "llm-serve",
                    "loaded": model_id == self._active_model_id,
                }
            )
        return models

    def resolve_model(self, requested_model: Optional[str], reasoning_effort: Optional[str] = None) -> str:
        model_id = requested_model or self.settings.default_model_id
        if model_id not in self.settings.model_allowlist:
            raise BadRequestError("Model '%s' is not in MODEL_ALLOWLIST" % model_id)
        if reasoning_effort and model_id not in self.settings.reasoning_model_allowlist:
            raise BadRequestError("Model '%s' is not in REASONING_MODEL_ALLOWLIST" % model_id)
        return model_id

    def check_readiness(self, model_id: str) -> LoadStatus:
        if self._active_model_id == model_id and self._active_backend is not None:
            return LoadStatus(state="ready", model_id=model_id, current_model=self._active_model_id)

        if self.switch_in_progress:
            if self._switch_target_model == model_id:
                return LoadStatus(state="spinning_up", model_id=model_id, current_model=self._active_model_id)
            return LoadStatus(state="conflict", model_id=model_id, current_model=self._active_model_id)

        self._start_switch(model_id)
        return LoadStatus(state="spinning_up", model_id=model_id, current_model=self._active_model_id)

    async def pull_model(self, model_id: str) -> LoadStatus:
        if self._active_model_id == model_id and self._active_backend is not None:
            return LoadStatus(state="ready", model_id=model_id, current_model=self._active_model_id)
        if self.switch_in_progress and self._switch_target_model != model_id:
            raise ConflictError("Model switch already in progress for '%s'" % self._switch_target_model)
        if not self.switch_in_progress:
            self._start_switch(model_id)
        return LoadStatus(state="spinning_up", model_id=model_id, current_model=self._active_model_id)

    async def ensure_loaded(self, model_id: str) -> None:
        while True:
            if self._active_model_id == model_id and self._active_backend is not None:
                return

            task = self._switch_task
            target = self._switch_target_model
            if task is not None and not task.done():
                await self._await_switch(task)
                continue

            async with self._switch_lock:
                if self._active_model_id == model_id and self._active_backend is not None:
                    return
                if self._switch_task is None or self._switch_task.done():
                    self._switch_target_model = model_id
                    self._switch_task = asyncio.create_task(self._perform_switch(model_id))
                    self._switch_task.add_done_callback(self._clear_finished_switch)
                    task = self._switch_task
                else:
                    task = self._switch_task
                    target = self._switch_target_model

            if target is not None and target != model_id:
                await self._await_switch(task)
                continue
            await self._await_switch(task)

    async def run_foreground(self, request: InferenceRequest) -> InferenceResult:
        return await self._run_request(
            request=request,
            semaphore=self._foreground_semaphore,
            lane="foreground",
            wait_for_model=False,
        )

    async def stream_foreground(self, request: InferenceRequest) -> AsyncIterator[str]:
        if self._foreground_pending >= self.settings.prompt_max_parallel + self.settings.foreground_queue_limit:
            raise TooManyRequestsError("Foreground queue is full")
        self._foreground_pending += 1
        await self._foreground_semaphore.acquire()
        try:
            async with self._backend_usage():
                backend = self._require_active_backend(request.model_id)
                async for chunk in backend.generate_stream(request):
                    yield chunk
        finally:
            self._foreground_pending -= 1
            self._foreground_semaphore.release()

    async def run_batch(self, request: InferenceRequest) -> InferenceResult:
        return await self._run_request(
            request=request,
            semaphore=self._batch_semaphore,
            lane="batch",
            wait_for_model=True,
        )

    @property
    def active_model_id(self) -> Optional[str]:
        return self._active_model_id

    @property
    def switch_in_progress(self) -> bool:
        return self._switch_task is not None and not self._switch_task.done()

    def queue_depth(self) -> int:
        queued = self._foreground_pending - self.settings.prompt_max_parallel
        return max(0, queued)

    def batch_queue_depth(self) -> int:
        queued = self._batch_pending - self.settings.batch_max_parallel
        return max(0, queued)

    def health_snapshot(self) -> Dict[str, object]:
        return {
            "loaded": self._active_backend is not None,
            "model_id": self._active_model_id,
            "inference_backend": self.settings.inference_backend,
            "queue_depth": self.queue_depth(),
            "batch_queue_depth": self.batch_queue_depth(),
            "switch_in_progress": self.switch_in_progress,
            "switch_target_model": self._switch_target_model,
            "last_switch_error": self._last_switch_error,
        }

    async def _run_request(
        self,
        request: InferenceRequest,
        semaphore: asyncio.Semaphore,
        lane: str,
        wait_for_model: bool,
    ) -> InferenceResult:
        pending_attr = "_foreground_pending" if lane == "foreground" else "_batch_pending"
        limit = (
            self.settings.prompt_max_parallel + self.settings.foreground_queue_limit
            if lane == "foreground"
            else self.settings.batch_max_parallel + self.settings.batch_queue_limit
        )

        if getattr(self, pending_attr) >= limit:
            raise TooManyRequestsError("%s queue is full" % lane.capitalize())

        setattr(self, pending_attr, getattr(self, pending_attr) + 1)
        await semaphore.acquire()
        try:
            if wait_for_model:
                await self.ensure_loaded(request.model_id)
            async with self._backend_usage():
                backend = self._require_active_backend(request.model_id)
                return await asyncio.wait_for(
                    backend.generate(request),
                    timeout=self.settings.request_timeout_seconds,
                )
        except asyncio.TimeoutError as exc:
            raise GatewayTimeoutError("Inference request timed out") from exc
        finally:
            setattr(self, pending_attr, getattr(self, pending_attr) - 1)
            semaphore.release()

    async def _await_switch(self, task: Optional["asyncio.Task[None]"]) -> None:
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=self.settings.switch_timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise GatewayTimeoutError("Model switch timed out") from exc

    async def _perform_switch(self, model_id: str) -> None:
        self._last_switch_error = None
        await self._usage_idle.wait()
        new_backend = await self._backend_factory(model_id)
        await new_backend.start()
        old_backend = self._active_backend
        self._active_backend = new_backend
        self._active_model_id = model_id
        self._switch_target_model = model_id
        if old_backend is not None:
            await old_backend.shutdown()

    def _start_switch(self, model_id: str) -> None:
        if self.switch_in_progress:
            return
        self._switch_target_model = model_id
        self._switch_task = asyncio.create_task(self._perform_switch(model_id))
        self._switch_task.add_done_callback(self._clear_finished_switch)

    def _clear_finished_switch(self, task: "asyncio.Task[None]") -> None:
        try:
            task.result()
        except Exception as exc:
            self._last_switch_error = str(exc)
        finally:
            if self._switch_task is task:
                self._switch_task = None

    async def _default_factory(self, model_id: str) -> ModelBackend:
        if self.settings.inference_backend == "mock":
            return MockModelBackend(model_id, self.settings)
        return VLLMModelBackend(model_id, self.settings)

    def _require_active_backend(self, model_id: str) -> ModelBackend:
        if self._active_backend is None or self._active_model_id != model_id:
            raise ConflictError("Requested model '%s' is not currently loaded" % model_id)
        return self._active_backend

    @asynccontextmanager
    async def _backend_usage(self):
        async with self._usage_lock:
            self._active_usages += 1
            self._usage_idle.clear()
        try:
            yield
        finally:
            async with self._usage_lock:
                self._active_usages -= 1
                if self._active_usages == 0:
                    self._usage_idle.set()
