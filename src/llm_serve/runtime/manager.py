from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from llm_serve.config import Settings
from llm_serve.errors import (
    BadRequestError,
    ConflictError,
    GatewayTimeoutError,
    TooManyRequestsError,
)
from llm_serve.types import InferenceRequest, InferenceResult, LoadStatus

from .base import ModelBackend
from .autotune import AutoTunedVLLMBackend, FixedTopologyVLLMBackend, ManualVLLMBackend
from .diagnostics import StartupDiagnostics
from .mock import MockModelBackend
from .ollama_backend import OllamaAPIClient, OllamaModelBackend


BackendFactory = Callable[[str], Awaitable[ModelBackend]]
OllamaClientFactory = Callable[[Settings], OllamaAPIClient]
logger = logging.getLogger(__name__)


class RuntimeManager:
    def __init__(
        self,
        settings: Settings,
        backend_factory: Optional[BackendFactory] = None,
        ollama_client_factory: Optional[OllamaClientFactory] = None,
    ) -> None:
        self.settings = settings
        self._backend_factory = backend_factory or self._default_factory
        self._ollama_client_factory = ollama_client_factory or OllamaAPIClient
        self._active_backend: Optional[ModelBackend] = None
        self._active_model_id: Optional[str] = None
        self._switch_lock = asyncio.Lock()
        self._usage_lock = asyncio.Lock()
        self._usage_idle = asyncio.Event()
        self._usage_idle.set()
        self._active_usages = 0
        self._switch_task: Optional[asyncio.Task[None]] = None
        self._switch_target_model: Optional[str] = None
        self._last_switch_error: Optional[str] = None
        self._foreground_pending = 0
        self._batch_pending = 0
        self._pending_lock = asyncio.Lock()
        self._foreground_capacity = settings.prompt_max_parallel
        self._foreground_semaphore = asyncio.Semaphore(self._foreground_capacity)
        self._batch_semaphore = asyncio.Semaphore(settings.batch_max_parallel)
        self._total_capacity = settings.prompt_max_parallel + settings.batch_max_parallel
        self._total_semaphore = asyncio.Semaphore(self._total_capacity)
        self._diagnostics = StartupDiagnostics(
            settings,
            self._generate_with_active_backend,
            self.run_foreground,
        )

    async def startup(self) -> None:
        if not self.settings.startup_load_default_model:
            self._diagnostics.skip()
            logger.info("Startup default-model load is disabled; skipping startup self-test.")
            return

        default_model_id = self.settings.effective_default_model_id
        logger.info("Loading default model '%s' during startup.", default_model_id)
        await self.ensure_loaded(default_model_id)
        logger.info("Default model '%s' is ready.", default_model_id)
        await self._diagnostics.start()

    async def shutdown(self) -> None:
        await self._diagnostics.shutdown()
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
        model_id = requested_model or self.settings.effective_default_model_id
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
            self._start_switch(model_id, pull_first=self.settings.inference_backend == "ollama")
        return LoadStatus(state="spinning_up", model_id=model_id, current_model=self._active_model_id)

    async def list_tags(self) -> List[Dict[str, Any]]:
        if self.settings.inference_backend != "ollama":
            return [
                {
                    "name": model["id"],
                    "model": model["id"],
                    "size": 0,
                    "digest": "",
                    "details": {"loaded": model["loaded"]},
                }
                for model in self.list_models()
            ]

        client = self._ollama_client_factory(self.settings)
        try:
            upstream_models = await client.list_models()
        finally:
            await client.close()

        filtered_models = []
        for model in upstream_models:
            model_name = OllamaAPIClient.extract_model_name(model)
            if model_name not in self.settings.model_allowlist:
                continue
            normalized = dict(model)
            normalized["name"] = normalized.get("name") or model_name
            normalized["model"] = model_name
            details = dict(normalized.get("details") or {})
            details["loaded"] = model_name == self._active_model_id
            normalized["details"] = details
            filtered_models.append(normalized)
        return filtered_models

    async def ensure_loaded(self, model_id: str) -> None:
        while True:
            if (
                self._active_model_id == model_id
                and self._active_backend is not None
                and not self.switch_in_progress
            ):
                return

            task = self._switch_task
            target = self._switch_target_model
            if task is not None and not task.done():
                await self._await_switch(task)
                continue

            async with self._switch_lock:
                if (
                    self._active_model_id == model_id
                    and self._active_backend is not None
                    and not self.switch_in_progress
                ):
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
        async with self._pending_lock:
            if self._foreground_pending >= self._foreground_capacity + self.settings.foreground_queue_limit:
                raise TooManyRequestsError("Foreground queue is full")
            self._foreground_pending += 1
        foreground_acquired = False
        total_acquired = False
        try:
            await self._foreground_semaphore.acquire()
            foreground_acquired = True
            await self._total_semaphore.acquire()
            total_acquired = True
            async with self._backend_usage():
                backend = self._require_active_backend(request.model_id)
                use_chat = request.messages is not None
                stream = backend.generate_chat_stream(request) if use_chat else backend.generate_stream(request)
                if self.settings.inference_backend != "ollama":
                    stream = self._timeout_stream(stream, self.settings.request_timeout_seconds)
                async for chunk in stream:
                    yield chunk
        except asyncio.TimeoutError as exc:
            raise GatewayTimeoutError("Streaming request timed out") from exc
        finally:
            async with self._pending_lock:
                self._foreground_pending -= 1
            if total_acquired:
                self._total_semaphore.release()
            if foreground_acquired:
                self._foreground_semaphore.release()

    async def run_batch(self, request: InferenceRequest) -> InferenceResult:
        return await self._run_request(
            request=request,
            semaphore=self._batch_semaphore,
            lane="batch",
            wait_for_model=True,
        )

    async def openai_chat(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self.ensure_loaded(model_id)
        async with self._vllm_foreground_backend(model_id) as backend:
            return await backend.openai_chat(payload)

    async def openai_chat_stream(
        self,
        model_id: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        await self.ensure_loaded(model_id)
        async with self._vllm_foreground_backend(model_id) as backend:
            stream = self._timeout_stream(
                backend.openai_chat_stream(payload),
                self.settings.request_timeout_seconds,
            )
            async for event in stream:
                yield event

    async def create_response(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self.ensure_loaded(model_id)
        async with self._vllm_foreground_backend(model_id) as backend:
            return await backend.create_response(payload)

    async def create_response_stream(
        self,
        model_id: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        await self.ensure_loaded(model_id)
        async with self._vllm_foreground_backend(model_id) as backend:
            stream = self._timeout_stream(
                backend.create_response_stream(payload),
                self.settings.request_timeout_seconds,
            )
            async for event in stream:
                yield event

    async def retrieve_response(self, response_id: str) -> Dict[str, Any]:
        async with self._backend_usage():
            backend = self._require_vllm_backend()
            return await backend.retrieve_response(response_id)

    async def retrieve_response_stream(
        self,
        response_id: str,
        starting_after: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        async with self._backend_usage():
            backend = self._require_vllm_backend()
            stream = self._timeout_stream(
                backend.retrieve_response_stream(response_id, starting_after),
                self.settings.request_timeout_seconds,
            )
            async for event in stream:
                yield event

    async def cancel_response(self, response_id: str) -> Dict[str, Any]:
        async with self._backend_usage():
            backend = self._require_vllm_backend()
            return await backend.cancel_response(response_id)

    @property
    def active_model_id(self) -> Optional[str]:
        return self._active_model_id

    @property
    def switch_in_progress(self) -> bool:
        return self._switch_task is not None and not self._switch_task.done()

    def queue_depth(self) -> int:
        queued = self._foreground_pending - self._foreground_capacity
        return max(0, queued)

    def batch_queue_depth(self) -> int:
        queued = self._batch_pending - self.settings.batch_max_parallel
        return max(0, queued)

    def foreground_active(self) -> int:
        return min(self._foreground_pending, self._foreground_capacity)

    def batch_active(self) -> int:
        return min(self._batch_pending, self.settings.batch_max_parallel)

    def health_snapshot(self) -> Dict[str, object]:
        return {
            "loaded": self._active_backend is not None,
            "model_id": self._active_model_id,
            "inference_backend": self.settings.inference_backend,
            "foreground_active": self.foreground_active(),
            "foreground_capacity": self._foreground_capacity,
            "queue_depth": self.queue_depth(),
            "batch_active": self.batch_active(),
            "batch_capacity": self.settings.batch_max_parallel,
            "batch_queue_depth": self.batch_queue_depth(),
            "switch_in_progress": self.switch_in_progress,
            "switch_target_model": self._switch_target_model,
            "last_switch_error": self._last_switch_error,
            "startup_self_test": self._diagnostics.snapshot(),
            "throughput_tuning": self._throughput_tuning_snapshot(),
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
            self._foreground_capacity + self.settings.foreground_queue_limit
            if lane == "foreground"
            else self.settings.batch_max_parallel + self.settings.batch_queue_limit
        )

        async with self._pending_lock:
            if getattr(self, pending_attr) >= limit:
                raise TooManyRequestsError("%s queue is full" % lane.capitalize())
            setattr(self, pending_attr, getattr(self, pending_attr) + 1)

        lane_acquired = False
        total_acquired = False
        try:
            await semaphore.acquire()
            lane_acquired = True
            prioritize_foreground = (
                lane == "batch"
                and self.settings.inference_backend == "vllm"
                and self.settings.vllm_gpu_mode == "auto"
            )
            if prioritize_foreground:
                while True:
                    while self._foreground_pending > 0:
                        await asyncio.sleep(0.01)
                    await self._total_semaphore.acquire()
                    total_acquired = True
                    if self._foreground_pending == 0:
                        break
                    self._total_semaphore.release()
                    total_acquired = False
            else:
                await self._total_semaphore.acquire()
                total_acquired = True
            if wait_for_model:
                await self.ensure_loaded(request.model_id)
            return await self._generate_with_active_backend(request)
        except asyncio.TimeoutError as exc:
            raise GatewayTimeoutError("Inference request timed out") from exc
        finally:
            async with self._pending_lock:
                setattr(self, pending_attr, getattr(self, pending_attr) - 1)
            if total_acquired:
                self._total_semaphore.release()
            if lane_acquired:
                semaphore.release()

    async def _await_switch(self, task: Optional[asyncio.Task[None]]) -> None:
        if task is None:
            return
        timeout = self.settings.switch_timeout_seconds
        timeout_name = "SWITCH_TIMEOUT_SECONDS"
        if self.settings.inference_backend == "vllm" and self.settings.vllm_gpu_mode == "auto":
            timeout = self.settings.vllm_autotune_timeout_seconds
            timeout_name = "VLLM_AUTOTUNE_TIMEOUT_SECONDS"
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise GatewayTimeoutError(
                "Model switch timed out after %.0f seconds; increase %s if startup is still progressing"
                % (timeout, timeout_name)
            ) from exc

    async def _perform_switch(self, model_id: str, pull_first: bool = False) -> None:
        self._last_switch_error = None
        await self._usage_idle.wait()
        active_backend = self._active_backend
        wait_for_background = getattr(active_backend, "wait_for_background", None)
        if callable(wait_for_background):
            await wait_for_background()
        if pull_first and self.settings.inference_backend == "ollama":
            client = self._ollama_client_factory(self.settings)
            try:
                await client.pull_model(model_id)
            finally:
                await client.close()
        new_backend = await self._backend_factory(model_id)
        try:
            await new_backend.start()
        except BaseException:
            await self._shutdown_backend_quietly(
                new_backend,
                context="partially started backend for '%s'" % model_id,
            )
            raise
        old_backend = self._active_backend
        self._active_backend = new_backend
        self._active_model_id = model_id
        self._switch_target_model = model_id
        self._apply_backend_capacity(new_backend)
        if old_backend is not None:
            await old_backend.shutdown()

    def _start_switch(self, model_id: str, pull_first: bool = False) -> None:
        if self.switch_in_progress:
            return
        self._switch_target_model = model_id
        self._switch_task = asyncio.create_task(self._perform_switch(model_id, pull_first=pull_first))
        self._switch_task.add_done_callback(self._clear_finished_switch)

    def _clear_finished_switch(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._last_switch_error = str(exc)
            self._switch_target_model = None
        finally:
            if self._switch_task is task:
                self._switch_task = None

    async def _shutdown_backend_quietly(self, backend: ModelBackend, *, context: str) -> None:
        try:
            await backend.shutdown()
        except Exception as exc:
            logger.warning("Backend shutdown failed while cleaning up %s: %s", context, exc)

    async def _default_factory(self, model_id: str) -> ModelBackend:
        if self.settings.inference_backend == "mock":
            return MockModelBackend(model_id, self.settings)
        if self.settings.inference_backend == "ollama":
            return OllamaModelBackend(
                model_id,
                self.settings,
                client_factory=self._ollama_client_factory,
            )
        if self.settings.vllm_gpu_mode == "auto":
            return AutoTunedVLLMBackend(model_id, self.settings)
        if self.settings.vllm_gpu_mode == "fixed":
            return FixedTopologyVLLMBackend(model_id, self.settings)
        return ManualVLLMBackend(model_id, self.settings)

    def _apply_backend_capacity(self, backend: ModelBackend) -> None:
        configured = getattr(backend, "effective_foreground_concurrency", None)
        capacity = configured if isinstance(configured, int) and configured > 0 else self.settings.prompt_max_parallel
        self._foreground_capacity = capacity
        self._foreground_semaphore = asyncio.Semaphore(capacity)
        self._total_capacity = (
            capacity
            if isinstance(configured, int) and configured > 0
            else self.settings.prompt_max_parallel + self.settings.batch_max_parallel
        )
        self._total_semaphore = asyncio.Semaphore(self._total_capacity)

    def _throughput_tuning_snapshot(self) -> Dict[str, object]:
        if self.settings.inference_backend != "vllm" or self.settings.vllm_gpu_mode not in {
            "auto",
            "fixed",
        }:
            return {"status": "disabled", "profile_valid": False}
        backend = self._active_backend
        snapshot = getattr(backend, "tuning_snapshot", None) if backend is not None else None
        if isinstance(snapshot, dict):
            return snapshot
        return {"status": "pending", "profile_valid": False}

    def _require_active_backend(self, model_id: str) -> ModelBackend:
        if self._active_backend is None or self._active_model_id != model_id:
            raise ConflictError("Requested model '%s' is not currently loaded" % model_id)
        return self._active_backend

    def _require_vllm_backend(self):
        if self.settings.inference_backend != "vllm" or self._active_backend is None:
            raise BadRequestError("This endpoint requires INFERENCE_BACKEND=vllm")
        return self._active_backend

    @asynccontextmanager
    async def _vllm_foreground_backend(self, model_id: str):
        if self.settings.inference_backend != "vllm":
            raise BadRequestError("Agent-compatible OpenAI proxying requires INFERENCE_BACKEND=vllm")
        async with self._pending_lock:
            limit = self._foreground_capacity + self.settings.foreground_queue_limit
            if self._foreground_pending >= limit:
                raise TooManyRequestsError("Foreground queue is full")
            self._foreground_pending += 1
        foreground_acquired = False
        total_acquired = False
        try:
            await self._foreground_semaphore.acquire()
            foreground_acquired = True
            await self._total_semaphore.acquire()
            total_acquired = True
            async with self._backend_usage():
                yield self._require_active_backend(model_id)
        finally:
            async with self._pending_lock:
                self._foreground_pending -= 1
            if total_acquired:
                self._total_semaphore.release()
            if foreground_acquired:
                self._foreground_semaphore.release()

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

    async def _timeout_stream(self, stream: AsyncIterator[str], timeout: float) -> AsyncIterator[str]:
        ait = stream.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(ait.__anext__(), timeout=timeout)
                yield chunk
            except StopAsyncIteration:
                return

    async def _generate_with_active_backend(self, request: InferenceRequest) -> InferenceResult:
        async with self._backend_usage():
            backend = self._require_active_backend(request.model_id)
            use_chat = request.messages is not None
            coro = backend.generate_chat(request) if use_chat else backend.generate(request)
            if self.settings.inference_backend == "ollama":
                return await coro
            return await asyncio.wait_for(
                coro,
                timeout=self.settings.request_timeout_seconds,
            )
