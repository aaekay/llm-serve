from __future__ import annotations

import atexit
import asyncio
import json
import logging
import os
import shutil
import signal
import socket
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence

import httpx

from llm_serve.config import Settings
from llm_serve.errors import NotFoundError, NotReadyError, UpstreamAPIError, UpstreamRuntimeError
from llm_serve.prompting import ThinkingContentStripper, split_reasoning_output
from llm_serve.tokenization import estimate_text_tokens
from llm_serve.types import InferenceRequest, InferenceResult

from .base import ModelBackend
from .throughput import TuningProfile, WorkerTopology


logger = logging.getLogger(__name__)
ProcessFactory = Callable[..., asyncio.subprocess.Process]
_tracked_pool_worker_pids: set[int] = set()


def _cleanup_pool_workers_at_exit() -> None:
    for pid in list(_tracked_pool_worker_pids):
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        finally:
            _tracked_pool_worker_pids.discard(pid)


atexit.register(_cleanup_pool_workers_at_exit)


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _vllm_executable() -> str:
    executable = shutil.which("vllm")
    if executable:
        return executable
    adjacent = Path(sys.executable).with_name("vllm")
    if adjacent.exists():
        return str(adjacent)
    raise NotReadyError(
        "The vLLM executable is not installed. Install runtime dependencies with 'uv sync --extra runtime'."
    )


class VLLMWorker:
    def __init__(
        self,
        index: int,
        model_id: str,
        gpu_indices: Sequence[int],
        tensor_parallel_size: int,
        settings: Settings,
        *,
        optimized_engine: bool,
        process_factory: Optional[Callable[..., Any]] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.index = index
        self.model_id = model_id
        self.gpu_indices = list(gpu_indices)
        self.tensor_parallel_size = tensor_parallel_size
        self.settings = settings
        self.optimized_engine = optimized_engine
        self.port = _unused_loopback_port()
        self.base_url = "http://127.0.0.1:%s" % self.port
        self.process: Optional[asyncio.subprocess.Process] = None
        self.healthy = False
        self.outstanding_tokens = 0
        self.restart_attempts = 0
        self._process_factory = process_factory or asyncio.create_subprocess_exec
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=None, transport=transport)

    def command(self) -> List[str]:
        command = [
            _vllm_executable(),
            "serve",
            self.model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--dtype",
            self.settings.vllm_dtype,
            "--tokenizer-mode",
            self.settings.vllm_tokenizer_mode,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.settings.vllm_gpu_memory_utilization),
            "--max-num-seqs",
            str(self.settings.vllm_max_num_seqs),
            "--max-num-batched-tokens",
            str(self.settings.vllm_max_num_batched_tokens),
        ]
        if self.settings.vllm_max_model_len is not None:
            command.extend(["--max-model-len", str(self.settings.vllm_max_model_len)])
        if self.settings.vllm_trust_remote_code:
            command.append("--trust-remote-code")
        if self.settings.vllm_language_model_only:
            command.extend(["--language-model-only", "--skip-mm-profiling"])
        tool_parser = self.settings.vllm_tool_call_parsers.get(self.model_id)
        if tool_parser:
            if self.settings.vllm_enable_auto_tool_choice:
                command.append("--enable-auto-tool-choice")
            command.extend(["--tool-call-parser", tool_parser])
        reasoning_parser = self.settings.vllm_reasoning_parsers.get(self.model_id)
        if reasoning_parser:
            command.extend(["--reasoning-parser", reasoning_parser])
        if self.settings.vllm_enable_prefix_caching:
            command.append("--enable-prefix-caching")
        if self.settings.vllm_enable_chunked_prefill:
            command.append("--enable-chunked-prefill")
        if self.settings.vllm_swap_space_gb is not None:
            command.extend(["--swap-space", str(self.settings.vllm_swap_space_gb)])
        if self.settings.vllm_gdn_prefill_backend is not None:
            command.extend(["--gdn-prefill-backend", self.settings.vllm_gdn_prefill_backend])
        if self.optimized_engine:
            command.append("--async-scheduling")
        else:
            if self.settings.vllm_enforce_eager:
                command.append("--enforce-eager")
            if self.settings.vllm_disable_custom_all_reduce:
                command.append("--disable-custom-all-reduce")
        return command

    async def start(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in self.gpu_indices),
                "HF_HOME": str(self.settings.huggingface_home),
                "HUGGINGFACE_HUB_CACHE": str(self.settings.huggingface_hub_cache),
                "TRANSFORMERS_CACHE": str(self.settings.transformers_cache),
                "VLLM_USE_V1": "1" if self.settings.vllm_use_v1 else "0",
                "VLLM_SKIP_GDN_WARMUP": "1",
                "VLLM_ENABLE_RESPONSES_API_STORE": (
                    "1" if self.settings.vllm_responses_store_enabled else "0"
                ),
            }
        )
        self.settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.process = await self._process_factory(
            *self.command(),
            env=env,
            start_new_session=True,
        )
        _tracked_pool_worker_pids.add(self.process.pid)
        deadline = asyncio.get_running_loop().time() + self.settings.switch_timeout_seconds
        last_error = "worker did not become ready"
        while asyncio.get_running_loop().time() < deadline:
            if self.process.returncode is not None:
                raise NotReadyError(
                    "vLLM worker %s exited during startup with status %s"
                    % (self.index, self.process.returncode)
                )
            try:
                response = await self._client.get("/health", timeout=2)
                if response.status_code == 200:
                    self.healthy = True
                    self.restart_attempts = 0
                    return
                last_error = "health endpoint returned %s" % response.status_code
            except httpx.HTTPError as exc:
                last_error = str(exc)
            await asyncio.sleep(0.5)
        await self.stop(close_client=False)
        raise NotReadyError("Timed out starting vLLM worker %s: %s" % (self.index, last_error))

    async def stop(self, *, close_client: bool = True) -> None:
        self.healthy = False
        process = self.process
        self.process = None
        if process is not None and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                await process.wait()
        if process is not None:
            _tracked_pool_worker_pids.discard(process.pid)
        if close_client:
            await self._client.aclose()

    def exited(self) -> bool:
        return self.process is None or self.process.returncode is not None

    async def request_json(
        self,
        path: str,
        payload: Optional[Dict[str, Any]],
        *,
        method: str = "POST",
    ) -> Dict[str, Any]:
        try:
            request_kwargs: Dict[str, Any] = {"timeout": self.settings.request_timeout_seconds}
            if payload is not None:
                request_kwargs["json"] = payload
            response = await self._client.request(
                method,
                path,
                **request_kwargs,
            )
            response.raise_for_status()
            value = response.json()
        except httpx.HTTPStatusError as exc:
            try:
                payload = exc.response.json()
            except ValueError:
                payload = {"error": {"message": exc.response.text, "type": "upstream_error"}}
            raise UpstreamAPIError(exc.response.status_code, payload) from exc
        except httpx.RequestError as exc:
            self.healthy = False
            raise UpstreamRuntimeError("vLLM worker %s request failed: %s" % (self.index, exc)) from exc
        except ValueError as exc:
            raise UpstreamRuntimeError("vLLM worker %s returned invalid JSON" % self.index) from exc
        if not isinstance(value, dict):
            raise UpstreamRuntimeError("vLLM worker %s returned invalid JSON" % self.index)
        return value

    async def stream_json(
        self,
        path: str,
        payload: Optional[Dict[str, Any]],
        *,
        method: str = "POST",
    ) -> AsyncIterator[Dict[str, Any]]:
        try:
            request_kwargs: Dict[str, Any] = {"timeout": self.settings.request_timeout_seconds}
            if payload is not None:
                request_kwargs["json"] = payload
            async with self._client.stream(method, path, **request_kwargs) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        return
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise UpstreamRuntimeError(
                            "vLLM worker %s returned invalid streaming JSON" % self.index
                        ) from exc
                    if isinstance(item, dict):
                        yield item
        except httpx.HTTPStatusError as exc:
            try:
                payload = await exc.response.aread()
                parsed = json.loads(payload)
            except (ValueError, json.JSONDecodeError):
                parsed = {
                    "error": {
                        "message": "vLLM worker %s rejected a stream" % self.index,
                        "type": "upstream_error",
                    }
                }
            raise UpstreamAPIError(exc.response.status_code, parsed) from exc
        except httpx.RequestError as exc:
            self.healthy = False
            raise UpstreamRuntimeError("vLLM worker %s stream failed: %s" % (self.index, exc)) from exc


class VLLMPoolBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        settings: Settings,
        topology: WorkerTopology,
        *,
        optimized_engine: bool = True,
        profile: Optional[TuningProfile] = None,
        worker_factory: Optional[Callable[..., VLLMWorker]] = None,
    ) -> None:
        super().__init__(model_id)
        self._settings = settings
        self.topology = topology
        self.optimized_engine = optimized_engine
        self.profile = profile
        factory = worker_factory or VLLMWorker
        self.workers = [
            factory(
                index,
                model_id,
                group,
                topology.tensor_parallel_size,
                settings,
                optimized_engine=optimized_engine,
            )
            for index, group in enumerate(topology.gpu_groups)
        ]
        self._selection_lock = asyncio.Lock()
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._stopping = False
        self._response_workers: Dict[str, int] = {}
        self._background_tasks: Dict[str, asyncio.Task[None]] = {}
        self._background_idle = asyncio.Event()
        self._background_idle.set()

    @property
    def effective_foreground_concurrency(self) -> Optional[int]:
        return self.profile.foreground_concurrency if self.profile else None

    @property
    def tuning_snapshot(self) -> Dict[str, Any]:
        profile = self.profile
        return {
            "status": "ready" if any(worker.healthy for worker in self.workers) else "degraded",
            "profile_valid": profile is not None,
            "tensor_parallel_size": self.topology.tensor_parallel_size,
            "replica_count": self.topology.replica_count,
            "foreground_concurrency": self.effective_foreground_concurrency,
            "throughput_tokens_per_second": profile.throughput_tokens_per_second if profile else None,
            "p95_latency_ratio": profile.p95_latency_ratio if profile else None,
            "degraded_workers": sum(not worker.healthy for worker in self.workers),
        }

    async def start(self) -> None:
        try:
            for position, worker in enumerate(self.workers, start=1):
                logger.info(
                    "Starting vLLM replica %s/%s for %s on host GPUs %s (tensor_parallel_size=%s)",
                    position,
                    len(self.workers),
                    self.model_id,
                    worker.gpu_indices,
                    self.topology.tensor_parallel_size,
                )
                await worker.start()
                logger.info(
                    "vLLM replica %s/%s is healthy; %s replica(s) remain to start",
                    position,
                    len(self.workers),
                    len(self.workers) - position,
                )
        except BaseException:
            await asyncio.gather(*(worker.stop() for worker in self.workers), return_exceptions=True)
            raise
        self._monitor_task = asyncio.create_task(self._monitor_workers())

    async def shutdown(self) -> None:
        self._stopping = True
        for task in self._background_tasks.values():
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks.values(), return_exceptions=True)
        self._background_tasks.clear()
        self._background_idle.set()
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        await asyncio.gather(*(worker.stop() for worker in self.workers), return_exceptions=True)

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        return await self._generate(request, chat=False)

    async def generate_chat(self, request: InferenceRequest) -> InferenceResult:
        return await self._generate(request, chat=True)

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        async for chunk in self._stream(request, chat=False):
            yield chunk

    async def generate_chat_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        async for chunk in self._stream(request, chat=True):
            yield chunk

    async def openai_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_tool_support(payload)
        async with self._raw_worker_lease(payload, operation="chat.completions") as worker:
            return await worker.request_json("/v1/chat/completions", payload)

    async def openai_chat_stream(self, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        self._validate_tool_support(payload)
        async with self._raw_worker_lease(payload, operation="chat.completions.stream") as worker:
            async for event in worker.stream_json("/v1/chat/completions", payload):
                yield event

    async def create_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_tool_support(payload)
        response_id = self._prepare_response_id(payload)
        worker_index = self._worker_for_previous_response(payload)
        async with self._raw_worker_lease(
            payload,
            worker_index=worker_index,
            operation="responses.create",
        ) as worker:
            self._response_workers[response_id] = worker.index
            response = await worker.request_json("/v1/responses", payload)
            returned_id = response.get("id")
            if isinstance(returned_id, str):
                self._response_workers[returned_id] = worker.index
                response_id = returned_id
            if payload.get("background") is True and response.get("status") in {"queued", "in_progress"}:
                self._track_background(response_id, worker)
            return response

    async def create_response_stream(self, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        self._validate_tool_support(payload)
        response_id = self._prepare_response_id(payload)
        worker_index = self._worker_for_previous_response(payload)
        async with self._raw_worker_lease(
            payload,
            worker_index=worker_index,
            operation="responses.stream",
        ) as worker:
            self._response_workers[response_id] = worker.index
            async for event in worker.stream_json("/v1/responses", payload):
                response = event.get("response")
                returned_id = response.get("id") if isinstance(response, dict) else None
                if isinstance(returned_id, str):
                    self._response_workers[returned_id] = worker.index
                yield event

    async def retrieve_response(self, response_id: str) -> Dict[str, Any]:
        worker = self._response_worker(response_id)
        return await worker.request_json("/v1/responses/%s" % response_id, None, method="GET")

    async def retrieve_response_stream(
        self,
        response_id: str,
        starting_after: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        worker = self._response_worker(response_id)
        path = "/v1/responses/%s?stream=true" % response_id
        if starting_after is not None:
            path += "&starting_after=%s" % starting_after
        async for event in worker.stream_json(path, None, method="GET"):
            yield event

    async def cancel_response(self, response_id: str) -> Dict[str, Any]:
        worker = self._response_worker(response_id)
        response = await worker.request_json("/v1/responses/%s/cancel" % response_id, {})
        if response.get("status") in {"completed", "failed", "cancelled", "incomplete"}:
            task = self._background_tasks.pop(response_id, None)
            if task is not None:
                task.cancel()
            if not self._background_tasks:
                self._background_idle.set()
        return response

    async def wait_for_background(self) -> None:
        await self._background_idle.wait()

    def _track_background(self, response_id: str, worker: VLLMWorker) -> None:
        if response_id in self._background_tasks:
            return
        self._background_idle.clear()
        task = asyncio.create_task(self._monitor_background(response_id, worker))
        self._background_tasks[response_id] = task

    async def _monitor_background(self, response_id: str, worker: VLLMWorker) -> None:
        try:
            while not self._stopping:
                await asyncio.sleep(0.5)
                response = await worker.request_json(
                    "/v1/responses/%s" % response_id,
                    None,
                    method="GET",
                )
                if response.get("status") in {"completed", "failed", "cancelled", "incomplete"}:
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Background response monitor failed for %s: %s", response_id, exc)
        finally:
            self._background_tasks.pop(response_id, None)
            if not self._background_tasks:
                self._background_idle.set()

    def _validate_tool_support(self, payload: Dict[str, Any]) -> None:
        if payload.get("tools") and self.model_id not in self._settings.vllm_tool_call_parsers:
            raise UpstreamAPIError(
                400,
                {
                    "error": {
                        "message": "Model '%s' has no configured vLLM tool-call parser" % self.model_id,
                        "type": "invalid_request_error",
                    }
                },
            )

    def _prepare_response_id(self, payload: Dict[str, Any]) -> str:
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            request_id = "resp_" + uuid.uuid4().hex
            payload["request_id"] = request_id
        return request_id

    def _worker_for_previous_response(self, payload: Dict[str, Any]) -> Optional[int]:
        previous = payload.get("previous_response_id")
        if not isinstance(previous, str):
            return None
        worker_index = self._response_workers.get(previous)
        if worker_index is None:
            raise NotFoundError("Response with id '%s' not found" % previous)
        return worker_index

    def _response_worker(self, response_id: str) -> VLLMWorker:
        worker_index = self._response_workers.get(response_id)
        if worker_index is None or worker_index >= len(self.workers):
            raise NotFoundError("Response with id '%s' not found" % response_id)
        worker = self.workers[worker_index]
        if not worker.healthy or worker.exited():
            self._response_workers.pop(response_id, None)
            raise NotFoundError("Response with id '%s' is no longer available" % response_id)
        return worker

    @asynccontextmanager
    async def _raw_worker_lease(
        self,
        payload: Dict[str, Any],
        *,
        worker_index: Optional[int] = None,
        operation: str = "inference",
    ):
        cost = max(1, len(json.dumps(payload, ensure_ascii=False)) // 4)
        async with self._selection_lock:
            healthy = [worker for worker in self.workers if worker.healthy and not worker.exited()]
            if not healthy:
                raise NotReadyError("No healthy vLLM workers are available")
            if worker_index is None:
                worker = min(healthy, key=lambda item: (item.outstanding_tokens, item.index))
            else:
                worker = next((item for item in healthy if item.index == worker_index), None)
                if worker is None:
                    raise NotFoundError("The worker holding this response is no longer available")
            worker.outstanding_tokens += cost
            logger.info(
                "Dispatching %s for model %s to worker %s (estimated_tokens=%s outstanding_tokens=%s)",
                operation,
                self.model_id,
                worker.index,
                cost,
                worker.outstanding_tokens,
            )
        try:
            yield worker
        finally:
            async with self._selection_lock:
                worker.outstanding_tokens = max(0, worker.outstanding_tokens - cost)
            logger.info(
                "Completed %s for model %s on worker %s (outstanding_tokens=%s)",
                operation,
                self.model_id,
                worker.index,
                worker.outstanding_tokens,
            )

    async def _generate(self, request: InferenceRequest, *, chat: bool) -> InferenceResult:
        chat = chat or self._completion_uses_chat(request)
        async with self._worker_lease(request) as worker:
            path, payload = self._payload(request, chat=chat, stream=False)
            raw = await worker.request_json(path, payload)
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise UpstreamRuntimeError("vLLM worker returned no completion choices")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if chat else None
        text = message.get("content", "") if isinstance(message, dict) else choice.get("text", "")
        if not isinstance(text, str):
            text = ""
        reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
        parsed_reasoning, answer = split_reasoning_output(text)
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return InferenceResult(
            model_id=self.model_id,
            text=answer,
            prompt_tokens=_integer_or_estimate(usage.get("prompt_tokens"), request.prompt),
            completion_tokens=_integer_or_estimate(usage.get("completion_tokens"), text),
            reasoning=reasoning if isinstance(reasoning, str) else parsed_reasoning,
        )

    async def _stream(self, request: InferenceRequest, *, chat: bool) -> AsyncIterator[str]:
        chat = chat or self._completion_uses_chat(request)
        stripper = ThinkingContentStripper()
        async with self._worker_lease(request) as worker:
            path, payload = self._payload(request, chat=chat, stream=True)
            async for item in worker.stream_json(path, payload):
                choices = item.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    continue
                choice = choices[0]
                delta = choice.get("delta") if chat else None
                text = delta.get("content", "") if isinstance(delta, dict) else choice.get("text", "")
                if not isinstance(text, str):
                    continue
                for visible in stripper.push(text):
                    if visible:
                        yield visible
        tail = stripper.finish()
        if tail:
            yield tail

    def _payload(self, request: InferenceRequest, *, chat: bool, stream: bool) -> tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if chat:
            payload["messages"] = request.messages or [{"role": "user", "content": request.prompt}]
            payload["chat_template_kwargs"] = {
                "enable_thinking": self._thinking_enabled(request),
            }
            return "/v1/chat/completions", payload
        payload["prompt"] = request.prompt
        return "/v1/completions", payload

    def _thinking_enabled(self, request: InferenceRequest) -> bool:
        return self._settings.enable_thinking and bool(request.reasoning_effort)

    def _completion_uses_chat(self, request: InferenceRequest) -> bool:
        return request.messages is None and self._thinking_enabled(request)

    @asynccontextmanager
    async def _worker_lease(self, request: InferenceRequest):
        cost = max(1, estimate_text_tokens(request.prompt) + request.max_output_tokens)
        async with self._selection_lock:
            healthy = [worker for worker in self.workers if worker.healthy and not worker.exited()]
            if not healthy:
                raise NotReadyError("No healthy vLLM workers are available")
            worker = min(healthy, key=lambda item: (item.outstanding_tokens, item.index))
            worker.outstanding_tokens += cost
        try:
            yield worker
        finally:
            async with self._selection_lock:
                worker.outstanding_tokens = max(0, worker.outstanding_tokens - cost)

    async def _monitor_workers(self) -> None:
        while not self._stopping:
            await asyncio.sleep(2)
            for worker in self.workers:
                if worker.healthy and not worker.exited():
                    continue
                worker.healthy = False
                if worker.restart_attempts >= 3:
                    continue
                worker.restart_attempts += 1
                await asyncio.sleep(2 ** (worker.restart_attempts - 1))
                try:
                    self._response_workers = {
                        response_id: owner
                        for response_id, owner in self._response_workers.items()
                        if owner != worker.index
                    }
                    await worker.stop(close_client=False)
                    await worker.start()
                    logger.info("Restarted vLLM worker %s", worker.index)
                except Exception as exc:
                    logger.warning("Failed to restart vLLM worker %s: %s", worker.index, exc)


def _integer_or_estimate(value: Any, text: str) -> int:
    return value if isinstance(value, int) else estimate_text_tokens(text)
