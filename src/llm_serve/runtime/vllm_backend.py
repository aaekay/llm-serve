from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import AsyncIterator, Callable, List, Optional

from llm_serve.config import Settings
from llm_serve.errors import NotReadyError, UpstreamRuntimeError
from llm_serve.tokenization import estimate_text_tokens
from llm_serve.types import InferenceRequest, InferenceResult

from .base import ModelBackend
from .gpu_selection import (
    GPUDeviceInfo,
    GPUSelectionResult,
    build_selection_result,
    query_nvidia_smi_gpus,
    summarize_gpus,
)


logger = logging.getLogger(__name__)


class VLLMModelBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        settings: Settings,
        enable_adaptive_gpu_selection: bool = False,
        gpu_query_fn: Optional[Callable[[], List[GPUDeviceInfo]]] = None,
    ) -> None:
        super().__init__(model_id)
        self._settings = settings
        self._enable_adaptive_gpu_selection = enable_adaptive_gpu_selection
        self._gpu_query_fn = gpu_query_fn or query_nvidia_smi_gpus
        self._engine = None
        self._sampling_params_cls = None
        self._engine_lock = asyncio.Lock()
        self._effective_cuda_visible_devices = settings.cuda_visible_devices
        self._effective_gpu_memory_utilization = settings.vllm_gpu_memory_utilization
        self._selected_gpus: List[GPUDeviceInfo] = []
        self._inspected_gpus: List[GPUDeviceInfo] = []

    async def start(self) -> None:
        await self._ensure_engine()

    async def shutdown(self) -> None:
        engine = self._engine
        self._engine = None
        if engine is None:
            return
        shutdown_background_loop = getattr(engine, "shutdown_background_loop", None)
        if callable(shutdown_background_loop):
            shutdown_background_loop()

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        engine = await self._ensure_engine()
        sampling_params = self._sampling_params_cls(
            max_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        request_id = uuid.uuid4().hex
        final_text = ""
        completion_tokens = 0
        try:
            async for output in engine.generate(request.prompt, sampling_params, request_id):
                if not output.outputs:
                    continue
                final_output = output.outputs[0]
                final_text = final_output.text
                token_ids = getattr(final_output, "token_ids", None)
                if token_ids is not None:
                    completion_tokens = len(token_ids)
        except Exception as exc:  # pragma: no cover - depends on runtime installation
            raise UpstreamRuntimeError("vLLM generation failed: %s" % exc) from exc

        return InferenceResult(
            model_id=self.model_id,
            text=final_text,
            prompt_tokens=estimate_text_tokens(request.prompt),
            completion_tokens=completion_tokens or estimate_text_tokens(final_text),
            reasoning=None,
        )

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        engine = await self._ensure_engine()
        sampling_params = self._sampling_params_cls(
            max_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        request_id = uuid.uuid4().hex
        previous_text = ""

        try:
            async for output in engine.generate(request.prompt, sampling_params, request_id):
                if not output.outputs:
                    continue
                current_text = output.outputs[0].text
                delta = current_text[len(previous_text) :]
                previous_text = current_text
                if delta:
                    yield delta
        except Exception as exc:  # pragma: no cover - depends on runtime installation
            raise UpstreamRuntimeError("vLLM generation failed: %s" % exc) from exc

    async def _ensure_engine(self):
        if self._engine is not None:
            return self._engine

        async with self._engine_lock:
            if self._engine is not None:
                return self._engine
            self._resolve_runtime_configuration()
            self._apply_runtime_environment()
            try:
                from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
            except ImportError as exc:  # pragma: no cover - depends on runtime installation
                raise NotReadyError(
                    "vLLM is not installed. Install the runtime dependencies before using INFERENCE_BACKEND=vllm."
                ) from exc

            engine_args = AsyncEngineArgs(
                model=self.model_id,
                tokenizer=self.model_id,
                dtype=self._settings.vllm_dtype,
                tokenizer_mode=self._settings.vllm_tokenizer_mode,
                trust_remote_code=self._settings.vllm_trust_remote_code,
                tensor_parallel_size=self._settings.vllm_gpu_count,
                gpu_memory_utilization=self._effective_gpu_memory_utilization,
            )
            try:
                self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            except Exception as exc:  # pragma: no cover - depends on runtime installation
                raise NotReadyError(self._format_engine_init_error(exc)) from exc
            self._sampling_params_cls = SamplingParams
            return self._engine

    def _resolve_runtime_configuration(self) -> None:
        self._effective_cuda_visible_devices = self._settings.cuda_visible_devices
        self._effective_gpu_memory_utilization = self._settings.vllm_gpu_memory_utilization
        self._selected_gpus = []
        self._inspected_gpus = []

        if not self._enable_adaptive_gpu_selection or not self._settings.vllm_gpu_auto_select:
            return

        selection = self._select_startup_gpus()
        self._selected_gpus = selection.selected_gpus
        self._inspected_gpus = selection.inspected_gpus
        self._effective_cuda_visible_devices = selection.cuda_visible_devices
        self._effective_gpu_memory_utilization = selection.gpu_memory_utilization
        logger.info(
            "Adaptive GPU startup selected CUDA_VISIBLE_DEVICES=%s with gpu_memory_utilization=%.3f (%s). Inspected GPUs: %s",
            selection.cuda_visible_devices,
            selection.gpu_memory_utilization,
            "preferred-cap" if selection.used_preferred_utilization else "derived-from-free-memory",
            summarize_gpus(selection.inspected_gpus),
        )

    def _select_startup_gpus(self) -> GPUSelectionResult:
        try:
            inspected_gpus = self._gpu_query_fn()
        except Exception as exc:
            raise NotReadyError(
                "Adaptive GPU selection requires successful host GPU inspection via nvidia-smi before vLLM startup. "
                "Original error: %s" % exc
            ) from exc

        try:
            return build_selection_result(
                inspected_gpus=inspected_gpus,
                gpu_count=self._settings.vllm_gpu_count,
                preferred_utilization=self._settings.vllm_gpu_memory_utilization,
                reserve_fraction=self._settings.vllm_gpu_memory_reserve_fraction,
                minimum_utilization=self._settings.vllm_gpu_memory_utilization_min,
            )
        except ValueError as exc:
            raise NotReadyError(
                "Adaptive GPU selection could not choose %s GPUs for startup. %s. Inspected GPUs: %s"
                % (
                    self._settings.vllm_gpu_count,
                    exc,
                    summarize_gpus(inspected_gpus),
                )
            ) from exc

    def _apply_runtime_environment(self) -> None:
        self._settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self._settings.huggingface_hub_cache.mkdir(parents=True, exist_ok=True)
        self._settings.transformers_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(self._settings.huggingface_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self._settings.huggingface_hub_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(self._settings.transformers_cache)
        if self._effective_cuda_visible_devices is None:
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = self._effective_cuda_visible_devices

    def _format_engine_init_error(self, exc: Exception) -> str:
        detail = str(exc).strip() or exc.__class__.__name__
        prefix = (
            "vLLM failed to initialize model '%s' with VLLM_GPU_COUNT=%s, "
            "gpu_memory_utilization=%.3f, CUDA_VISIBLE_DEVICES=%s. "
            "This setting maps directly to vLLM tensor_parallel_size."
            % (
                self.model_id,
                self._settings.vllm_gpu_count,
                self._effective_gpu_memory_utilization,
                self._effective_cuda_visible_devices or "<unset>",
            )
        )
        if "not divisible by" in detail:
            return (
                "%s Choose a VLLM_GPU_COUNT value that evenly divides the model's tensor-parallel dimensions, "
                "or reduce it to a compatible value for the selected model. Original error: %s"
                % (prefix, detail)
            )
        return "%s Original error: %s" % (prefix, detail)
