from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import AsyncIterator, Callable, List, Optional

from llm_serve.config import Settings
from llm_serve.errors import BadRequestError, NotReadyError, UpstreamRuntimeError
from llm_serve.prompting import (
    ThinkingContentStripper,
    normalize_messages_for_chat_template,
    split_reasoning_output,
)
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
        self._tokenizer = None
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
        shutdown_fn = getattr(engine, "shutdown", None)
        if callable(shutdown_fn):
            try:
                await asyncio.to_thread(shutdown_fn, 180.0)
            except Exception as exc:
                logger.warning("vLLM engine shutdown failed: %s", exc)
            return
        shutdown_background_loop = getattr(engine, "shutdown_background_loop", None)
        if callable(shutdown_background_loop):
            shutdown_background_loop()

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        prompt = self._resolve_prompt(request, use_messages=False)
        return await self._generate_from_prompt(request, prompt)

    async def generate_chat(self, request: InferenceRequest) -> InferenceResult:
        prompt = self._resolve_prompt(request, use_messages=True)
        return await self._generate_from_prompt(request, prompt)

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        prompt = self._resolve_prompt(request, use_messages=False)
        async for chunk in self._generate_stream_from_prompt(request, prompt):
            yield chunk

    async def generate_chat_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        prompt = self._resolve_prompt(request, use_messages=True)
        async for chunk in self._generate_stream_from_prompt(request, prompt):
            yield chunk

    async def _generate_from_prompt(self, request: InferenceRequest, prompt: str) -> InferenceResult:
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
            async for output in engine.generate(prompt, sampling_params, request_id):
                if not output.outputs:
                    continue
                final_output = output.outputs[0]
                final_text = final_output.text
                token_ids = getattr(final_output, "token_ids", None)
                if token_ids is not None:
                    completion_tokens = len(token_ids)
        except Exception as exc:  # pragma: no cover
            raise UpstreamRuntimeError("vLLM generation failed: %s" % exc) from exc

        reasoning, answer = split_reasoning_output(final_text)
        return InferenceResult(
            model_id=self.model_id,
            text=answer,
            prompt_tokens=estimate_text_tokens(prompt),
            completion_tokens=completion_tokens or estimate_text_tokens(final_text),
            reasoning=reasoning,
        )

    async def _generate_stream_from_prompt(self, request: InferenceRequest, prompt: str) -> AsyncIterator[str]:
        engine = await self._ensure_engine()
        sampling_params = self._sampling_params_cls(
            max_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        request_id = uuid.uuid4().hex
        previous_text = ""
        stripper = ThinkingContentStripper()

        try:
            async for output in engine.generate(prompt, sampling_params, request_id):
                if not output.outputs:
                    continue
                current_text = output.outputs[0].text
                delta = current_text[len(previous_text):]
                previous_text = current_text
                for visible_chunk in stripper.push(delta):
                    if visible_chunk:
                        yield visible_chunk
        except Exception as exc:  # pragma: no cover
            raise UpstreamRuntimeError("vLLM generation failed: %s" % exc) from exc

        tail = stripper.finish()
        if tail:
            yield tail

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
            except ImportError as exc:  # pragma: no cover
                raise NotReadyError(
                    "vLLM is not installed. Install the runtime dependencies before using INFERENCE_BACKEND=vllm."
                ) from exc

            engine_args = AsyncEngineArgs(
                model=self.model_id,
                tokenizer=self.model_id,
                dtype=self._settings.vllm_dtype,
                tokenizer_mode=self._settings.vllm_tokenizer_mode,
                trust_remote_code=self._settings.vllm_trust_remote_code,
                enforce_eager=self._settings.vllm_enforce_eager,
                tensor_parallel_size=self._settings.vllm_gpu_count,
                gpu_memory_utilization=self._effective_gpu_memory_utilization,
                max_model_len=self._settings.vllm_max_model_len,
                language_model_only=self._settings.vllm_language_model_only,
                skip_mm_profiling=self._settings.vllm_language_model_only,
                gdn_prefill_backend=self._settings.vllm_gdn_prefill_backend,
                disable_custom_all_reduce=self._settings.vllm_disable_custom_all_reduce,
            )
            try:
                self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            except Exception as exc:  # pragma: no cover
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
        os.environ["VLLM_USE_V1"] = "1" if self._settings.vllm_use_v1 else "0"
        os.environ.setdefault("VLLM_SKIP_GDN_WARMUP", "1")
        if self._effective_cuda_visible_devices is None:
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = self._effective_cuda_visible_devices

    def _resolve_prompt(self, request: InferenceRequest, use_messages: bool) -> str:
        enable_thinking = self._settings.enable_thinking and bool(request.reasoning_effort)
        if use_messages:
            if not request.messages:
                raise BadRequestError("Chat requests must include messages")
            return self._render_chat_prompt(request.messages, enable_thinking=enable_thinking)
        if enable_thinking:
            return self._render_chat_prompt(
                [{"role": "user", "content": request.prompt}],
                enable_thinking=True,
            )
        return request.prompt

    def _render_chat_prompt(self, messages, enable_thinking: bool) -> str:
        tokenizer = self._ensure_tokenizer()
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise NotReadyError(
                "Tokenizer for model '%s' does not support chat templates required by the vLLM backend."
                % self.model_id
            )

        normalized_messages = normalize_messages_for_chat_template(messages)
        base_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            return apply_chat_template(
                normalized_messages,
                enable_thinking=enable_thinking,
                **base_kwargs,
            )
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise UpstreamRuntimeError("vLLM chat template rendering failed: %s" % exc) from exc
            if enable_thinking:
                raise BadRequestError(
                    "Model '%s' does not support reasoning controls through the vLLM backend."
                    % self.model_id
                ) from exc
            try:
                return apply_chat_template(normalized_messages, **base_kwargs)
            except Exception as fallback_exc:
                raise UpstreamRuntimeError("vLLM chat template rendering failed: %s" % fallback_exc) from fallback_exc
        except Exception as exc:
            raise UpstreamRuntimeError("vLLM chat template rendering failed: %s" % exc) from exc

    def _ensure_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise NotReadyError(
                "transformers is not installed. Install the runtime dependencies before using chat requests with INFERENCE_BACKEND=vllm."
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self._settings.vllm_trust_remote_code,
                cache_dir=str(self._settings.model_cache_dir),
            )
        except Exception as exc:  # pragma: no cover
            raise NotReadyError("Failed to load tokenizer for model '%s': %s" % (self.model_id, exc)) from exc
        return self._tokenizer

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
