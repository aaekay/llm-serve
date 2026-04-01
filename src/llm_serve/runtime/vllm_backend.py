from __future__ import annotations

import asyncio
import os
import uuid
from typing import AsyncIterator, Optional

from llm_serve.config import Settings
from llm_serve.errors import NotReadyError, UpstreamRuntimeError
from llm_serve.tokenization import estimate_text_tokens
from llm_serve.types import InferenceRequest, InferenceResult

from .base import ModelBackend


class VLLMModelBackend(ModelBackend):
    def __init__(self, model_id: str, settings: Settings) -> None:
        super().__init__(model_id)
        self._settings = settings
        self._engine = None
        self._sampling_params_cls = None
        self._engine_lock = asyncio.Lock()

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
        final_text = ""
        async for delta in self.generate_stream(request):
            final_text += delta
        return InferenceResult(
            model_id=self.model_id,
            text=final_text,
            prompt_tokens=estimate_text_tokens(request.prompt),
            completion_tokens=estimate_text_tokens(final_text),
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
                gpu_memory_utilization=self._settings.vllm_gpu_memory_utilization,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._sampling_params_cls = SamplingParams
            return self._engine

    def _apply_runtime_environment(self) -> None:
        self._settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self._settings.huggingface_hub_cache.mkdir(parents=True, exist_ok=True)
        self._settings.transformers_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(self._settings.huggingface_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self._settings.huggingface_hub_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(self._settings.transformers_cache)
        if self._settings.cuda_visible_devices is None:
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = self._settings.cuda_visible_devices
