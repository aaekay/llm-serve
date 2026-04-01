from __future__ import annotations

import asyncio
from typing import AsyncIterator

from llm_serve.config import Settings
from llm_serve.tokenization import estimate_text_tokens
from llm_serve.types import InferenceRequest, InferenceResult

from .base import ModelBackend


class MockModelBackend(ModelBackend):
    def __init__(self, model_id: str, settings: Settings) -> None:
        super().__init__(model_id)
        self._delay = settings.mock_response_delay_seconds

    async def shutdown(self) -> None:
        return None

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        text = self._build_text(request)
        return InferenceResult(
            model_id=self.model_id,
            text=text,
            prompt_tokens=estimate_text_tokens(request.prompt),
            completion_tokens=estimate_text_tokens(text),
            reasoning=self._build_reasoning(request),
        )

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        text = self._build_text(request)
        chunk_size = 12
        index = 0
        while index < len(text):
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            yield text[index : index + chunk_size]
            index += chunk_size

    def _build_text(self, request: InferenceRequest) -> str:
        prompt = " ".join(request.prompt.strip().split())
        if len(prompt) > 140:
            prompt = prompt[:137] + "..."
        suffix = ""
        if request.reasoning_effort:
            suffix = " reasoning=%s" % request.reasoning_effort
        return "mock-response[%s]%s: %s" % (self.model_id, suffix, prompt)

    def _build_reasoning(self, request: InferenceRequest) -> str:
        if not request.include_reasoning:
            return None
        if not request.reasoning_effort:
            return None
        return "Mock reasoning trace for %s at %s effort." % (self.model_id, request.reasoning_effort)
