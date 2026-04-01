from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from llm_serve.types import InferenceRequest, InferenceResult


class ModelBackend(ABC):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    async def start(self) -> None:
        return None

    @abstractmethod
    async def shutdown(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResult:
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        raise NotImplementedError
