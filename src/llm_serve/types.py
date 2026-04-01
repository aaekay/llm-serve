from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceRequest:
    model_id: str
    prompt: str
    max_output_tokens: int
    temperature: float
    top_p: float
    stream: bool
    reasoning_effort: Optional[str] = None
    include_reasoning: bool = False


@dataclass
class InferenceResult:
    model_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    reasoning: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LoadStatus:
    state: str
    model_id: str
    current_model: Optional[str]
    retry_after_seconds: int = 2

    @property
    def ready(self) -> bool:
        return self.state == "ready"
