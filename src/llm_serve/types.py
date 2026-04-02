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
    upstream_timeout_seconds: Optional[float] = None
    timeout_retry_enabled: bool = True


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


@dataclass
class StartupSelfTestResult:
    status: str
    prompt: Optional[str] = None
    model_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    latency_seconds: Optional[float] = None
    completion_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    error: Optional[str] = None
