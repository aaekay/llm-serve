from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: Dict[str, Any]


ContentPart = Union[TextContentPart, ImageContentPart]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentPart]]


class OpenAIChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, gt=0, le=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    include_reasoning: bool = False

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: List[ChatMessage]) -> List[ChatMessage]:
        if not value:
            raise ValueError("messages must not be empty")
        return value

    def resolved_max_tokens(self, default_value: int) -> int:
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return default_value


class OllamaOptions(BaseModel):
    num_predict: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, gt=0, le=1)
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    include_reasoning: Optional[bool] = None


class OllamaChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    options: Optional[OllamaOptions] = None

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: List[ChatMessage]) -> List[ChatMessage]:
        if not value:
            raise ValueError("messages must not be empty")
        return value


class OllamaGenerateRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    stream: bool = False
    options: Optional[OllamaOptions] = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value


class PullRequest(BaseModel):
    name: str


class BatchCreateRequest(BaseModel):
    input_file_id: str
    endpoint: str
    completion_window: str


class BatchInputLine(BaseModel):
    custom_id: str
    method: str
    url: str
    body: Dict[str, Any]


class FileRecord(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str = "processed"


class BatchRequestCounts(BaseModel):
    total: int = 0
    completed: int = 0
    failed: int = 0


class BatchRecord(BaseModel):
    id: str
    object: str = "batch"
    input_file_id: str
    endpoint: str
    completion_window: str
    created_at: int
    status: str = "queued"
    request_counts: BatchRequestCounts = Field(default_factory=BatchRequestCounts)
    in_progress_at: Optional[int] = None
    completed_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
