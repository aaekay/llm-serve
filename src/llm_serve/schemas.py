from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: Dict[str, Any]


ContentPart = Union[TextContentPart, ImageContentPart]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: Optional[Union[str, List[ContentPart]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    @model_validator(mode="after")
    def validate_agent_message(self) -> "ChatMessage":
        if self.role == "assistant" and (self.content is not None or self.tool_calls):
            return self
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool messages require tool_call_id")
            if self.content is None:
                raise ValueError("tool messages require content")
            return self
        if self.content is None:
            raise ValueError("content is required unless the message contains tool calls")
        return self


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    messages: List[ChatMessage]
    model: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, gt=0, le=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    reasoning_effort: Optional[Literal["low", "medium", "high", "xhigh"]] = None
    include_reasoning: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    stream_options: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    n: int = Field(default=1, ge=1)

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

    def upstream_payload(self, model_id: str, default_max_tokens: int) -> Dict[str, Any]:
        payload = self.model_dump(exclude_none=True)
        payload["model"] = model_id
        payload["max_tokens"] = min(self.resolved_max_tokens(default_max_tokens), default_max_tokens)
        payload.pop("max_completion_tokens", None)
        payload.pop("include_reasoning", None)
        payload.pop("reasoning_effort", None)
        return payload


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
