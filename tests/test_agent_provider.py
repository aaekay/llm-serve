from __future__ import annotations

from typing import AsyncIterator

from fastapi.testclient import TestClient

from llm_serve.app import create_app
from llm_serve.runtime.base import ModelBackend
from llm_serve.runtime.manager import RuntimeManager
from llm_serve.types import InferenceRequest, InferenceResult

from .conftest import make_settings


class AgentBackend(ModelBackend):
    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)
        self.chat_payloads = []
        self.response_payloads = []

    async def shutdown(self) -> None:
        return None

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(self.model_id, "ok", 1, 1)

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        yield "ok"

    async def openai_chat(self, payload):
        self.chat_payloads.append(payload)
        return {
            "id": "chatcmpl-agent",
            "object": "chat.completion",
            "created": 1,
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        }

    async def openai_chat_stream(self, payload):
        self.chat_payloads.append(payload)
        yield {
            "id": "chatcmpl-agent",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "id": "chatcmpl-agent",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"README.md"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        }

    async def create_response(self, payload):
        self.response_payloads.append(payload)
        return {
            "id": "resp-agent",
            "object": "response",
            "status": "completed",
            "model": self.model_id,
            "output": [
                {
                    "type": "function_call",
                    "id": "fc-1",
                    "call_id": "call-1",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                    "status": "completed",
                }
            ],
        }

    async def create_response_stream(self, payload):
        self.response_payloads.append(payload)
        yield {"type": "response.created", "response": {"id": "resp-agent", "status": "in_progress"}}
        yield {"type": "response.completed", "response": {"id": "resp-agent", "status": "completed"}}

    async def retrieve_response(self, response_id):
        return {"id": response_id, "object": "response", "status": "completed", "output": []}

    async def retrieve_response_stream(self, response_id, starting_after=None):
        yield {"type": "response.completed", "sequence_number": starting_after or 0}

    async def cancel_response(self, response_id):
        return {"id": response_id, "object": "response", "status": "cancelled", "output": []}

    async def wait_for_background(self):
        return None


def _agent_app(tmp_path):
    model = "Qwen/Qwen3.5-27B"
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        DEFAULT_MODEL_ID=model,
        VLLM_DEFAULT_MODEL_ID=model,
        MODEL_ALLOWLIST=model,
        REASONING_MODEL_ALLOWLIST=model,
        STARTUP_SELF_TEST_ENABLED="false",
        VLLM_ENABLE_AUTO_TOOL_CHOICE="true",
        VLLM_TOOL_CALL_PARSERS='{"%s":"qwen3_coder"}' % model,
        VLLM_REASONING_PARSERS='{"%s":"qwen3"}' % model,
        VLLM_RESPONSES_STORE_ENABLED="true",
    )
    backend = AgentBackend(model)

    async def backend_factory(_model_id):
        return backend

    runtime = RuntimeManager(settings, backend_factory=backend_factory)
    return create_app(settings, runtime_manager=runtime), backend


def _tool():
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def test_vllm_chat_preserves_agent_fields_and_tool_response(tmp_path):
    app, backend = _agent_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "developer", "content": "Use tools."},
                    {"role": "user", "content": "Read the README."},
                ],
                "tools": [_tool()],
                "tool_choice": "required",
                "parallel_tool_calls": True,
                "custom_vllm_field": "preserved",
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert response.json()["choices"][0]["message"]["tool_calls"][0]["id"] == "call-1"
    assert backend.chat_payloads[0]["custom_vllm_field"] == "preserved"
    assert backend.chat_payloads[0]["tools"][0]["function"]["strict"] is True


def test_vllm_chat_stream_preserves_tool_argument_deltas(tmp_path):
    app, _ = _agent_app(tmp_path)
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Read."}], "tools": [_tool()], "stream": True},
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"finish_reason": "tool_calls"' in body
    assert '"arguments": "{\\"path\\":"' in body
    assert "data: [DONE]" in body


def test_responses_create_stream_retrieve_and_cancel(tmp_path):
    app, backend = _agent_app(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/v1/responses",
            json={
                "input": "Read README",
                "tools": [
                    {
                        "type": "function",
                        "name": "read_file",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
                "future_field": "preserved",
            },
        )
        retrieved = client.get("/v1/responses/resp-agent")
        cancelled = client.post("/v1/responses/resp-agent/cancel")
        with client.stream(
            "POST",
            "/v1/responses",
            json={"input": "hello", "stream": True},
        ) as stream:
            stream_body = "".join(stream.iter_text())

    assert created.status_code == 200
    assert created.json()["output"][0]["type"] == "function_call"
    assert backend.response_payloads[0]["future_field"] == "preserved"
    assert retrieved.json()["status"] == "completed"
    assert cancelled.json()["status"] == "cancelled"
    assert "response.created" in stream_body
    assert "response.completed" in stream_body
    assert "[DONE]" not in stream_body
