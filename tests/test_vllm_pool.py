from __future__ import annotations

import asyncio
from types import SimpleNamespace

from llm_serve.runtime.throughput import WorkerTopology
from llm_serve.runtime import vllm_pool
from llm_serve.runtime.vllm_pool import VLLMPoolBackend, VLLMWorker
from llm_serve.types import InferenceRequest

from .conftest import make_settings


class FakeWorker:
    def __init__(self, index, model_id, gpu_indices, tensor_parallel_size, settings, *, optimized_engine):
        self.index = index
        self.model_id = model_id
        self.gpu_indices = list(gpu_indices)
        self.tensor_parallel_size = tensor_parallel_size
        self.healthy = False
        self.outstanding_tokens = 0
        self.calls = []

    async def start(self):
        self.healthy = True

    async def stop(self, **kwargs):
        self.healthy = False

    def exited(self):
        return False

    async def request_json(self, path, payload):
        self.calls.append((path, payload))
        return {
            "choices": [{"text": "worker-%s" % self.index}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }

    async def stream_json(self, path, payload):
        self.calls.append((path, payload))
        yield {"choices": [{"text": "hello "}]}
        yield {"choices": [{"text": "world"}]}


def test_worker_command_avoids_removed_disable_log_requests_flag(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, INFERENCE_BACKEND="vllm")
    monkeypatch.setattr(vllm_pool, "_vllm_executable", lambda: "/opt/bin/vllm")
    monkeypatch.setattr(vllm_pool, "_unused_loopback_port", lambda: 18000)
    worker = VLLMWorker(0, "model", [0, 1], 2, settings, optimized_engine=True)

    async def scenario():
        command = worker.command()
        await worker.stop()
        return command

    command = asyncio.run(scenario())

    assert command[:3] == ["/opt/bin/vllm", "serve", "model"]
    assert "--disable-log-requests" not in command


def test_worker_command_enables_model_specific_agent_parsers(tmp_path, monkeypatch):
    model = "Qwen/Qwen3.5-27B"
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        DEFAULT_MODEL_ID=model,
        VLLM_DEFAULT_MODEL_ID=model,
        MODEL_ALLOWLIST=model,
        REASONING_MODEL_ALLOWLIST=model,
        VLLM_ENABLE_AUTO_TOOL_CHOICE="true",
        VLLM_TOOL_CALL_PARSERS='{"%s":"qwen3_coder"}' % model,
        VLLM_REASONING_PARSERS='{"%s":"qwen3"}' % model,
        VLLM_RESPONSES_STORE_ENABLED="true",
    )
    monkeypatch.setattr(vllm_pool, "_vllm_executable", lambda: "/opt/bin/vllm")
    monkeypatch.setattr(vllm_pool, "_ninja_executable", lambda: "/opt/ninja/bin/ninja")
    monkeypatch.setattr(vllm_pool, "_unused_loopback_port", lambda: 18000)
    worker = VLLMWorker(0, model, [0], 1, settings, optimized_engine=True)

    async def scenario():
        command = worker.command()
        await worker.stop()
        return command

    command = asyncio.run(scenario())

    assert "--enable-auto-tool-choice" in command
    assert command[command.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"


def test_pool_routes_to_worker_with_least_outstanding_token_load(tmp_path):
    settings = make_settings(tmp_path, INFERENCE_BACKEND="vllm")
    pool = VLLMPoolBackend(
        "model",
        settings,
        WorkerTopology(2, [[0, 1], [2, 3]]),
        worker_factory=FakeWorker,
    )
    request = InferenceRequest("model", "hello", 10, 0.0, 1.0, False)

    async def scenario():
        await pool.start()
        pool.workers[0].outstanding_tokens = 100
        result = await pool.generate(request)
        await pool.shutdown()
        return result

    result = asyncio.run(scenario())

    assert result.text == "worker-1"
    assert pool.workers[0].calls == []
    assert pool.workers[1].calls[0][0] == "/v1/completions"
    assert pool.workers[1].outstanding_tokens == 0


def test_pool_stream_releases_worker_load(tmp_path):
    settings = make_settings(tmp_path, INFERENCE_BACKEND="vllm")
    pool = VLLMPoolBackend(
        "model",
        settings,
        WorkerTopology(1, [[0]]),
        worker_factory=FakeWorker,
    )
    request = InferenceRequest("model", "hello", 10, 0.0, 1.0, True)

    async def scenario():
        await pool.start()
        chunks = [chunk async for chunk in pool.generate_stream(request)]
        outstanding = pool.workers[0].outstanding_tokens
        await pool.shutdown()
        return chunks, outstanding

    chunks, outstanding = asyncio.run(scenario())

    assert "".join(chunks) == "hello world"
    assert outstanding == 0


def test_pool_preserves_raw_tool_calls_and_response_worker_affinity(tmp_path):
    model = "Qwen/Qwen3.5-27B"
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        DEFAULT_MODEL_ID=model,
        VLLM_DEFAULT_MODEL_ID=model,
        MODEL_ALLOWLIST=model,
        REASONING_MODEL_ALLOWLIST=model,
        VLLM_ENABLE_AUTO_TOOL_CHOICE="true",
        VLLM_TOOL_CALL_PARSERS='{"%s":"qwen3_coder"}' % model,
    )

    class AgentWorker(FakeWorker):
        async def request_json(self, path, payload, *, method="POST"):
            self.calls.append((method, path, payload))
            if path == "/v1/chat/completions":
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {"name": "read", "arguments": "{}"},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            if method == "GET":
                return {"id": path.split("/")[-1], "status": "completed", "output": []}
            if path.endswith("/cancel"):
                return {"id": path.split("/")[-2], "status": "cancelled", "output": []}
            return {
                "id": payload["request_id"],
                "status": "completed",
                "output": [],
            }

    pool = VLLMPoolBackend(
        model,
        settings,
        WorkerTopology(1, [[0], [1]]),
        worker_factory=AgentWorker,
    )
    tool = {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}

    async def scenario():
        await pool.start()
        pool.workers[0].outstanding_tokens = 100
        chat = await pool.openai_chat({"model": model, "messages": [], "tools": [tool]})
        first = await pool.create_response({"model": model, "input": "hello", "tools": [tool]})
        second = await pool.create_response(
            {"model": model, "input": "continue", "previous_response_id": first["id"]}
        )
        retrieved = await pool.retrieve_response(first["id"])
        cancelled = await pool.cancel_response(first["id"])
        await pool.shutdown()
        return chat, first, second, retrieved, cancelled

    chat, first, second, retrieved, cancelled = asyncio.run(scenario())

    assert chat["choices"][0]["message"]["tool_calls"][0]["id"] == "call-1"
    assert first["id"].startswith("resp_")
    assert second["id"].startswith("resp_")
    owner = pool._response_workers[first["id"]]
    owner_calls = pool.workers[owner].calls
    assert any(call[2] and call[2].get("previous_response_id") == first["id"] for call in owner_calls)
    assert retrieved["status"] == "completed"
    assert cancelled["status"] == "cancelled"


def test_pool_preserves_per_request_thinking_controls(tmp_path):
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        ENABLE_THINKING="true",
    )

    class ChatWorker(FakeWorker):
        async def request_json(self, path, payload):
            self.calls.append((path, payload))
            return {
                "choices": [
                    {
                        "message": {
                            "content": "<think>trace</think>answer",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3},
            }

    pool = VLLMPoolBackend(
        "model",
        settings,
        WorkerTopology(1, [[0]]),
        worker_factory=ChatWorker,
    )
    request = InferenceRequest(
        "model",
        "solve",
        10,
        0.0,
        1.0,
        False,
        reasoning_effort="medium",
    )

    async def scenario():
        await pool.start()
        result = await pool.generate(request)
        call = pool.workers[0].calls[0]
        await pool.shutdown()
        return result, call

    result, (path, payload) = asyncio.run(scenario())

    assert path == "/v1/chat/completions"
    assert payload["messages"] == [{"role": "user", "content": "solve"}]
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert result.reasoning == "trace"
    assert result.text == "answer"

    disabled_settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        ENABLE_THINKING="false",
    )
    disabled_pool = VLLMPoolBackend(
        "model",
        disabled_settings,
        WorkerTopology(1, [[0]]),
        worker_factory=ChatWorker,
    )

    async def disabled_scenario():
        await disabled_pool.start()
        await disabled_pool.generate(request)
        call = disabled_pool.workers[0].calls[0]
        await disabled_pool.shutdown()
        return call

    disabled_path, disabled_payload = asyncio.run(disabled_scenario())

    assert disabled_path == "/v1/completions"
    assert "chat_template_kwargs" not in disabled_payload


def test_worker_passes_runtime_and_cache_environment(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setenv("TQDM_DISABLE", "1")
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_USE_V1="true",
    )
    monkeypatch.setattr(vllm_pool, "_vllm_executable", lambda: "/opt/bin/vllm")
    monkeypatch.setattr(vllm_pool, "_unused_loopback_port", lambda: 18000)

    async def process_factory(*command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        captured["stdout"] = kwargs["stdout"]
        captured["stderr"] = kwargs["stderr"]
        return SimpleNamespace(pid=100, returncode=None)

    class HealthyClient:
        async def get(self, path, timeout):
            return SimpleNamespace(status_code=200)

        async def aclose(self):
            return None

    worker = VLLMWorker(
        0,
        "model",
        [2, 3],
        2,
        settings,
        optimized_engine=False,
        process_factory=process_factory,
    )
    worker._client = HealthyClient()

    async def scenario():
        await worker.start()
        worker.process.returncode = 0
        await worker.stop()

    asyncio.run(scenario())

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert captured["env"]["HF_HOME"] == str(settings.huggingface_home)
    assert captured["env"]["HUGGINGFACE_HUB_CACHE"] == str(settings.huggingface_hub_cache)
    assert captured["env"]["TRANSFORMERS_CACHE"] == str(settings.transformers_cache)
    assert captured["env"]["VLLM_USE_V1"] == "1"
    assert captured["env"]["VLLM_SKIP_GDN_WARMUP"] == "1"
    assert captured["env"]["PATH"].split(":", 1)[0] == "/opt/ninja/bin"
    assert captured["env"]["HF_HUB_DISABLE_PROGRESS_BARS"] == "0"
    assert captured["env"]["HF_HUB_VERBOSITY"] == "info"
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert "TQDM_DISABLE" not in captured["env"]
    assert captured["stdout"] is None
    assert captured["stderr"] is None


def test_ninja_preflight_has_actionable_error(monkeypatch):
    monkeypatch.setattr(vllm_pool.sys, "executable", "/missing/venv/bin/python")
    monkeypatch.setattr(vllm_pool.shutil, "which", lambda name: None)

    try:
        vllm_pool._ninja_executable()
    except Exception as exc:
        assert "Ninja build tool" in str(exc)
        assert "uv sync --extra runtime" in str(exc)
    else:
        raise AssertionError("missing Ninja should fail before vLLM workers start")
