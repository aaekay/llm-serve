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
