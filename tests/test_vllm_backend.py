from __future__ import annotations

import asyncio
import os
import signal
import sys
from types import ModuleType, SimpleNamespace

import pytest

from llm_serve.errors import BadRequestError, NotReadyError
from llm_serve.types import InferenceRequest
from llm_serve.runtime import vllm_backend as vllm_backend_module
from llm_serve.runtime.vllm_backend import VLLMModelBackend

from .conftest import make_settings


def test_settings_load_cuda_visible_devices_and_gpu_count(tmp_path):
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        CUDA_VISIBLE_DEVICES="0,2,5",
        VLLM_GPU_COUNT=2,
    )

    assert settings.model_cache_dir == (tmp_path / "models").resolve()
    assert settings.huggingface_home == (tmp_path / "models").resolve()
    assert settings.huggingface_hub_cache == (tmp_path / "models" / "hub").resolve()
    assert settings.transformers_cache == (tmp_path / "models" / "transformers").resolve()
    assert settings.cuda_visible_devices == "0,2,5"
    assert settings.cuda_visible_device_list == ["0", "2", "5"]
    assert settings.vllm_gpu_count == 2


def test_vllm_backend_applies_cuda_visible_devices_and_tensor_parallel_size(tmp_path, monkeypatch):
    captured = {}

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured["engine_args"] = kwargs

    class FakeEngine:
        def shutdown(self, timeout=None):
            captured["shutdown"] = True

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            captured["engine"] = args
            return FakeEngine()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            captured["sampling_params"] = kwargs

    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        MODEL_CACHE_DIR="repo-model-cache",
        VLLM_GPU_AUTO_SELECT="false",
        CUDA_VISIBLE_DEVICES="3,4",
        VLLM_GPU_COUNT=2,
    )

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=FakeAsyncEngineArgs,
            AsyncLLMEngine=FakeAsyncLLMEngine,
            SamplingParams=FakeSamplingParams,
        ),
    )

    backend = VLLMModelBackend("mock/reasoning", settings)
    asyncio.run(backend.start())

    assert settings.model_cache_dir.exists()
    assert settings.huggingface_hub_cache.exists()
    assert settings.transformers_cache.exists()
    assert os.environ["HF_HOME"] == str(settings.huggingface_home)
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(settings.huggingface_hub_cache)
    assert os.environ["TRANSFORMERS_CACHE"] == str(settings.transformers_cache)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,4"
    assert captured["engine_args"]["model"] == "mock/reasoning"
    assert captured["engine_args"]["tensor_parallel_size"] == 2
    assert captured["engine_args"]["gpu_memory_utilization"] == pytest.approx(0.9)
    assert captured["engine_args"]["disable_custom_all_reduce"] is True


def test_vllm_backend_auto_selects_best_gpus_and_derives_utilization(tmp_path, monkeypatch):
    captured = {}

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured["engine_args"] = kwargs

    class FakeEngine:
        def shutdown(self, timeout=None):
            captured["shutdown"] = True

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            captured["engine"] = args
            return FakeEngine()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            captured["sampling_params"] = kwargs

    def fake_gpu_query():
        return [
            _gpu(0, 10000, 7800),
            _gpu(1, 10000, 9500),
            _gpu(2, 10000, 8800),
        ]

    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        MODEL_CACHE_DIR="repo-model-cache",
        VLLM_GPU_AUTO_SELECT="true",
        CUDA_VISIBLE_DEVICES="0,1",
        VLLM_GPU_COUNT=2,
        VLLM_GPU_MEMORY_UTILIZATION=0.9,
        VLLM_GPU_MEMORY_RESERVE_FRACTION=0.05,
        VLLM_GPU_MEMORY_UTILIZATION_MIN=0.5,
    )

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=FakeAsyncEngineArgs,
            AsyncLLMEngine=FakeAsyncLLMEngine,
            SamplingParams=FakeSamplingParams,
        ),
    )

    backend = VLLMModelBackend(
        "mock/reasoning",
        settings,
        enable_adaptive_gpu_selection=True,
        gpu_query_fn=fake_gpu_query,
    )
    asyncio.run(backend.start())

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"
    assert captured["engine_args"]["gpu_memory_utilization"] == pytest.approx(0.83)
    assert captured["engine_args"]["tensor_parallel_size"] == 2


def test_vllm_backend_auto_select_uses_preferred_utilization_when_available(tmp_path, monkeypatch):
    captured = {}

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured["engine_args"] = kwargs

    class FakeEngine:
        def shutdown(self, timeout=None):
            return None

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            return FakeEngine()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_GPU_AUTO_SELECT="true",
        VLLM_GPU_COUNT=2,
        VLLM_GPU_MEMORY_UTILIZATION=0.9,
    )

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=FakeAsyncEngineArgs,
            AsyncLLMEngine=FakeAsyncLLMEngine,
            SamplingParams=FakeSamplingParams,
        ),
    )

    backend = VLLMModelBackend(
        "mock/reasoning",
        settings,
        enable_adaptive_gpu_selection=True,
        gpu_query_fn=lambda: [_gpu(0, 10000, 9500), _gpu(2, 10000, 9200), _gpu(3, 10000, 9100)],
    )
    asyncio.run(backend.start())

    assert captured["engine_args"]["gpu_memory_utilization"] == pytest.approx(0.9)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,2"


def test_vllm_backend_surfaces_actionable_parallelism_error(tmp_path, monkeypatch):
    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            raise ValueError("10240 is not divisible by 3")

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        MODEL_CACHE_DIR="repo-model-cache",
        VLLM_GPU_AUTO_SELECT="false",
        CUDA_VISIBLE_DEVICES="0,1,2",
        VLLM_GPU_COUNT=3,
    )

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=FakeAsyncEngineArgs,
            AsyncLLMEngine=FakeAsyncLLMEngine,
            SamplingParams=FakeSamplingParams,
        ),
    )

    backend = VLLMModelBackend("Qwen/Qwen3.5", settings)

    with pytest.raises(NotReadyError) as exc_info:
        asyncio.run(backend.start())

    message = str(exc_info.value)
    assert "VLLM_GPU_COUNT=3" in message
    assert "tensor_parallel_size" in message
    assert "Choose a VLLM_GPU_COUNT value that evenly divides" in message


def test_vllm_backend_reports_missing_nvidia_smi_for_adaptive_selection(tmp_path):
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_GPU_AUTO_SELECT="true",
        VLLM_GPU_COUNT=2,
    )

    backend = VLLMModelBackend(
        "mock/reasoning",
        settings,
        enable_adaptive_gpu_selection=True,
        gpu_query_fn=lambda: (_ for _ in ()).throw(RuntimeError("nvidia-smi is not available on the host")),
    )

    with pytest.raises(NotReadyError) as exc_info:
        asyncio.run(backend.start())

    assert "Adaptive GPU selection requires successful host GPU inspection via nvidia-smi" in str(exc_info.value)


def test_vllm_backend_reports_insufficient_gpu_memory_from_adaptive_selection(tmp_path):
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_GPU_AUTO_SELECT="true",
        VLLM_GPU_COUNT=2,
        VLLM_GPU_MEMORY_UTILIZATION=0.9,
        VLLM_GPU_MEMORY_UTILIZATION_MIN=0.5,
        VLLM_GPU_MEMORY_RESERVE_FRACTION=0.05,
    )

    backend = VLLMModelBackend(
        "mock/reasoning",
        settings,
        enable_adaptive_gpu_selection=True,
        gpu_query_fn=lambda: [_gpu(0, 10000, 4900), _gpu(1, 10000, 4800), _gpu(2, 10000, 2000)],
    )

    with pytest.raises(NotReadyError) as exc_info:
        asyncio.run(backend.start())

    message = str(exc_info.value)
    assert "Adaptive GPU selection could not choose 2 GPUs for startup" in message
    assert "Selected GPUs do not have enough free memory" in message


def test_vllm_backend_sets_vllm_use_v1_env_var(tmp_path, monkeypatch):
    captured = {}

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured["engine_args"] = kwargs

    class FakeEngine:
        def shutdown(self, timeout=None):
            return None

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            return FakeEngine()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.delenv("VLLM_USE_V1", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=FakeAsyncEngineArgs,
            AsyncLLMEngine=FakeAsyncLLMEngine,
            SamplingParams=FakeSamplingParams,
        ),
    )

    # Default (not set) uses V0 engine
    settings_default = make_settings(tmp_path, INFERENCE_BACKEND="vllm", VLLM_GPU_AUTO_SELECT="false")
    assert settings_default.vllm_use_v1 is False
    backend = VLLMModelBackend("mock/reasoning", settings_default)
    asyncio.run(backend.start())
    assert os.environ["VLLM_USE_V1"] == "0"

    # Explicit VLLM_USE_V1=true enables V1 engine
    settings_on = make_settings(tmp_path, INFERENCE_BACKEND="vllm", VLLM_GPU_AUTO_SELECT="false", VLLM_USE_V1="true")
    assert settings_on.vllm_use_v1 is True
    backend2 = VLLMModelBackend("mock/reasoning", settings_on)
    asyncio.run(backend2.start())
    assert os.environ["VLLM_USE_V1"] == "1"


def test_vllm_backend_renders_chat_template_and_extracts_reasoning(tmp_path, monkeypatch):
    captured = {}

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured["engine_args"] = kwargs

    class FakeEngine:
        async def generate(self, prompt, sampling_params, request_id):
            captured["prompt"] = prompt
            captured["request_id"] = request_id
            captured["sampling_params"] = sampling_params.kwargs
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="<think>\ntrace\n</think>\n\nFinal answer",
                        token_ids=[1, 2, 3, 4],
                    )
                ]
            )

        def shutdown(self, timeout=None):
            return None

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            return FakeEngine()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured["template_messages"] = messages
            captured["template_kwargs"] = kwargs
            return "templated-chat-prompt"

    transformers_module = ModuleType("transformers")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, trust_remote_code=False, cache_dir=None):
            captured["tokenizer_args"] = {
                "model_id": model_id,
                "trust_remote_code": trust_remote_code,
                "cache_dir": cache_dir,
            }
            return FakeTokenizer()

    transformers_module.AutoTokenizer = FakeAutoTokenizer

    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        MODEL_CACHE_DIR="repo-model-cache",
        VLLM_GPU_AUTO_SELECT="false",
        CUDA_VISIBLE_DEVICES="0,1",
        VLLM_GPU_COUNT=2,
        ENABLE_THINKING="true",
    )

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=FakeAsyncEngineArgs,
            AsyncLLMEngine=FakeAsyncLLMEngine,
            SamplingParams=FakeSamplingParams,
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    backend = VLLMModelBackend("Qwen/Qwen3.5-27B", settings)
    asyncio.run(backend.start())

    request = InferenceRequest(
        model_id="Qwen/Qwen3.5-27B",
        prompt="unused fallback prompt",
        max_output_tokens=64,
        temperature=0.2,
        top_p=0.95,
        stream=False,
        reasoning_effort="medium",
        include_reasoning=True,
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": [{"type": "text", "text": "Say hello."}]},
        ],
    )

    result = asyncio.run(backend.generate_chat(request))

    assert captured["template_messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Say hello."},
    ]
    assert captured["template_kwargs"]["enable_thinking"] is True
    assert captured["template_kwargs"]["tokenize"] is False
    assert captured["template_kwargs"]["add_generation_prompt"] is True
    assert captured["prompt"] == "templated-chat-prompt"
    assert result.text == "Final answer"
    assert result.reasoning == "trace"
    assert result.completion_tokens == 4


def test_vllm_backend_stream_hides_reasoning_content(tmp_path, monkeypatch):
    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeEngine:
        async def generate(self, prompt, sampling_params, request_id):
            yield SimpleNamespace(outputs=[SimpleNamespace(text="<th")])
            yield SimpleNamespace(outputs=[SimpleNamespace(text="<think>trace")])
            yield SimpleNamespace(outputs=[SimpleNamespace(text="<think>trace</think>\n\nFinal")])
            yield SimpleNamespace(outputs=[SimpleNamespace(text="<think>trace</think>\n\nFinal answer")])

        def shutdown(self, timeout=None):
            return None

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            return FakeEngine()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "templated-chat-prompt"

    transformers_module = ModuleType("transformers")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, trust_remote_code=False, cache_dir=None):
            return FakeTokenizer()

    transformers_module.AutoTokenizer = FakeAutoTokenizer

    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_GPU_AUTO_SELECT="false",
        CUDA_VISIBLE_DEVICES="0,1",
        VLLM_GPU_COUNT=2,
        ENABLE_THINKING="true",
    )

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=FakeAsyncEngineArgs,
            AsyncLLMEngine=FakeAsyncLLMEngine,
            SamplingParams=FakeSamplingParams,
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    backend = VLLMModelBackend("Qwen/Qwen3.5-27B", settings)
    asyncio.run(backend.start())
    request = InferenceRequest(
        model_id="Qwen/Qwen3.5-27B",
        prompt="unused fallback prompt",
        max_output_tokens=64,
        temperature=0.2,
        top_p=0.95,
        stream=True,
        reasoning_effort="high",
        include_reasoning=False,
        messages=[{"role": "user", "content": "Count to two."}],
    )

    async def collect():
        chunks = []
        async for chunk in backend.generate_chat_stream(request):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())
    assert "".join(chunks).strip() == "Final answer"
    assert "<think>" not in "".join(chunks)


def test_vllm_backend_rejects_reasoning_controls_without_enable_thinking_support(tmp_path, monkeypatch):
    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeEngine:
        def shutdown(self, timeout=None):
            return None

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            return FakeEngine()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            raise TypeError("apply_chat_template() got an unexpected keyword argument 'enable_thinking'")

    transformers_module = ModuleType("transformers")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, trust_remote_code=False, cache_dir=None):
            return FakeTokenizer()

    transformers_module.AutoTokenizer = FakeAutoTokenizer

    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_GPU_AUTO_SELECT="false",
        CUDA_VISIBLE_DEVICES="0,1",
        VLLM_GPU_COUNT=2,
        ENABLE_THINKING="true",
    )

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=FakeAsyncEngineArgs,
            AsyncLLMEngine=FakeAsyncLLMEngine,
            SamplingParams=FakeSamplingParams,
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    backend = VLLMModelBackend("mock/reasoning", settings)
    asyncio.run(backend.start())
    request = InferenceRequest(
        model_id="mock/reasoning",
        prompt="unused fallback prompt",
        max_output_tokens=64,
        temperature=0.2,
        top_p=0.95,
        stream=False,
        reasoning_effort="medium",
        include_reasoning=False,
        messages=[{"role": "user", "content": "Think carefully."}],
    )

    with pytest.raises(BadRequestError, match="does not support reasoning controls"):
        asyncio.run(backend.generate_chat(request))


def test_vllm_backend_thinking_disabled_ignores_reasoning_effort(tmp_path, monkeypatch):
    """When ENABLE_THINKING=false, enable_thinking is always False even with reasoning_effort set."""
    captured = {}

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured["engine_args"] = kwargs

    class FakeEngine:
        async def generate(self, prompt, sampling_params, request_id):
            captured["prompt"] = prompt
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="Normal answer",
                        token_ids=[1, 2],
                    )
                ]
            )

        def shutdown(self, timeout=None):
            return None

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            return FakeEngine()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured["template_kwargs"] = kwargs
            return "templated-prompt"

    transformers_module = ModuleType("transformers")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, trust_remote_code=False, cache_dir=None):
            return FakeTokenizer()

    transformers_module.AutoTokenizer = FakeAutoTokenizer

    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_GPU_AUTO_SELECT="false",
        CUDA_VISIBLE_DEVICES="0,1",
        VLLM_GPU_COUNT=2,
        ENABLE_THINKING="false",
    )

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=FakeAsyncEngineArgs,
            AsyncLLMEngine=FakeAsyncLLMEngine,
            SamplingParams=FakeSamplingParams,
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    backend = VLLMModelBackend("mock/reasoning", settings)
    asyncio.run(backend.start())

    request = InferenceRequest(
        model_id="mock/reasoning",
        prompt="unused",
        max_output_tokens=64,
        temperature=0.2,
        top_p=0.95,
        stream=False,
        reasoning_effort="medium",
        include_reasoning=False,
        messages=[{"role": "user", "content": "Think about it."}],
    )

    result = asyncio.run(backend.generate_chat(request))

    assert captured["template_kwargs"]["enable_thinking"] is False
    assert result.text == "Normal answer"


def test_vllm_backend_thinking_enabled_without_reasoning_effort_stays_off(tmp_path, monkeypatch):
    """When ENABLE_THINKING=true but reasoning_effort is not set, enable_thinking is False."""
    captured = {}

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured["engine_args"] = kwargs

    class FakeEngine:
        async def generate(self, prompt, sampling_params, request_id):
            captured["prompt"] = prompt
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="Normal answer",
                        token_ids=[1, 2],
                    )
                ]
            )

        def shutdown(self, timeout=None):
            return None

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            return FakeEngine()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured["template_kwargs"] = kwargs
            return "templated-prompt"

    transformers_module = ModuleType("transformers")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, trust_remote_code=False, cache_dir=None):
            return FakeTokenizer()

    transformers_module.AutoTokenizer = FakeAutoTokenizer

    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_GPU_AUTO_SELECT="false",
        CUDA_VISIBLE_DEVICES="0,1",
        VLLM_GPU_COUNT=2,
        ENABLE_THINKING="true",
    )

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=FakeAsyncEngineArgs,
            AsyncLLMEngine=FakeAsyncLLMEngine,
            SamplingParams=FakeSamplingParams,
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    backend = VLLMModelBackend("mock/reasoning", settings)
    asyncio.run(backend.start())

    request = InferenceRequest(
        model_id="mock/reasoning",
        prompt="unused",
        max_output_tokens=64,
        temperature=0.2,
        top_p=0.95,
        stream=False,
        messages=[{"role": "user", "content": "Hello."}],
    )

    result = asyncio.run(backend.generate_chat(request))

    assert captured["template_kwargs"]["enable_thinking"] is False
    assert result.text == "Normal answer"


def test_vllm_backend_start_failure_tracks_spawned_descendants_for_cleanup(tmp_path, monkeypatch):
    captured = {}

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            raise RuntimeError("engine init exploded")

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    child_snapshots = iter([{10}, {10, 11}])
    descendant_snapshots = iter([{10, 12}, {10, 12, 11, 13}, {13}])

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=FakeAsyncEngineArgs,
            AsyncLLMEngine=FakeAsyncLLMEngine,
            SamplingParams=FakeSamplingParams,
        ),
    )
    monkeypatch.setattr(vllm_backend_module, "_get_child_pids", lambda parent_pid: next(child_snapshots))
    monkeypatch.setattr(vllm_backend_module, "_get_descendant_pids", lambda root_pids: next(descendant_snapshots))

    def fake_cleanup(self):
        captured["roots"] = set(self._worker_root_pids)
        captured["pids"] = set(self._worker_pids)
        vllm_backend_module._tracked_worker_root_pids.clear()
        vllm_backend_module._tracked_worker_pids.clear()
        self._worker_root_pids.clear()
        self._worker_pids.clear()

    monkeypatch.setattr(VLLMModelBackend, "_cleanup_worker_processes", fake_cleanup)

    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_GPU_AUTO_SELECT="false",
        CUDA_VISIBLE_DEVICES="0,1",
        VLLM_GPU_COUNT=2,
    )

    backend = VLLMModelBackend("mock/reasoning", settings)

    with pytest.raises(NotReadyError, match="engine init exploded"):
        asyncio.run(backend.start())

    assert captured["roots"] == {11}
    assert captured["pids"] == {11, 13}


def test_vllm_backend_cleanup_includes_descendants_of_tracked_roots(tmp_path, monkeypatch):
    alive = {101, 102, 103}
    delivered_signals = []

    def fake_os_kill(pid, sig):
        if sig == 0:
            if pid not in alive:
                raise ProcessLookupError
            return None
        delivered_signals.append((pid, sig))
        if sig == signal.SIGKILL:
            alive.discard(pid)
        return None

    monkeypatch.setattr(vllm_backend_module, "_get_descendant_pids", lambda root_pids: {102, 103})
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr(os, "kill", fake_os_kill)

    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_GPU_AUTO_SELECT="false",
        CUDA_VISIBLE_DEVICES="0,1",
        VLLM_GPU_COUNT=2,
    )

    backend = VLLMModelBackend("mock/reasoning", settings)
    backend._worker_root_pids = {101}
    backend._worker_pids = {101, 102}
    vllm_backend_module._tracked_worker_root_pids.update({101})
    vllm_backend_module._tracked_worker_pids.update({101, 102})

    backend._cleanup_worker_processes()

    term_targets = {pid for pid, sig in delivered_signals if sig == signal.SIGTERM}
    kill_targets = {pid for pid, sig in delivered_signals if sig == signal.SIGKILL}
    assert term_targets == {101, 102, 103}
    assert kill_targets == {101, 102, 103}
    assert backend._worker_root_pids == set()
    assert backend._worker_pids == set()
    assert vllm_backend_module._tracked_worker_root_pids == set()
    assert vllm_backend_module._tracked_worker_pids == set()


def _gpu(index: int, total_mib: int, free_mib: int):
    used_mib = total_mib - free_mib
    return SimpleNamespace(
        index=index,
        uuid="GPU-%s" % index,
        name="GPU %s" % index,
        memory_total_mib=total_mib,
        memory_used_mib=used_mib,
        memory_free_mib=free_mib,
        free_ratio=free_mib / float(total_mib),
    )
