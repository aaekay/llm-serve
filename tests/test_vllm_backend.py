from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

from llm_serve.errors import NotReadyError
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
        def shutdown_background_loop(self):
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


def test_vllm_backend_auto_selects_best_gpus_and_derives_utilization(tmp_path, monkeypatch):
    captured = {}

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured["engine_args"] = kwargs

    class FakeEngine:
        def shutdown_background_loop(self):
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
        def shutdown_background_loop(self):
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
