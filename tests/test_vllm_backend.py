from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

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


def test_settings_reject_gpu_count_above_visible_devices(tmp_path):
    with pytest.raises(ValueError):
        make_settings(
            tmp_path,
            INFERENCE_BACKEND="vllm",
            CUDA_VISIBLE_DEVICES="0,1",
            VLLM_GPU_COUNT=3,
        )


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
