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
        CUDA_VISIBLE_DEVICES="3,4",
        VLLM_GPU_COUNT=2,
    )

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
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

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,4"
    assert captured["engine_args"]["model"] == "mock/reasoning"
    assert captured["engine_args"]["tensor_parallel_size"] == 2
