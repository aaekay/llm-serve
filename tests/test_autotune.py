from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from llm_serve.errors import NotReadyError
from llm_serve.runtime import autotune
from llm_serve.runtime.autotune import AutoTunedVLLMBackend, FixedTopologyVLLMBackend
from llm_serve.runtime.gpu_selection import GPUDeviceInfo
from llm_serve.runtime.throughput import (
    BenchmarkMeasurement,
    WorkerTopology,
    build_profile_fingerprint,
    load_workload,
    new_profile,
    save_tuning_profile,
    tuning_profile_path,
)

from .conftest import make_settings


class FakePool:
    starts = 0
    instances = []

    def __init__(self, model_id, settings, topology, *, optimized_engine=True, profile=None):
        self.model_id = model_id
        self.settings = settings
        self.topology = topology
        self.optimized_engine = optimized_engine
        self.profile = profile
        type(self).instances.append(self)

    @property
    def tuning_snapshot(self):
        return {
            "status": "ready",
            "profile_valid": self.profile is not None,
            "tensor_parallel_size": self.topology.tensor_parallel_size,
            "replica_count": self.topology.replica_count,
        }

    async def start(self):
        type(self).starts += 1

    async def shutdown(self):
        return None


def _gpu(index: int) -> GPUDeviceInfo:
    return GPUDeviceInfo(index, "GPU-%s" % index, "RTX", 24576, 512, 24064)


def test_gemma4_unified_rejects_vllm_without_native_support(monkeypatch):
    monkeypatch.setattr(autotune.importlib.metadata, "version", lambda package: "0.19.0")

    with pytest.raises(NotReadyError, match="vLLM>=0.23.0"):
        autotune._validate_vllm_version("google/gemma-4-12B-it")


def test_auto_backend_reuses_matching_profile_without_retuning(tmp_path, monkeypatch):
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_GPU_MODE="auto",
        VLLM_GPU_COUNT=4,
        DEFAULT_MODEL_ID="google/gemma-4-12B-it",
        MODEL_ALLOWLIST="google/gemma-4-12B-it",
        REASONING_MODEL_ALLOWLIST="",
    )
    gpus = [_gpu(index) for index in range(4)]
    workload = load_workload(None)
    monkeypatch.setattr(autotune.importlib.metadata, "version", lambda package: "0.23.0")
    fingerprint = build_profile_fingerprint(
        settings,
        settings.effective_default_model_id,
        "revision",
        gpus,
        workload,
        {},
    )
    measurement = BenchmarkMeasurement(
        WorkerTopology(2, [[0, 1], [2, 3]]),
        16,
        900,
        8,
        {"short": 1.5, "medium": 2.0, "long": 3.0},
        {"short": 0.1, "medium": 0.1, "long": 0.2},
    )
    profile = new_profile(
        fingerprint,
        settings.effective_default_model_id,
        measurement,
        1.5,
        workload,
        [measurement],
    )
    save_tuning_profile(tuning_profile_path(settings, settings.effective_default_model_id), profile)
    monkeypatch.setattr(
        autotune,
        "_load_model_config",
        lambda model_id, current_settings: (
            SimpleNamespace(text_config=SimpleNamespace(num_attention_heads=16, num_key_value_heads=8)),
            "revision",
        ),
    )
    FakePool.starts = 0
    backend = AutoTunedVLLMBackend(
        settings.effective_default_model_id,
        settings,
        gpu_query_fn=lambda: gpus,
        topology_query_fn=lambda: {},
        pool_factory=FakePool,
    )

    async def scenario():
        await backend.start()
        snapshot = backend.tuning_snapshot
        await backend.shutdown()
        return snapshot

    snapshot = asyncio.run(scenario())

    assert FakePool.starts == 1
    assert snapshot["profile_valid"] is True
    assert snapshot["tensor_parallel_size"] == 2


def test_fixed_backend_starts_requested_tp1_r4_topology(tmp_path, monkeypatch):
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_GPU_MODE="fixed",
        VLLM_GPU_COUNT=4,
        VLLM_TENSOR_PARALLEL_SIZE=1,
        VLLM_REPLICA_COUNT=4,
        DEFAULT_MODEL_ID="google/gemma-4-12B-it",
        MODEL_ALLOWLIST="google/gemma-4-12B-it",
        REASONING_MODEL_ALLOWLIST="",
    )
    gpus = [_gpu(index) for index in range(4)]
    monkeypatch.setattr(autotune.importlib.metadata, "version", lambda package: "0.23.0")
    FakePool.starts = 0
    FakePool.instances = []
    backend = FixedTopologyVLLMBackend(
        settings.effective_default_model_id,
        settings,
        gpu_query_fn=lambda: gpus,
        topology_query_fn=lambda: {},
        pool_factory=FakePool,
    )

    async def scenario():
        await backend.start()
        snapshot = backend.tuning_snapshot
        await backend.shutdown()
        return snapshot

    snapshot = asyncio.run(scenario())

    assert FakePool.starts == 1
    assert FakePool.instances[0].topology == WorkerTopology(1, [[0], [1], [2], [3]])
    assert snapshot["mode"] == "fixed"
    assert snapshot["profile_valid"] is False
    assert snapshot["tensor_parallel_size"] == 1
    assert snapshot["replica_count"] == 4
