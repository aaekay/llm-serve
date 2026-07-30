from __future__ import annotations

import asyncio

import pytest

from llm_serve.runtime.autotune import ManualVLLMBackend
from llm_serve.runtime.gpu_selection import GPUDeviceInfo
from llm_serve.runtime.throughput import WorkerTopology
from llm_serve.runtime.vllm_backend import VLLMModelBackend

from .conftest import make_settings


class FakePool:
    instances = []

    def __init__(self, model_id, settings, topology, *, optimized_engine=True, profile=None):
        self.model_id = model_id
        self.settings = settings
        self.topology = topology
        self.optimized_engine = optimized_engine
        self.profile = profile
        self.started = False
        self.stopped = False
        type(self).instances.append(self)

    @property
    def tuning_snapshot(self):
        return {
            "status": "ready",
            "profile_valid": False,
            "tensor_parallel_size": self.topology.tensor_parallel_size,
            "replica_count": self.topology.replica_count,
        }

    async def start(self):
        self.started = True

    async def shutdown(self):
        self.stopped = True


def _gpu(index: int, free_mib: int = 24000) -> GPUDeviceInfo:
    return GPUDeviceInfo(index, "GPU-%s" % index, "RTX", 24576, 24576 - free_mib, free_mib)


def test_legacy_backend_name_points_to_unified_manual_backend():
    assert VLLMModelBackend is ManualVLLMBackend


def test_manual_mode_builds_one_supervised_tensor_parallel_worker(tmp_path):
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_GPU_MODE="manual",
        VLLM_GPU_COUNT=2,
        VLLM_GPU_AUTO_SELECT="true",
    )
    FakePool.instances = []
    backend = ManualVLLMBackend(
        "model",
        settings,
        gpu_query_fn=lambda: [_gpu(0), _gpu(1), _gpu(2, 1000)],
        pool_factory=FakePool,
    )

    async def scenario():
        await backend.start()
        snapshot = backend.tuning_snapshot
        await backend.shutdown()
        return snapshot

    snapshot = asyncio.run(scenario())
    pool = FakePool.instances[0]

    assert pool.topology == WorkerTopology(2, [[0, 1]])
    assert pool.optimized_engine is False
    assert pool.started is True
    assert pool.stopped is True
    assert snapshot["mode"] == "manual"
    assert snapshot["tensor_parallel_size"] == 2
    assert snapshot["replica_count"] == 1


def test_manual_mode_cleans_up_failed_pool_start(tmp_path):
    settings = make_settings(
        tmp_path,
        INFERENCE_BACKEND="vllm",
        VLLM_GPU_MODE="manual",
        VLLM_GPU_COUNT=1,
    )

    class FailingPool(FakePool):
        async def start(self):
            raise RuntimeError("startup failed")

    FailingPool.instances = []
    backend = ManualVLLMBackend(
        "model",
        settings,
        gpu_query_fn=lambda: [_gpu(0)],
        pool_factory=FailingPool,
    )

    async def scenario():
        with pytest.raises(RuntimeError, match="startup failed"):
            await backend.start()

    asyncio.run(scenario())
    assert FailingPool.instances[0].stopped is True
