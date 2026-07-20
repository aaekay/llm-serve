from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_serve.runtime.gpu_selection import GPUDeviceInfo, group_gpus_for_tensor_parallel
from llm_serve.runtime.throughput import (
    BenchmarkMeasurement,
    WorkerTopology,
    build_profile_fingerprint,
    load_tuning_profile,
    load_workload,
    new_profile,
    plan_topologies,
    save_tuning_profile,
    select_best_measurement,
)

from .conftest import make_settings


def _gpu(index: int, free_mib: int = 24000) -> GPUDeviceInfo:
    return GPUDeviceInfo(index, "GPU-%s" % index, "RTX", 24576, 24576 - free_mib, free_mib)


def test_plan_topologies_for_gemma4_bf16_on_four_24gb_gpus():
    config = SimpleNamespace(
        text_config=SimpleNamespace(num_attention_heads=16, num_key_value_heads=8, hidden_size=4096)
    )

    topologies = plan_topologies(
        [_gpu(index) for index in range(4)],
        config,
        "google/gemma-4-12B-it",
        0.9,
    )

    assert [(item.tensor_parallel_size, item.replica_count) for item in topologies] == [(2, 2), (4, 1)]


def test_gpu_groups_prefer_strong_tensor_parallel_links():
    gpus = [_gpu(index) for index in range(4)]
    scores = {
        (0, 2): 110,
        (2, 0): 110,
        (1, 3): 110,
        (3, 1): 110,
    }

    groups = group_gpus_for_tensor_parallel(gpus, 2, scores)

    assert [[gpu.index for gpu in group] for group in groups] == [[0, 2], [1, 3]]


def test_select_best_measurement_enforces_latency_ratio():
    topology = WorkerTopology(2, [[0, 1], [2, 3]])
    measurements = [
        BenchmarkMeasurement(topology, 1, 100, 1, {"short": 1.0}, {"short": 0.1}),
        BenchmarkMeasurement(topology, 8, 500, 8, {"short": 1.8}, {"short": 0.2}),
        BenchmarkMeasurement(topology, 16, 900, 16, {"short": 2.5}, {"short": 0.3}),
    ]

    best, ratio = select_best_measurement(measurements, 2.0)

    assert best.concurrency == 8
    assert ratio == pytest.approx(1.8)


def test_profile_round_trip_and_fingerprint_ignore_current_free_memory(tmp_path):
    settings = make_settings(tmp_path, INFERENCE_BACKEND="vllm", VLLM_GPU_MODE="auto", VLLM_GPU_COUNT=4)
    workload = load_workload(None)
    first_gpus = [_gpu(index, 24000) for index in range(4)]
    later_gpus = [_gpu(index, 20000) for index in range(4)]
    first = build_profile_fingerprint(settings, "google/gemma-4-12B-it", "abc", first_gpus, workload, {})
    later = build_profile_fingerprint(settings, "google/gemma-4-12B-it", "abc", later_gpus, workload, {})
    measurement = BenchmarkMeasurement(
        WorkerTopology(2, [[0, 1], [2, 3]]),
        16,
        1000,
        8,
        {"short": 1.5},
        {"short": 0.1},
    )
    profile = new_profile(first, "google/gemma-4-12B-it", measurement, 1.5, workload, [measurement])
    path = tmp_path / "profile.json"

    save_tuning_profile(path, profile)
    loaded = load_tuning_profile(path)

    assert first == later
    assert loaded == profile


def test_workload_file_validation(tmp_path):
    path = tmp_path / "workload.json"
    path.write_text('{"buckets": [{"name": "chat", "input_tokens": 100, "output_tokens": 50, "weight": 2}]}')

    workload = load_workload(path)

    assert workload[0].weight == pytest.approx(1.0)
