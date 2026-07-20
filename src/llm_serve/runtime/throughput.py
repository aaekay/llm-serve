from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from llm_serve.config import Settings

from .gpu_selection import GPUDeviceInfo, group_gpus_for_tensor_parallel


DEFAULT_WORKLOAD = (
    {"name": "short", "input_tokens": 256, "output_tokens": 128, "weight": 0.5},
    {"name": "medium", "input_tokens": 1024, "output_tokens": 256, "weight": 0.3},
    {"name": "long", "input_tokens": 4096, "output_tokens": 512, "weight": 0.2},
)


@dataclass(frozen=True)
class WorkloadBucket:
    name: str
    input_tokens: int
    output_tokens: int
    weight: float


@dataclass(frozen=True)
class WorkerTopology:
    tensor_parallel_size: int
    gpu_groups: List[List[int]]

    @property
    def replica_count(self) -> int:
        return len(self.gpu_groups)

    @property
    def label(self) -> str:
        return "tp%s-r%s" % (self.tensor_parallel_size, self.replica_count)


@dataclass(frozen=True)
class BenchmarkMeasurement:
    topology: WorkerTopology
    concurrency: int
    throughput_tokens_per_second: float
    requests_per_second: float
    p95_latency_seconds: Dict[str, float]
    p95_ttft_seconds: Dict[str, float]
    optimized_engine: bool = True


@dataclass(frozen=True)
class TuningProfile:
    schema_version: int
    fingerprint: str
    model_id: str
    created_at: str
    topology: WorkerTopology
    foreground_concurrency: int
    throughput_tokens_per_second: float
    requests_per_second: float
    p95_latency_ratio: float
    optimized_engine: bool
    workload: List[WorkloadBucket]
    measurements: List[BenchmarkMeasurement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TuningProfile":
        topology = WorkerTopology(**raw["topology"])
        workload = [WorkloadBucket(**item) for item in raw["workload"]]
        measurements = []
        for item in raw.get("measurements", []):
            normalized = dict(item)
            normalized["topology"] = WorkerTopology(**normalized["topology"])
            measurements.append(BenchmarkMeasurement(**normalized))
        return cls(
            schema_version=int(raw["schema_version"]),
            fingerprint=str(raw["fingerprint"]),
            model_id=str(raw["model_id"]),
            created_at=str(raw["created_at"]),
            topology=topology,
            foreground_concurrency=int(raw["foreground_concurrency"]),
            throughput_tokens_per_second=float(raw["throughput_tokens_per_second"]),
            requests_per_second=float(raw["requests_per_second"]),
            p95_latency_ratio=float(raw["p95_latency_ratio"]),
            optimized_engine=bool(raw.get("optimized_engine", True)),
            workload=workload,
            measurements=measurements,
        )


def load_workload(path: Optional[Path]) -> List[WorkloadBucket]:
    raw: Any = list(DEFAULT_WORKLOAD)
    if path is not None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Could not load VLLM autotune workload file '%s': %s" % (path, exc)) from exc
        if isinstance(raw, dict):
            raw = raw.get("buckets")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Autotune workload must contain a non-empty 'buckets' list")
    buckets: List[WorkloadBucket] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each autotune workload bucket must be an object")
        bucket = WorkloadBucket(
            name=str(item.get("name", "")).strip(),
            input_tokens=int(item.get("input_tokens", 0)),
            output_tokens=int(item.get("output_tokens", 0)),
            weight=float(item.get("weight", 0)),
        )
        if not bucket.name or bucket.input_tokens < 1 or bucket.output_tokens < 1 or bucket.weight <= 0:
            raise ValueError("Autotune workload buckets require a name and positive token counts/weight")
        buckets.append(bucket)
    total_weight = sum(bucket.weight for bucket in buckets)
    return [
        WorkloadBucket(bucket.name, bucket.input_tokens, bucket.output_tokens, bucket.weight / total_weight)
        for bucket in buckets
    ]


def compatible_tensor_parallel_sizes(model_config: Any, gpu_count: int) -> List[int]:
    config = getattr(model_config, "text_config", model_config)
    dimensions = [
        getattr(config, "num_attention_heads", None),
        getattr(config, "num_key_value_heads", None),
        getattr(config, "hidden_size", None),
    ]
    known_dimensions = [int(value) for value in dimensions if isinstance(value, int) and value > 0]
    return [
        size
        for size in range(1, gpu_count + 1)
        if gpu_count % size == 0 and all(dimension % size == 0 for dimension in known_dimensions)
    ]


def infer_parameter_billions(model_id: str, model_config: Any) -> Optional[float]:
    match = re.search(r"(?:^|[-_/])(\d+(?:\.\d+)?)b(?:[-_/]|$)", model_id, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    value = getattr(model_config, "num_parameters", None)
    if isinstance(value, (int, float)) and value > 0:
        return float(value) / 1_000_000_000
    return None


def plan_topologies(
    gpus: Sequence[GPUDeviceInfo],
    model_config: Any,
    model_id: str,
    memory_utilization: float,
    link_scores: Optional[Dict[tuple[int, int], int]] = None,
) -> List[WorkerTopology]:
    parameter_billions = infer_parameter_billions(model_id, model_config)
    candidates: List[WorkerTopology] = []
    for tp_size in compatible_tensor_parallel_sizes(model_config, len(gpus)):
        groups = group_gpus_for_tensor_parallel(gpus, tp_size, link_scores)
        if parameter_billions is not None:
            estimated_mib_per_gpu = parameter_billions * 1_000_000_000 * 2 * 1.2 / tp_size / (1024 * 1024)
            if any(
                estimated_mib_per_gpu + 2048
                > min(gpu.memory_total_mib for gpu in group) * memory_utilization
                for group in groups
            ):
                continue
        candidates.append(
            WorkerTopology(
                tensor_parallel_size=tp_size,
                gpu_groups=[[gpu.index for gpu in group] for group in groups],
            )
        )
    return candidates


def concurrency_levels(max_concurrency: int) -> List[int]:
    levels = [level for level in (1, 2, 4, 8, 16, 32, 64) if level <= max_concurrency]
    if not levels or levels[-1] != max_concurrency:
        levels.append(max_concurrency)
    return sorted(set(levels))


def select_best_measurement(
    measurements: Sequence[BenchmarkMeasurement],
    slo_multiplier: float,
) -> tuple[BenchmarkMeasurement, float]:
    if not measurements:
        raise ValueError("No successful throughput measurements were produced")
    buckets = set().union(*(measurement.p95_latency_seconds for measurement in measurements))
    baselines: Dict[str, float] = {}
    for bucket in buckets:
        values = [
            measurement.p95_latency_seconds[bucket]
            for measurement in measurements
            if measurement.concurrency == 1 and bucket in measurement.p95_latency_seconds
        ]
        if not values:
            raise ValueError("No concurrency-1 latency baseline was measured for bucket '%s'" % bucket)
        baselines[bucket] = min(values)

    qualified: List[tuple[BenchmarkMeasurement, float]] = []
    for measurement in measurements:
        ratios = [
            latency / max(baselines[bucket], 1e-9)
            for bucket, latency in measurement.p95_latency_seconds.items()
        ]
        ratio = max(ratios, default=math.inf)
        if ratio <= slo_multiplier:
            qualified.append((measurement, ratio))
    if not qualified:
        raise ValueError("No throughput measurement satisfied the latency SLO")
    return max(
        qualified,
        key=lambda item: (
            item[0].throughput_tokens_per_second,
            -max(item[0].p95_latency_seconds.values(), default=math.inf),
            -item[0].topology.tensor_parallel_size,
        ),
    )


def build_profile_fingerprint(
    settings: Settings,
    model_id: str,
    model_revision: Optional[str],
    gpus: Sequence[GPUDeviceInfo],
    workload: Sequence[WorkloadBucket],
    topology_scores: Mapping[tuple[int, int], int],
) -> str:
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = "not-installed"
    payload = {
        "schema": 1,
        "model_id": model_id,
        "model_revision": model_revision,
        "vllm_version": vllm_version,
        "gpus": [
            {
                "index": gpu.index,
                "uuid": gpu.uuid,
                "name": gpu.name,
                "memory_total_mib": gpu.memory_total_mib,
            }
            for gpu in gpus
        ],
        "topology": sorted((left, right, score) for (left, right), score in topology_scores.items()),
        "dtype": settings.vllm_dtype,
        "max_model_len": settings.vllm_max_model_len,
        "gpu_memory_utilization": settings.vllm_gpu_memory_utilization,
        "max_num_seqs": settings.vllm_max_num_seqs,
        "max_num_batched_tokens": settings.vllm_max_num_batched_tokens,
        "slo_multiplier": settings.vllm_autotune_slo_multiplier,
        "max_concurrency": settings.vllm_autotune_max_concurrency,
        "workload": [asdict(bucket) for bucket in workload],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tuning_profile_path(settings: Settings, model_id: str) -> Path:
    safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_") or "model"
    return settings.storage_root / "tuning" / (safe_model_id + ".json")


def load_tuning_profile(path: Path) -> Optional[TuningProfile]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return TuningProfile.from_dict(raw)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def save_tuning_profile(path: Path, profile: TuningProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.tmp" % (path.name, os.getpid()))
    temporary.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def new_profile(
    fingerprint: str,
    model_id: str,
    measurement: BenchmarkMeasurement,
    latency_ratio: float,
    workload: Sequence[WorkloadBucket],
    measurements: Sequence[BenchmarkMeasurement],
) -> TuningProfile:
    return TuningProfile(
        schema_version=1,
        fingerprint=fingerprint,
        model_id=model_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        topology=measurement.topology,
        foreground_concurrency=measurement.concurrency,
        throughput_tokens_per_second=measurement.throughput_tokens_per_second,
        requests_per_second=measurement.requests_per_second,
        p95_latency_ratio=latency_ratio,
        optimized_engine=measurement.optimized_engine,
        workload=list(workload),
        measurements=list(measurements),
    )
