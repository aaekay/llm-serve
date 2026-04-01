from __future__ import annotations

import csv
import io
import os
import subprocess
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class GPUDeviceInfo:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int

    @property
    def free_ratio(self) -> float:
        if self.memory_total_mib <= 0:
            return 0.0
        return self.memory_free_mib / float(self.memory_total_mib)


@dataclass(frozen=True)
class GPUSelectionResult:
    selected_gpus: List[GPUDeviceInfo]
    inspected_gpus: List[GPUDeviceInfo]
    cuda_visible_devices: str
    gpu_memory_utilization: float
    used_preferred_utilization: bool


def parse_nvidia_smi_csv(output: str) -> List[GPUDeviceInfo]:
    devices: List[GPUDeviceInfo] = []
    reader = csv.reader(io.StringIO(output))
    for row in reader:
        if not row:
            continue
        normalized = [column.strip() for column in row]
        if len(normalized) != 6:
            raise ValueError("Unexpected nvidia-smi output row: %s" % row)
        devices.append(
            GPUDeviceInfo(
                index=int(normalized[0]),
                uuid=normalized[1],
                name=normalized[2],
                memory_total_mib=int(normalized[3]),
                memory_used_mib=int(normalized[4]),
                memory_free_mib=int(normalized[5]),
            )
        )
    if not devices:
        raise ValueError("No GPUs were returned by nvidia-smi")
    return devices


def query_nvidia_smi_gpus() -> List[GPUDeviceInfo]:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi is not available on the host") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError("nvidia-smi failed: %s" % detail) from exc
    return parse_nvidia_smi_csv(completed.stdout)


def select_best_gpus(gpus: Sequence[GPUDeviceInfo], count: int) -> List[GPUDeviceInfo]:
    if count < 1:
        raise ValueError("GPU count must be at least 1")
    if len(gpus) < count:
        raise ValueError("Need %s GPUs but only %s were detected" % (count, len(gpus)))
    ranked = sorted(gpus, key=lambda gpu: (-gpu.memory_free_mib, gpu.index))
    chosen = ranked[:count]
    return sorted(chosen, key=lambda gpu: gpu.index)


def derive_gpu_memory_utilization(
    selected_gpus: Sequence[GPUDeviceInfo],
    preferred_utilization: float,
    reserve_fraction: float,
    minimum_utilization: float,
) -> Tuple[float, bool]:
    if not selected_gpus:
        raise ValueError("At least one GPU is required to derive gpu_memory_utilization")

    min_free_ratio = min(gpu.free_ratio for gpu in selected_gpus)
    if min_free_ratio >= preferred_utilization:
        return preferred_utilization, True

    derived_utilization = min_free_ratio - reserve_fraction
    if derived_utilization < minimum_utilization:
        raise ValueError(
            "Selected GPUs do not have enough free memory. "
            "minimum free ratio=%.3f, reserve_fraction=%.3f, minimum_utilization=%.3f"
            % (min_free_ratio, reserve_fraction, minimum_utilization)
        )
    return min(derived_utilization, preferred_utilization), False


def build_selection_result(
    inspected_gpus: Sequence[GPUDeviceInfo],
    gpu_count: int,
    preferred_utilization: float,
    reserve_fraction: float,
    minimum_utilization: float,
) -> GPUSelectionResult:
    selected_gpus = select_best_gpus(inspected_gpus, gpu_count)
    gpu_memory_utilization, used_preferred = derive_gpu_memory_utilization(
        selected_gpus=selected_gpus,
        preferred_utilization=preferred_utilization,
        reserve_fraction=reserve_fraction,
        minimum_utilization=minimum_utilization,
    )
    return GPUSelectionResult(
        selected_gpus=list(selected_gpus),
        inspected_gpus=list(inspected_gpus),
        cuda_visible_devices=",".join(str(gpu.index) for gpu in selected_gpus),
        gpu_memory_utilization=gpu_memory_utilization,
        used_preferred_utilization=used_preferred,
    )


def summarize_gpus(gpus: Sequence[GPUDeviceInfo]) -> str:
    return "; ".join(
        "index=%s name=%s free=%s/%sMiB"
        % (gpu.index, gpu.name, gpu.memory_free_mib, gpu.memory_total_mib)
        for gpu in gpus
    )
