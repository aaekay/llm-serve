from __future__ import annotations

import pytest

from llm_serve.runtime.gpu_selection import (
    GPUDeviceInfo,
    build_selection_result,
    derive_gpu_memory_utilization,
    parse_nvidia_smi_csv,
    select_best_gpus,
)


def test_parse_nvidia_smi_csv_parses_rows():
    parsed = parse_nvidia_smi_csv(
        "\n".join(
            [
                "0, GPU-aaa, RTX 6000, 24576, 1024, 23552",
                "1, GPU-bbb, RTX 6000, 24576, 4096, 20480",
            ]
        )
    )

    assert [gpu.index for gpu in parsed] == [0, 1]
    assert parsed[0].uuid == "GPU-aaa"
    assert parsed[1].memory_free_mib == 20480


def test_select_best_gpus_prefers_more_free_memory_then_lower_index():
    gpus = [
        GPUDeviceInfo(2, "GPU-2", "GPU 2", 10000, 3000, 7000),
        GPUDeviceInfo(0, "GPU-0", "GPU 0", 10000, 2000, 8000),
        GPUDeviceInfo(1, "GPU-1", "GPU 1", 10000, 2000, 8000),
    ]

    selected = select_best_gpus(gpus, 2)

    assert [gpu.index for gpu in selected] == [0, 1]


def test_derive_gpu_memory_utilization_uses_preferred_cap_when_possible():
    selected = [
        GPUDeviceInfo(0, "GPU-0", "GPU 0", 10000, 500, 9500),
        GPUDeviceInfo(1, "GPU-1", "GPU 1", 10000, 1000, 9000),
    ]

    utilization, used_preferred = derive_gpu_memory_utilization(
        selected_gpus=selected,
        preferred_utilization=0.9,
        reserve_fraction=0.05,
        minimum_utilization=0.5,
    )

    assert utilization == pytest.approx(0.9)
    assert used_preferred is True


def test_derive_gpu_memory_utilization_falls_back_from_current_free_memory():
    selected = [
        GPUDeviceInfo(0, "GPU-0", "GPU 0", 10000, 2200, 7800),
        GPUDeviceInfo(2, "GPU-2", "GPU 2", 10000, 1800, 8200),
    ]

    utilization, used_preferred = derive_gpu_memory_utilization(
        selected_gpus=selected,
        preferred_utilization=0.9,
        reserve_fraction=0.05,
        minimum_utilization=0.5,
    )

    assert utilization == pytest.approx(0.73)
    assert used_preferred is False


def test_derive_gpu_memory_utilization_rejects_insufficient_free_memory():
    selected = [
        GPUDeviceInfo(0, "GPU-0", "GPU 0", 10000, 5200, 4800),
        GPUDeviceInfo(1, "GPU-1", "GPU 1", 10000, 5100, 4900),
    ]

    with pytest.raises(ValueError):
        derive_gpu_memory_utilization(
            selected_gpus=selected,
            preferred_utilization=0.9,
            reserve_fraction=0.05,
            minimum_utilization=0.5,
        )


def test_build_selection_result_selects_best_pair_and_derives_ratio():
    selection = build_selection_result(
        inspected_gpus=[
            GPUDeviceInfo(0, "GPU-0", "GPU 0", 10000, 2200, 7800),
            GPUDeviceInfo(1, "GPU-1", "GPU 1", 10000, 500, 9500),
            GPUDeviceInfo(2, "GPU-2", "GPU 2", 10000, 1200, 8800),
        ],
        gpu_count=2,
        preferred_utilization=0.9,
        reserve_fraction=0.05,
        minimum_utilization=0.5,
    )

    assert [gpu.index for gpu in selection.selected_gpus] == [1, 2]
    assert selection.cuda_visible_devices == "1,2"
    assert selection.gpu_memory_utilization == pytest.approx(0.83)
