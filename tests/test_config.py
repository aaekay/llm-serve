from __future__ import annotations

import pytest

from llm_serve.config import Settings


def test_settings_loads_csv_and_boolean_fields(tmp_path):
    settings = Settings.load(
        base_dir=tmp_path,
        environ={
            "DEFAULT_MODEL_ID": "mock/default",
            "MODEL_ALLOWLIST": "mock/default,mock/reasoning",
            "REASONING_MODEL_ALLOWLIST": "mock/reasoning",
            "STARTUP_LOAD_DEFAULT_MODEL": "false",
            "VLLM_TRUST_REMOTE_CODE": "true",
            "VLLM_GPU_AUTO_SELECT": "false",
        },
    )

    assert settings.model_allowlist == ["mock/default", "mock/reasoning"]
    assert settings.reasoning_model_allowlist == ["mock/reasoning"]
    assert settings.startup_load_default_model is False
    assert settings.vllm_trust_remote_code is True
    assert settings.vllm_gpu_auto_select is False


def test_settings_rejects_invalid_default_model(tmp_path):
    with pytest.raises(ValueError):
        Settings.load(
            base_dir=tmp_path,
            environ={
                "DEFAULT_MODEL_ID": "missing/model",
                "MODEL_ALLOWLIST": "mock/default",
            },
        )


def test_settings_reject_fixed_gpu_count_above_visible_devices_when_auto_select_disabled(tmp_path):
    with pytest.raises(ValueError):
        Settings.load(
            base_dir=tmp_path,
            environ={
                "DEFAULT_MODEL_ID": "mock/default",
                "MODEL_ALLOWLIST": "mock/default,mock/reasoning",
                "REASONING_MODEL_ALLOWLIST": "mock/reasoning",
                "VLLM_GPU_AUTO_SELECT": "false",
                "CUDA_VISIBLE_DEVICES": "0,1",
                "VLLM_GPU_COUNT": "3",
            },
        )
