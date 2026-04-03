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
            "STARTUP_SELF_TEST_BLOCKING": "true",
            "VLLM_TRUST_REMOTE_CODE": "true",
            "VLLM_GPU_AUTO_SELECT": "false",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        },
    )

    assert settings.model_allowlist == ["mock/default", "mock/reasoning"]
    assert settings.reasoning_model_allowlist == ["mock/reasoning"]
    assert settings.startup_load_default_model is False
    assert settings.startup_self_test_blocking is True
    assert settings.vllm_trust_remote_code is True
    assert settings.vllm_enforce_eager is True
    assert settings.vllm_use_v1 is False
    assert settings.vllm_gpu_auto_select is False
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.ollama_request_timeout_seconds == 120
    assert settings.ollama_request_timeout_retry_enabled is True
    assert settings.ollama_request_timeout_retry_multiplier == 2.0
    assert settings.effective_default_model_id == "mock/default"


def test_settings_rejects_invalid_default_model(tmp_path):
    with pytest.raises(
        ValueError,
        match=r"DEFAULT_MODEL_ID 'missing/model' must be present in MODEL_ALLOWLIST \(mock/default\)",
    ):
        Settings.load(
            base_dir=tmp_path,
            environ={
                "DEFAULT_MODEL_ID": "missing/model",
                "MODEL_ALLOWLIST": "mock/default",
            },
        )


def test_settings_resolve_backend_specific_default_model_ids(tmp_path):
    vllm_settings = Settings.load(
        base_dir=tmp_path,
        environ={
            "DEFAULT_MODEL_ID": "mock/default",
            "VLLM_DEFAULT_MODEL_ID": "mock/reasoning",
            "MODEL_ALLOWLIST": "mock/default,mock/reasoning",
            "REASONING_MODEL_ALLOWLIST": "mock/reasoning",
            "INFERENCE_BACKEND": "vllm",
        },
    )
    ollama_settings = Settings.load(
        base_dir=tmp_path,
        environ={
            "DEFAULT_MODEL_ID": "mock/default",
            "OLLAMA_DEFAULT_MODEL_ID": "mock/reasoning",
            "MODEL_ALLOWLIST": "mock/default,mock/reasoning",
            "REASONING_MODEL_ALLOWLIST": "mock/reasoning",
            "INFERENCE_BACKEND": "ollama",
        },
    )

    assert vllm_settings.effective_default_model_id == "mock/reasoning"
    assert ollama_settings.effective_default_model_id == "mock/reasoning"


def test_settings_allow_active_backend_override_to_replace_invalid_fallback(tmp_path):
    settings = Settings.load(
        base_dir=tmp_path,
        environ={
            "DEFAULT_MODEL_ID": "missing/model",
            "VLLM_DEFAULT_MODEL_ID": "mock/default",
            "MODEL_ALLOWLIST": "mock/default,mock/reasoning",
            "REASONING_MODEL_ALLOWLIST": "mock/reasoning",
            "INFERENCE_BACKEND": "vllm",
        },
    )

    assert settings.effective_default_model_id == "mock/default"


def test_settings_reject_invalid_vllm_default_model(tmp_path):
    with pytest.raises(
        ValueError,
        match=r"VLLM_DEFAULT_MODEL_ID 'missing/model' must be present in MODEL_ALLOWLIST \(mock/default\)",
    ):
        Settings.load(
            base_dir=tmp_path,
            environ={
                "DEFAULT_MODEL_ID": "mock/default",
                "VLLM_DEFAULT_MODEL_ID": "missing/model",
                "MODEL_ALLOWLIST": "mock/default",
                "INFERENCE_BACKEND": "vllm",
            },
        )


def test_settings_reject_invalid_ollama_default_model(tmp_path):
    with pytest.raises(
        ValueError,
        match=r"OLLAMA_DEFAULT_MODEL_ID 'missing/model' must be present in MODEL_ALLOWLIST \(mock/default\)",
    ):
        Settings.load(
            base_dir=tmp_path,
            environ={
                "DEFAULT_MODEL_ID": "mock/default",
                "OLLAMA_DEFAULT_MODEL_ID": "missing/model",
                "MODEL_ALLOWLIST": "mock/default",
                "INFERENCE_BACKEND": "ollama",
            },
        )


def test_settings_parses_vllm_enforce_eager(tmp_path):
    settings = Settings.load(
        base_dir=tmp_path,
        environ={
            "DEFAULT_MODEL_ID": "mock/default",
            "MODEL_ALLOWLIST": "mock/default,mock/reasoning",
            "REASONING_MODEL_ALLOWLIST": "mock/reasoning",
            "VLLM_ENFORCE_EAGER": "true",
        },
    )

    assert settings.vllm_enforce_eager is True


def test_settings_parses_vllm_use_v1(tmp_path):
    off = Settings.load(
        base_dir=tmp_path,
        environ={
            "DEFAULT_MODEL_ID": "mock/default",
            "MODEL_ALLOWLIST": "mock/default,mock/reasoning",
            "REASONING_MODEL_ALLOWLIST": "mock/reasoning",
            "VLLM_USE_V1": "false",
        },
    )
    on = Settings.load(
        base_dir=tmp_path,
        environ={
            "DEFAULT_MODEL_ID": "mock/default",
            "MODEL_ALLOWLIST": "mock/default,mock/reasoning",
            "REASONING_MODEL_ALLOWLIST": "mock/reasoning",
            "VLLM_USE_V1": "true",
        },
    )

    assert off.vllm_use_v1 is False
    assert on.vllm_use_v1 is True


def test_settings_reject_startup_self_test_tokens_above_max_output_tokens(tmp_path):
    with pytest.raises(ValueError):
        Settings.load(
            base_dir=tmp_path,
            environ={
                "DEFAULT_MODEL_ID": "mock/default",
                "MODEL_ALLOWLIST": "mock/default,mock/reasoning",
                "REASONING_MODEL_ALLOWLIST": "mock/reasoning",
                "MAX_OUTPUT_TOKENS": "128",
                "STARTUP_SELF_TEST_MAX_OUTPUT_TOKENS": "256",
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
                "INFERENCE_BACKEND": "vllm",
                "VLLM_GPU_AUTO_SELECT": "false",
                "CUDA_VISIBLE_DEVICES": "0,1",
                "VLLM_GPU_COUNT": "3",
            },
        )


def test_settings_accepts_ollama_backend(tmp_path):
    settings = Settings.load(
        base_dir=tmp_path,
        environ={
            "DEFAULT_MODEL_ID": "mock/default",
            "MODEL_ALLOWLIST": "mock/default,mock/reasoning",
            "REASONING_MODEL_ALLOWLIST": "mock/reasoning",
            "INFERENCE_BACKEND": "ollama",
            "OLLAMA_BASE_URL": "http://ollama.local:11434/",
        },
    )

    assert settings.inference_backend == "ollama"
    assert settings.ollama_base_url == "http://ollama.local:11434"


def test_settings_loads_ollama_retry_controls(tmp_path):
    settings = Settings.load(
        base_dir=tmp_path,
        environ={
            "DEFAULT_MODEL_ID": "mock/default",
            "MODEL_ALLOWLIST": "mock/default,mock/reasoning",
            "REASONING_MODEL_ALLOWLIST": "mock/reasoning",
            "INFERENCE_BACKEND": "ollama",
            "OLLAMA_REQUEST_TIMEOUT_SECONDS": "9",
            "OLLAMA_REQUEST_TIMEOUT_RETRY_ENABLED": "false",
            "OLLAMA_REQUEST_TIMEOUT_RETRY_MULTIPLIER": "3",
            "OLLAMA_BATCH_TIMEOUT_RETRY_ENABLED": "true",
            "OLLAMA_BATCH_TIMEOUT_RETRY_MULTIPLIER": "4",
            "OLLAMA_BATCH_RETRY_OUTPUT_TOKENS_MULTIPLIER": "1.5",
            "OLLAMA_BATCH_RETRY_MAX_OUTPUT_TOKENS": "4096",
        },
    )

    assert settings.ollama_request_timeout_seconds == 9
    assert settings.ollama_request_timeout_retry_enabled is False
    assert settings.ollama_request_timeout_retry_multiplier == 3
    assert settings.ollama_batch_timeout_retry_enabled is True
    assert settings.ollama_batch_timeout_retry_multiplier == 4
    assert settings.ollama_batch_retry_output_tokens_multiplier == 1.5
    assert settings.ollama_batch_retry_max_output_tokens == 4096
