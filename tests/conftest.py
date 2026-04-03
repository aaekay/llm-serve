from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_serve.config import Settings


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    env = {
        "HOST": "127.0.0.1",
        "PORT": "11424",
        "INFERENCE_BACKEND": "mock",
        "DEFAULT_MODEL_ID": "mock/default",
        "MODEL_ALLOWLIST": "mock/default,mock/reasoning",
        "REASONING_MODEL_ALLOWLIST": "mock/reasoning",
        "PROMPT_MAX_PARALLEL": "8",
        "BATCH_MAX_PARALLEL": "2",
        "FOREGROUND_QUEUE_LIMIT": "16",
        "BATCH_QUEUE_LIMIT": "64",
        "MAX_INPUT_TOKENS": "4096",
        "MAX_OUTPUT_TOKENS": "128",
        "DEFAULT_TEMPERATURE": "0.2",
        "DEFAULT_TOP_P": "0.95",
        "STORAGE_ROOT": "runtime",
        "MODEL_CACHE_DIR": "models",
        "REQUEST_TIMEOUT_SECONDS": "5",
        "SWITCH_TIMEOUT_SECONDS": "5",
        "STARTUP_LOAD_DEFAULT_MODEL": "true",
        "STARTUP_SELF_TEST_ENABLED": "true",
        "STARTUP_SELF_TEST_BLOCKING": "false",
        "STARTUP_SELF_TEST_PROMPT": "Write a thousand word poem about sunrise.",
        "STARTUP_SELF_TEST_MAX_OUTPUT_TOKENS": "128",
        "VLLM_DTYPE": "auto",
        "VLLM_TOKENIZER_MODE": "auto",
        "VLLM_TRUST_REMOTE_CODE": "false",
        "VLLM_GPU_AUTO_SELECT": "false",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.9",
        "VLLM_GPU_MEMORY_UTILIZATION_MIN": "0.5",
        "VLLM_GPU_MEMORY_RESERVE_FRACTION": "0.05",
        "VLLM_DISABLE_CUSTOM_ALL_REDUCE": "true",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "OLLAMA_REQUEST_TIMEOUT_SECONDS": "5",
        "OLLAMA_REQUEST_TIMEOUT_RETRY_ENABLED": "true",
        "OLLAMA_REQUEST_TIMEOUT_RETRY_MULTIPLIER": "2.0",
        "OLLAMA_BATCH_TIMEOUT_RETRY_ENABLED": "true",
        "OLLAMA_BATCH_TIMEOUT_RETRY_MULTIPLIER": "2.0",
        "OLLAMA_BATCH_RETRY_OUTPUT_TOKENS_MULTIPLIER": "2.0",
        "OLLAMA_BATCH_RETRY_MAX_OUTPUT_TOKENS": "256",
        "MOCK_RESPONSE_DELAY_SECONDS": "0.0",
    }
    for key, value in overrides.items():
        env[key] = str(value)
    return Settings.load(base_dir=tmp_path, environ=env)
