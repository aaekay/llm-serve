from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


TRUTHY = {"1", "true", "yes", "on"}
FALSY = {"0", "false", "no", "off"}


def _parse_bool(raw_value: str, field_name: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in TRUTHY:
        return True
    if normalized in FALSY:
        return False
    raise ValueError("Invalid boolean for %s: %s" % (field_name, raw_value))


def _parse_csv(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _optional_str(raw_value: Optional[str]) -> Optional[str]:
    if raw_value is None:
        return None
    value = raw_value.strip()
    return value or None


def load_dotenv(path: Path) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if not path.exists():
        return parsed

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            parsed[key] = value
    return parsed


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    inference_backend: str
    default_model_id: str
    model_allowlist: List[str]
    reasoning_model_allowlist: List[str]
    prompt_max_parallel: int
    batch_max_parallel: int
    foreground_queue_limit: int
    batch_queue_limit: int
    max_input_tokens: int
    max_output_tokens: int
    default_temperature: float
    default_top_p: float
    storage_root: Path
    model_cache_dir: Path
    request_timeout_seconds: float
    switch_timeout_seconds: float
    startup_load_default_model: bool
    vllm_dtype: str
    vllm_tokenizer_mode: str
    vllm_trust_remote_code: bool
    cuda_visible_devices: Optional[str]
    vllm_gpu_count: int
    vllm_gpu_memory_utilization: float
    mock_response_delay_seconds: float

    @classmethod
    def load(
        cls,
        base_dir: Optional[Path] = None,
        environ: Optional[Dict[str, str]] = None,
    ) -> "Settings":
        root = base_dir or Path.cwd()
        env_source = dict(load_dotenv(root / ".env"))
        if environ:
            env_source.update(environ)
        else:
            env_source.update(os.environ)

        model_allowlist = _parse_csv(env_source.get("MODEL_ALLOWLIST", "mock/default,mock/reasoning"))
        reasoning_allowlist = _parse_csv(env_source.get("REASONING_MODEL_ALLOWLIST", "mock/reasoning"))
        default_model_id = env_source.get("DEFAULT_MODEL_ID", model_allowlist[0] if model_allowlist else "")
        model_cache_dir = _optional_str(env_source.get("MODEL_CACHE_DIR")) or "data/models"

        settings = cls(
            host=env_source.get("HOST", "127.0.0.1"),
            port=int(env_source.get("PORT", "11424")),
            inference_backend=env_source.get("INFERENCE_BACKEND", "mock").strip().lower(),
            default_model_id=default_model_id,
            model_allowlist=model_allowlist,
            reasoning_model_allowlist=reasoning_allowlist,
            prompt_max_parallel=int(env_source.get("PROMPT_MAX_PARALLEL", "8")),
            batch_max_parallel=int(env_source.get("BATCH_MAX_PARALLEL", "2")),
            foreground_queue_limit=int(env_source.get("FOREGROUND_QUEUE_LIMIT", "32")),
            batch_queue_limit=int(env_source.get("BATCH_QUEUE_LIMIT", "128")),
            max_input_tokens=int(env_source.get("MAX_INPUT_TOKENS", "4096")),
            max_output_tokens=int(env_source.get("MAX_OUTPUT_TOKENS", "1024")),
            default_temperature=float(env_source.get("DEFAULT_TEMPERATURE", "0.2")),
            default_top_p=float(env_source.get("DEFAULT_TOP_P", "0.95")),
            storage_root=(root / env_source.get("STORAGE_ROOT", "data/runtime")).resolve(),
            model_cache_dir=(root / model_cache_dir).resolve(),
            request_timeout_seconds=float(env_source.get("REQUEST_TIMEOUT_SECONDS", "120")),
            switch_timeout_seconds=float(env_source.get("SWITCH_TIMEOUT_SECONDS", "600")),
            startup_load_default_model=_parse_bool(
                env_source.get("STARTUP_LOAD_DEFAULT_MODEL", "true"),
                "STARTUP_LOAD_DEFAULT_MODEL",
            ),
            vllm_dtype=env_source.get("VLLM_DTYPE", "auto"),
            vllm_tokenizer_mode=env_source.get("VLLM_TOKENIZER_MODE", "auto"),
            vllm_trust_remote_code=_parse_bool(
                env_source.get("VLLM_TRUST_REMOTE_CODE", "false"),
                "VLLM_TRUST_REMOTE_CODE",
            ),
            cuda_visible_devices=_optional_str(env_source.get("CUDA_VISIBLE_DEVICES")),
            vllm_gpu_count=int(env_source.get("VLLM_GPU_COUNT", "1")),
            vllm_gpu_memory_utilization=float(env_source.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
            mock_response_delay_seconds=float(env_source.get("MOCK_RESPONSE_DELAY_SECONDS", "0.0")),
        )
        settings.validate()
        return settings

    @property
    def cuda_visible_device_list(self) -> List[str]:
        if not self.cuda_visible_devices:
            return []
        return _parse_csv(self.cuda_visible_devices)

    @property
    def huggingface_home(self) -> Path:
        return self.model_cache_dir

    @property
    def huggingface_hub_cache(self) -> Path:
        return self.model_cache_dir / "hub"

    @property
    def transformers_cache(self) -> Path:
        return self.model_cache_dir / "transformers"

    def validate(self) -> None:
        if not self.model_allowlist:
            raise ValueError("MODEL_ALLOWLIST must not be empty")
        if self.default_model_id not in self.model_allowlist:
            raise ValueError("DEFAULT_MODEL_ID must be present in MODEL_ALLOWLIST")
        if not set(self.reasoning_model_allowlist).issubset(set(self.model_allowlist)):
            raise ValueError("REASONING_MODEL_ALLOWLIST must be a subset of MODEL_ALLOWLIST")
        if self.prompt_max_parallel < 1:
            raise ValueError("PROMPT_MAX_PARALLEL must be at least 1")
        if self.batch_max_parallel < 1:
            raise ValueError("BATCH_MAX_PARALLEL must be at least 1")
        if self.max_input_tokens < 1:
            raise ValueError("MAX_INPUT_TOKENS must be at least 1")
        if self.max_output_tokens < 1:
            raise ValueError("MAX_OUTPUT_TOKENS must be at least 1")
        if self.foreground_queue_limit < 0:
            raise ValueError("FOREGROUND_QUEUE_LIMIT must be non-negative")
        if self.batch_queue_limit < 0:
            raise ValueError("BATCH_QUEUE_LIMIT must be non-negative")
        if self.inference_backend not in {"mock", "vllm"}:
            raise ValueError("INFERENCE_BACKEND must be 'mock' or 'vllm'")
        if self.vllm_gpu_count < 1:
            raise ValueError("VLLM_GPU_COUNT must be at least 1")
        visible_devices = self.cuda_visible_device_list
        if self.cuda_visible_devices and not visible_devices:
            raise ValueError("CUDA_VISIBLE_DEVICES must list at least one device when set")
        if visible_devices and self.vllm_gpu_count > len(visible_devices):
            raise ValueError("VLLM_GPU_COUNT cannot exceed the number of CUDA_VISIBLE_DEVICES entries")
