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
        if key.startswith("export "):
            key = key[len("export "):].strip()
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
    vllm_default_model_id: Optional[str]
    ollama_default_model_id: Optional[str]
    model_allowlist: List[str]
    reasoning_model_allowlist: List[str]
    enable_thinking: bool
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
    startup_self_test_enabled: bool
    startup_self_test_blocking: bool
    startup_self_test_prompt: str
    startup_self_test_max_output_tokens: int
    startup_concurrency_test: bool
    vllm_dtype: str
    vllm_tokenizer_mode: str
    vllm_trust_remote_code: bool
    vllm_enforce_eager: bool
    vllm_use_v1: bool
    vllm_language_model_only: bool
    vllm_max_model_len: Optional[int]
    vllm_gdn_prefill_backend: Optional[str]
    vllm_disable_custom_all_reduce: bool
    vllm_enable_prefix_caching: bool
    vllm_enable_chunked_prefill: bool
    vllm_max_num_seqs: int
    vllm_max_num_batched_tokens: int
    vllm_swap_space_gb: Optional[float]
    startup_concurrency_test_max_level: int
    vllm_gpu_auto_select: bool
    cuda_visible_devices: Optional[str]
    vllm_gpu_count: int
    vllm_gpu_memory_utilization: float
    vllm_gpu_memory_utilization_min: float
    vllm_gpu_memory_reserve_fraction: float
    ollama_base_url: str
    ollama_request_timeout_seconds: float
    ollama_request_timeout_retry_enabled: bool
    ollama_request_timeout_retry_multiplier: float
    ollama_batch_timeout_retry_enabled: bool
    ollama_batch_timeout_retry_multiplier: float
    ollama_batch_retry_output_tokens_multiplier: float
    ollama_batch_retry_max_output_tokens: int
    storage_retention_hours: int
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
        vllm_default_model_id = _optional_str(env_source.get("VLLM_DEFAULT_MODEL_ID"))
        ollama_default_model_id = _optional_str(env_source.get("OLLAMA_DEFAULT_MODEL_ID"))
        model_cache_dir = _optional_str(env_source.get("MODEL_CACHE_DIR")) or "data/models"

        settings = cls(
            host=env_source.get("HOST", "127.0.0.1"),
            port=int(env_source.get("PORT", "11424")),
            inference_backend=env_source.get("INFERENCE_BACKEND", "mock").strip().lower(),
            default_model_id=default_model_id,
            vllm_default_model_id=vllm_default_model_id,
            ollama_default_model_id=ollama_default_model_id,
            model_allowlist=model_allowlist,
            reasoning_model_allowlist=reasoning_allowlist,
            enable_thinking=_parse_bool(
                env_source.get("ENABLE_THINKING", "false"),
                "ENABLE_THINKING",
            ),
            prompt_max_parallel=int(env_source.get("PROMPT_MAX_PARALLEL", "8")),
            batch_max_parallel=int(env_source.get("BATCH_MAX_PARALLEL", "4")),
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
            startup_self_test_enabled=_parse_bool(
                env_source.get("STARTUP_SELF_TEST_ENABLED", "true"),
                "STARTUP_SELF_TEST_ENABLED",
            ),
            startup_self_test_blocking=_parse_bool(
                env_source.get("STARTUP_SELF_TEST_BLOCKING", "false"),
                "STARTUP_SELF_TEST_BLOCKING",
            ),
            startup_self_test_prompt=env_source.get(
                "STARTUP_SELF_TEST_PROMPT",
                "Write a thousand word poem about dawn breaking over mountains.",
            ),
            startup_self_test_max_output_tokens=int(
                env_source.get(
                    "STARTUP_SELF_TEST_MAX_OUTPUT_TOKENS",
                    env_source.get("MAX_OUTPUT_TOKENS", "1024"),
                )
            ),
            startup_concurrency_test=_parse_bool(
                env_source.get("STARTUP_CONCURRENCY_TEST", "true"),
                "STARTUP_CONCURRENCY_TEST",
            ),
            vllm_dtype=env_source.get("VLLM_DTYPE", "bfloat16"),
            vllm_tokenizer_mode=env_source.get("VLLM_TOKENIZER_MODE", "auto"),
            vllm_trust_remote_code=_parse_bool(
                env_source.get("VLLM_TRUST_REMOTE_CODE", "false"),
                "VLLM_TRUST_REMOTE_CODE",
            ),
            vllm_enforce_eager=_parse_bool(
                env_source.get("VLLM_ENFORCE_EAGER", "true"),
                "VLLM_ENFORCE_EAGER",
            ),
            vllm_use_v1=_parse_bool(
                env_source.get("VLLM_USE_V1", "false"),
                "VLLM_USE_V1",
            ),
            vllm_language_model_only=_parse_bool(
                env_source.get("VLLM_LANGUAGE_MODEL_ONLY", "true"),
                "VLLM_LANGUAGE_MODEL_ONLY",
            ),
            vllm_max_model_len=(
                int(env_source["VLLM_MAX_MODEL_LEN"])
                if _optional_str(env_source.get("VLLM_MAX_MODEL_LEN")) is not None
                else None
            ),
            vllm_gdn_prefill_backend=_optional_str(
                env_source.get("VLLM_GDN_PREFILL_BACKEND")
            ),
            vllm_disable_custom_all_reduce=_parse_bool(
                env_source.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "true"),
                "VLLM_DISABLE_CUSTOM_ALL_REDUCE",
            ),
            vllm_enable_prefix_caching=_parse_bool(
                env_source.get("VLLM_ENABLE_PREFIX_CACHING", "true"),
                "VLLM_ENABLE_PREFIX_CACHING",
            ),
            vllm_enable_chunked_prefill=_parse_bool(
                env_source.get("VLLM_ENABLE_CHUNKED_PREFILL", "true"),
                "VLLM_ENABLE_CHUNKED_PREFILL",
            ),
            vllm_max_num_seqs=int(env_source.get("VLLM_MAX_NUM_SEQS", "64")),
            vllm_max_num_batched_tokens=int(
                env_source.get("VLLM_MAX_NUM_BATCHED_TOKENS", "8192")
            ),
            vllm_swap_space_gb=(
                float(env_source["VLLM_SWAP_SPACE_GB"])
                if _optional_str(env_source.get("VLLM_SWAP_SPACE_GB")) is not None
                else None
            ),
            startup_concurrency_test_max_level=int(
                env_source.get("STARTUP_CONCURRENCY_TEST_MAX_LEVEL", "32")
            ),
            vllm_gpu_auto_select=_parse_bool(
                env_source.get("VLLM_GPU_AUTO_SELECT", "true"),
                "VLLM_GPU_AUTO_SELECT",
            ),
            cuda_visible_devices=_optional_str(env_source.get("CUDA_VISIBLE_DEVICES")),
            vllm_gpu_count=int(env_source.get("VLLM_GPU_COUNT", "1")),
            vllm_gpu_memory_utilization=float(env_source.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
            vllm_gpu_memory_utilization_min=float(
                env_source.get("VLLM_GPU_MEMORY_UTILIZATION_MIN", "0.5")
            ),
            vllm_gpu_memory_reserve_fraction=float(
                env_source.get("VLLM_GPU_MEMORY_RESERVE_FRACTION", "0.05")
            ),
            ollama_base_url=env_source.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            ollama_request_timeout_seconds=float(
                env_source.get(
                    "OLLAMA_REQUEST_TIMEOUT_SECONDS",
                    env_source.get("REQUEST_TIMEOUT_SECONDS", "120"),
                )
            ),
            ollama_request_timeout_retry_enabled=_parse_bool(
                env_source.get("OLLAMA_REQUEST_TIMEOUT_RETRY_ENABLED", "true"),
                "OLLAMA_REQUEST_TIMEOUT_RETRY_ENABLED",
            ),
            ollama_request_timeout_retry_multiplier=float(
                env_source.get("OLLAMA_REQUEST_TIMEOUT_RETRY_MULTIPLIER", "2.0")
            ),
            ollama_batch_timeout_retry_enabled=_parse_bool(
                env_source.get("OLLAMA_BATCH_TIMEOUT_RETRY_ENABLED", "true"),
                "OLLAMA_BATCH_TIMEOUT_RETRY_ENABLED",
            ),
            ollama_batch_timeout_retry_multiplier=float(
                env_source.get("OLLAMA_BATCH_TIMEOUT_RETRY_MULTIPLIER", "2.0")
            ),
            ollama_batch_retry_output_tokens_multiplier=float(
                env_source.get("OLLAMA_BATCH_RETRY_OUTPUT_TOKENS_MULTIPLIER", "2.0")
            ),
            ollama_batch_retry_max_output_tokens=int(
                env_source.get(
                    "OLLAMA_BATCH_RETRY_MAX_OUTPUT_TOKENS",
                    str(int(env_source.get("MAX_OUTPUT_TOKENS", "1024")) * 2),
                )
            ),
            storage_retention_hours=int(env_source.get("STORAGE_RETENTION_HOURS", "720")),
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

    @property
    def effective_default_model_id(self) -> str:
        if self.inference_backend == "vllm" and self.vllm_default_model_id:
            return self.vllm_default_model_id
        if self.inference_backend == "ollama" and self.ollama_default_model_id:
            return self.ollama_default_model_id
        return self.default_model_id

    @property
    def effective_default_model_env_var(self) -> str:
        if self.inference_backend == "vllm" and self.vllm_default_model_id:
            return "VLLM_DEFAULT_MODEL_ID"
        if self.inference_backend == "ollama" and self.ollama_default_model_id:
            return "OLLAMA_DEFAULT_MODEL_ID"
        return "DEFAULT_MODEL_ID"

    def validate(self) -> None:
        if not self.model_allowlist:
            raise ValueError("MODEL_ALLOWLIST must not be empty")
        if self.vllm_default_model_id is not None:
            self._validate_allowed_default_model(self.vllm_default_model_id, "VLLM_DEFAULT_MODEL_ID")
        if self.ollama_default_model_id is not None:
            self._validate_allowed_default_model(self.ollama_default_model_id, "OLLAMA_DEFAULT_MODEL_ID")
        self._validate_allowed_default_model(
            self.effective_default_model_id,
            self.effective_default_model_env_var,
        )
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
        if self.startup_self_test_max_output_tokens < 1:
            raise ValueError("STARTUP_SELF_TEST_MAX_OUTPUT_TOKENS must be at least 1")
        if self.startup_self_test_max_output_tokens > self.max_output_tokens:
            raise ValueError("STARTUP_SELF_TEST_MAX_OUTPUT_TOKENS cannot exceed MAX_OUTPUT_TOKENS")
        if self.startup_concurrency_test_max_level < 2:
            raise ValueError("STARTUP_CONCURRENCY_TEST_MAX_LEVEL must be at least 2")
        if self.foreground_queue_limit < 0:
            raise ValueError("FOREGROUND_QUEUE_LIMIT must be non-negative")
        if self.batch_queue_limit < 0:
            raise ValueError("BATCH_QUEUE_LIMIT must be non-negative")
        if self.storage_retention_hours < 0:
            raise ValueError("STORAGE_RETENTION_HOURS must be non-negative (0 disables cleanup)")
        if self.inference_backend not in {"mock", "vllm", "ollama"}:
            raise ValueError("INFERENCE_BACKEND must be 'mock', 'vllm', or 'ollama'")
        if not self.ollama_base_url:
            raise ValueError("OLLAMA_BASE_URL must not be empty")
        if self.ollama_request_timeout_seconds <= 0:
            raise ValueError("OLLAMA_REQUEST_TIMEOUT_SECONDS must be greater than 0")
        if self.ollama_request_timeout_retry_multiplier < 1:
            raise ValueError("OLLAMA_REQUEST_TIMEOUT_RETRY_MULTIPLIER must be at least 1")
        if self.ollama_batch_timeout_retry_multiplier < 1:
            raise ValueError("OLLAMA_BATCH_TIMEOUT_RETRY_MULTIPLIER must be at least 1")
        if self.ollama_batch_retry_output_tokens_multiplier < 1:
            raise ValueError("OLLAMA_BATCH_RETRY_OUTPUT_TOKENS_MULTIPLIER must be at least 1")
        if self.ollama_batch_retry_max_output_tokens < 1:
            raise ValueError("OLLAMA_BATCH_RETRY_MAX_OUTPUT_TOKENS must be at least 1")
        if self.inference_backend != "vllm":
            return
        if self.vllm_gpu_count < 1:
            raise ValueError("VLLM_GPU_COUNT must be at least 1")
        if self.vllm_max_model_len is not None and self.vllm_max_model_len < 1:
            raise ValueError("VLLM_MAX_MODEL_LEN must be at least 1 when set")
        if self.vllm_gdn_prefill_backend not in {None, "triton", "flashinfer"}:
            raise ValueError("VLLM_GDN_PREFILL_BACKEND must be 'triton', 'flashinfer', or empty")
        if not 0 < self.vllm_gpu_memory_utilization <= 1:
            raise ValueError("VLLM_GPU_MEMORY_UTILIZATION must be between 0 and 1")
        if not 0 < self.vllm_gpu_memory_utilization_min <= 1:
            raise ValueError("VLLM_GPU_MEMORY_UTILIZATION_MIN must be between 0 and 1")
        if self.vllm_gpu_memory_utilization_min > self.vllm_gpu_memory_utilization:
            raise ValueError("VLLM_GPU_MEMORY_UTILIZATION_MIN cannot exceed VLLM_GPU_MEMORY_UTILIZATION")
        if not 0 <= self.vllm_gpu_memory_reserve_fraction < 1:
            raise ValueError("VLLM_GPU_MEMORY_RESERVE_FRACTION must be between 0 and 1")
        visible_devices = self.cuda_visible_device_list
        if self.cuda_visible_devices and not visible_devices:
            raise ValueError("CUDA_VISIBLE_DEVICES must list at least one device when set")
        if not self.vllm_gpu_auto_select and visible_devices and self.vllm_gpu_count > len(visible_devices):
            raise ValueError("VLLM_GPU_COUNT cannot exceed the number of CUDA_VISIBLE_DEVICES entries")
        if self.vllm_max_num_seqs < 1:
            raise ValueError("VLLM_MAX_NUM_SEQS must be at least 1")
        if self.vllm_max_num_batched_tokens < 1:
            raise ValueError("VLLM_MAX_NUM_BATCHED_TOKENS must be at least 1")
        if (
            self.vllm_max_model_len is not None
            and self.vllm_max_num_batched_tokens < self.vllm_max_model_len
        ):
            raise ValueError(
                "VLLM_MAX_NUM_BATCHED_TOKENS must be >= VLLM_MAX_MODEL_LEN"
            )
        if self.vllm_swap_space_gb is not None and self.vllm_swap_space_gb < 0:
            raise ValueError("VLLM_SWAP_SPACE_GB must be non-negative when set")

    def _validate_allowed_default_model(self, model_id: str, env_var_name: str) -> None:
        if model_id not in self.model_allowlist:
            raise ValueError(
                "%s '%s' must be present in MODEL_ALLOWLIST (%s)"
                % (env_var_name, model_id, ", ".join(self.model_allowlist))
            )
