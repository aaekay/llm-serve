from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import re
import time
from dataclasses import dataclass
from dataclasses import replace
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence

from llm_serve.config import Settings
from llm_serve.errors import NotReadyError
from llm_serve.tokenization import estimate_text_tokens
from llm_serve.types import InferenceRequest, InferenceResult

from .base import ModelBackend
from .gpu_selection import (
    GPUDeviceInfo,
    query_nvidia_smi_gpus,
    query_nvidia_smi_topology,
    derive_gpu_memory_utilization,
    select_best_gpus,
)
from .throughput import (
    BenchmarkMeasurement,
    TuningProfile,
    WorkloadBucket,
    WorkerTopology,
    build_profile_fingerprint,
    concurrency_levels,
    load_tuning_profile,
    load_workload,
    new_profile,
    plan_topologies,
    save_tuning_profile,
    select_best_measurement,
    tuning_profile_path,
)
from .vllm_pool import VLLMPoolBackend


logger = logging.getLogger(__name__)
PoolFactory = Callable[..., VLLMPoolBackend]


@dataclass(frozen=True)
class _RequestTiming:
    bucket: str
    latency_seconds: float
    ttft_seconds: float
    total_tokens: int


class ThroughputTuner:
    def __init__(
        self,
        model_id: str,
        settings: Settings,
        *,
        pool_factory: Optional[PoolFactory] = None,
    ) -> None:
        self.model_id = model_id
        self.settings = settings
        self.pool_factory = pool_factory or VLLMPoolBackend

    async def tune(
        self,
        topologies: Sequence[WorkerTopology],
        workload: Sequence[WorkloadBucket],
        fingerprint: str,
    ) -> TuningProfile:
        measurements: List[BenchmarkMeasurement] = []
        failures: List[str] = []
        for topology in topologies:
            pool: Optional[VLLMPoolBackend] = None
            optimized = True
            try:
                pool = self.pool_factory(
                    self.model_id,
                    self.settings,
                    topology,
                    optimized_engine=True,
                )
                try:
                    await pool.start()
                except Exception as optimized_exc:
                    await pool.shutdown()
                    optimized = False
                    logger.warning(
                        "Optimized vLLM flags failed for %s; retrying safe settings: %s",
                        topology.label,
                        optimized_exc,
                    )
                    pool = self.pool_factory(
                        self.model_id,
                        self.settings,
                        topology,
                        optimized_engine=False,
                    )
                    await pool.start()
                measurements.extend(await self._benchmark_pool(pool, topology, workload, optimized))
            except Exception as exc:
                failures.append("%s: %s" % (topology.label, exc))
                logger.exception("Autotune candidate %s failed", topology.label)
            finally:
                if pool is not None:
                    await pool.shutdown()
        if not measurements:
            raise NotReadyError(
                "No vLLM autotune topology could be benchmarked. %s" % "; ".join(failures)
            )
        best, latency_ratio = select_best_measurement(
            measurements,
            self.settings.vllm_autotune_slo_multiplier,
        )
        return new_profile(
            fingerprint,
            self.model_id,
            best,
            latency_ratio,
            workload,
            measurements,
        )

    async def _benchmark_pool(
        self,
        pool: VLLMPoolBackend,
        topology: WorkerTopology,
        workload: Sequence[WorkloadBucket],
        optimized_engine: bool,
    ) -> List[BenchmarkMeasurement]:
        manifest = _build_manifest(workload, self.settings.vllm_autotune_requests_per_level)
        # One short warmup per replica keeps compilation out of measured samples.
        warmup = WorkloadBucket("warmup", 32, 8, 1.0)
        await asyncio.gather(
            *(self._run_one(pool, warmup, -index - 1) for index in range(topology.replica_count))
        )
        results: List[BenchmarkMeasurement] = []
        for concurrency in concurrency_levels(self.settings.vllm_autotune_max_concurrency):
            all_timings: List[_RequestTiming] = []
            total_elapsed = 0.0
            for repetition in range(self.settings.vllm_autotune_repetitions):
                semaphore = asyncio.Semaphore(concurrency)

                async def run_bucket(bucket: WorkloadBucket, nonce: int) -> _RequestTiming:
                    async with semaphore:
                        return await self._run_one(pool, bucket, nonce)

                started = time.perf_counter()
                timings = await asyncio.gather(
                    *(
                        run_bucket(
                            bucket,
                            repetition * len(manifest) + index + concurrency * 10_000,
                        )
                        for index, bucket in enumerate(manifest)
                    )
                )
                total_elapsed += max(time.perf_counter() - started, 1e-9)
                all_timings.extend(timings)
            total_tokens = sum(timing.total_tokens for timing in all_timings)
            results.append(
                BenchmarkMeasurement(
                    topology=topology,
                    concurrency=concurrency,
                    throughput_tokens_per_second=total_tokens / max(total_elapsed, 1e-9),
                    requests_per_second=len(all_timings) / max(total_elapsed, 1e-9),
                    p95_latency_seconds=_bucket_percentiles(all_timings, "latency_seconds"),
                    p95_ttft_seconds=_bucket_percentiles(all_timings, "ttft_seconds"),
                    optimized_engine=optimized_engine,
                )
            )
            logger.info(
                "Autotune %s concurrency=%s throughput=%.2f total-tok/s requests=%.2f/s",
                topology.label,
                concurrency,
                results[-1].throughput_tokens_per_second,
                results[-1].requests_per_second,
            )
        return results

    async def _run_one(
        self,
        pool: VLLMPoolBackend,
        bucket: WorkloadBucket,
        nonce: int,
    ) -> _RequestTiming:
        prompt = _benchmark_prompt(bucket.input_tokens, bucket.output_tokens, nonce)
        request = InferenceRequest(
            model_id=self.model_id,
            prompt=prompt,
            max_output_tokens=bucket.output_tokens,
            temperature=0.0,
            top_p=1.0,
            stream=True,
        )
        started = time.perf_counter()
        first_token_at: Optional[float] = None
        chunks: List[str] = []
        async for chunk in pool.generate_stream(request):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            chunks.append(chunk)
        completed = time.perf_counter()
        output = "".join(chunks)
        return _RequestTiming(
            bucket=bucket.name,
            latency_seconds=max(completed - started, 1e-9),
            ttft_seconds=max((first_token_at or completed) - started, 1e-9),
            total_tokens=estimate_text_tokens(prompt) + estimate_text_tokens(output),
        )


class AutoTunedVLLMBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        settings: Settings,
        *,
        force_tune: bool = False,
        gpu_query_fn: Optional[Callable[[], List[GPUDeviceInfo]]] = None,
        topology_query_fn: Optional[Callable[[], Dict[tuple[int, int], int]]] = None,
        pool_factory: Optional[PoolFactory] = None,
    ) -> None:
        super().__init__(model_id)
        self.settings = settings
        self.force_tune = force_tune
        self.gpu_query_fn = gpu_query_fn or query_nvidia_smi_gpus
        self.topology_query_fn = topology_query_fn or query_nvidia_smi_topology
        self.pool_factory = pool_factory or VLLMPoolBackend
        self._pool: Optional[VLLMPoolBackend] = None
        self._profile: Optional[TuningProfile] = None
        self._status = "pending"

    @property
    def effective_foreground_concurrency(self) -> Optional[int]:
        return self._profile.foreground_concurrency if self._profile else None

    @property
    def tuning_snapshot(self) -> Dict[str, Any]:
        if self._pool is not None:
            snapshot = self._pool.tuning_snapshot
            snapshot["status"] = self._status if self._status != "ready" else snapshot["status"]
            return snapshot
        return {
            "status": self._status,
            "profile_valid": self._profile is not None,
            "tensor_parallel_size": None,
            "replica_count": None,
            "foreground_concurrency": None,
            "throughput_tokens_per_second": None,
            "p95_latency_ratio": None,
            "degraded_workers": 0,
        }

    async def start(self) -> None:
        self._status = "inspecting"
        _validate_vllm_version(self.model_id)
        inspected = self.gpu_query_fn()
        try:
            selected = _select_gpu_budget(inspected, self.settings)
        except ValueError as exc:
            raise NotReadyError("Could not select autotune GPU budget: %s" % exc) from exc
        try:
            effective_utilization, _ = derive_gpu_memory_utilization(
                selected,
                self.settings.vllm_gpu_memory_utilization,
                self.settings.vllm_gpu_memory_reserve_fraction,
                self.settings.vllm_gpu_memory_utilization_min,
            )
        except ValueError as exc:
            raise NotReadyError("Selected autotune GPUs do not have enough free memory: %s" % exc) from exc
        runtime_settings = replace(
            self.settings,
            vllm_gpu_memory_utilization=effective_utilization,
        )
        try:
            link_scores = self.topology_query_fn()
        except Exception as exc:
            logger.warning("GPU topology inspection failed; using index grouping: %s", exc)
            link_scores = {}
        model_config, revision = _load_model_config(self.model_id, self.settings)
        workload = load_workload(self.settings.vllm_autotune_workload_file)
        fingerprint = build_profile_fingerprint(
            runtime_settings,
            self.model_id,
            revision,
            selected,
            workload,
            link_scores,
        )
        path = tuning_profile_path(self.settings, self.model_id)
        profile = None if self.force_tune else load_tuning_profile(path)
        if profile is not None and profile.fingerprint != fingerprint:
            logger.info("Ignoring stale vLLM tuning profile at %s", path)
            profile = None
        if profile is None:
            if not self.settings.vllm_autotune_on_miss and not self.force_tune:
                raise NotReadyError(
                    "No valid vLLM tuning profile exists for '%s'. Run 'llm-serve tune'." % self.model_id
                )
            self._status = "tuning"
            topologies = plan_topologies(
                selected,
                model_config,
                self.model_id,
                runtime_settings.vllm_gpu_memory_utilization,
                link_scores,
            )
            if not topologies:
                raise NotReadyError("No feasible BF16 topology was found for the selected GPUs")
            tuner = ThroughputTuner(
                self.model_id,
                runtime_settings,
                pool_factory=self.pool_factory,
            )
            profile = await tuner.tune(topologies, workload, fingerprint)
            save_tuning_profile(path, profile)
            logger.info("Saved vLLM tuning profile to %s", path)
        self._profile = profile
        self._status = "starting"
        self._pool = self.pool_factory(
            self.model_id,
            runtime_settings,
            profile.topology,
            optimized_engine=profile.optimized_engine,
            profile=profile,
        )
        await self._pool.start()
        self._status = "ready"

    async def shutdown(self) -> None:
        if self._pool is not None:
            await self._pool.shutdown()
            self._pool = None
        self._status = "stopped"

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        return await self._require_pool().generate(request)

    async def generate_chat(self, request: InferenceRequest) -> InferenceResult:
        return await self._require_pool().generate_chat(request)

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        async for chunk in self._require_pool().generate_stream(request):
            yield chunk

    async def generate_chat_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        async for chunk in self._require_pool().generate_chat_stream(request):
            yield chunk

    def _require_pool(self) -> VLLMPoolBackend:
        if self._pool is None:
            raise NotReadyError("The auto-tuned vLLM worker pool is not ready")
        return self._pool


def _load_model_config(model_id: str, settings: Settings) -> tuple[Any, Optional[str]]:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise NotReadyError(
            "transformers is required for vLLM autotuning; install runtime dependencies first"
        ) from exc
    try:
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=settings.vllm_trust_remote_code,
            cache_dir=str(settings.model_cache_dir),
        )
    except Exception as exc:
        raise NotReadyError("Could not load model configuration for autotuning: %s" % exc) from exc
    revision = getattr(config, "_commit_hash", None)
    return config, revision if isinstance(revision, str) else None


def _validate_vllm_version(model_id: str) -> None:
    if "gemma-4-12b" not in model_id.lower():
        return
    try:
        raw_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise NotReadyError("vLLM is not installed; install the runtime dependency set") from exc
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", raw_version)
    if match is None:
        raise NotReadyError("Could not determine whether vLLM %s supports Gemma 4 Unified" % raw_version)
    version = tuple(int(part) for part in match.groups())
    if version < (0, 23, 0):
        raise NotReadyError(
            "Gemma 4 12B Unified requires vLLM>=0.23.0 for its native optimized implementation; "
            "found vLLM %s. Upgrade with 'uv sync --extra runtime --upgrade-package vllm'."
            % raw_version
        )


def _select_gpu_budget(inspected: Sequence[GPUDeviceInfo], settings: Settings) -> List[GPUDeviceInfo]:
    if settings.vllm_gpu_auto_select or not settings.cuda_visible_device_list:
        return select_best_gpus(inspected, settings.vllm_gpu_count)
    by_index = {str(gpu.index): gpu for gpu in inspected}
    by_uuid = {gpu.uuid: gpu for gpu in inspected}
    selected: List[GPUDeviceInfo] = []
    for identifier in settings.cuda_visible_device_list:
        gpu = by_index.get(identifier) or by_uuid.get(identifier)
        if gpu is None:
            raise ValueError("CUDA_VISIBLE_DEVICES entry '%s' was not detected" % identifier)
        if gpu not in selected:
            selected.append(gpu)
    if len(selected) < settings.vllm_gpu_count:
        raise ValueError(
            "VLLM_GPU_COUNT=%s requires at least that many fixed CUDA devices" % settings.vllm_gpu_count
        )
    return selected[: settings.vllm_gpu_count]


def _build_manifest(workload: Sequence[WorkloadBucket], request_count: int) -> List[WorkloadBucket]:
    manifest: List[WorkloadBucket] = []
    for bucket in workload:
        count = max(1, int(round(request_count * bucket.weight)))
        manifest.extend([bucket] * count)
    if len(manifest) > request_count:
        manifest = manifest[:request_count]
    while len(manifest) < request_count:
        manifest.append(workload[len(manifest) % len(workload)])
    return manifest


def _benchmark_prompt(input_tokens: int, output_tokens: int, nonce: int) -> str:
    prefix = "Benchmark request %s. Produce up to %s concise benchmark tokens. " % (nonce, output_tokens)
    remaining = max(1, input_tokens - estimate_text_tokens(prefix))
    return prefix + ("benchmark " * remaining)


def _bucket_percentiles(timings: Sequence[_RequestTiming], field_name: str) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for timing in timings:
        grouped.setdefault(timing.bucket, []).append(float(getattr(timing, field_name)))
    return {bucket: _nearest_rank_p95(values) for bucket, values in grouped.items()}


def _nearest_rank_p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, int(round(0.95 * (len(ordered) - 1))))
    return ordered[index]
