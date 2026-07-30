from __future__ import annotations

import asyncio
import logging
import statistics
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, Optional

from llm_serve.config import Settings
from llm_serve.types import InferenceRequest, InferenceResult, StartupSelfTestResult


logger = logging.getLogger(__name__)
Generate = Callable[[InferenceRequest], Awaitable[InferenceResult]]


class StartupDiagnostics:
    def __init__(
        self,
        settings: Settings,
        generate: Generate,
        run_foreground: Generate,
    ) -> None:
        self.settings = settings
        self._generate = generate
        self._run_foreground = run_foreground
        self.result = StartupSelfTestResult(status="pending")
        self._task: Optional[asyncio.Task[None]] = None

    def skip(self) -> None:
        self.result = StartupSelfTestResult(status="skipped")

    async def start(self) -> None:
        await self._start_self_test()
        if not self.settings.startup_concurrency_test or (
            self.settings.inference_backend == "vllm"
            and self.settings.vllm_gpu_mode == "auto"
        ):
            return
        if self._task and not self._task.done():
            logger.info("Waiting for startup self-test to finish before concurrency test.")
            try:
                await self._task
            except Exception:
                pass
        await self._run_concurrency_test()

    async def shutdown(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def snapshot(self) -> Dict[str, object]:
        return {
            "status": self.result.status,
            "prompt": self.result.prompt,
            "model_id": self.result.model_id,
            "started_at": self.result.started_at,
            "completed_at": self.result.completed_at,
            "latency_seconds": self.result.latency_seconds,
            "completion_tokens": self.result.completion_tokens,
            "tokens_per_second": self.result.tokens_per_second,
            "error": self.result.error,
        }

    async def _run_self_test(self) -> None:
        prompt = self.settings.startup_self_test_prompt.strip()
        model_id = self.settings.effective_default_model_id
        started_at = datetime.now(timezone.utc)
        self.result = StartupSelfTestResult(
            status="running",
            prompt=prompt,
            model_id=model_id,
            started_at=started_at.isoformat(),
        )
        request = InferenceRequest(
            model_id=model_id,
            prompt=prompt,
            max_output_tokens=self.settings.startup_self_test_max_output_tokens,
            temperature=self.settings.default_temperature,
            top_p=self.settings.default_top_p,
            stream=False,
        )
        logger.info(
            "Startup self-test started for model '%s' with max_output_tokens=%s and prompt=%r",
            model_id,
            self.settings.startup_self_test_max_output_tokens,
            _preview_text(prompt),
        )
        started = time.perf_counter()
        try:
            result = await self._generate(request)
        except Exception as exc:
            self.result = StartupSelfTestResult(
                status="failed",
                prompt=prompt,
                model_id=model_id,
                started_at=started_at.isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                error=str(exc),
            )
            logger.exception("Startup self-test failed for model '%s'", model_id)
            raise

        latency = max(time.perf_counter() - started, 1e-9)
        self.result = StartupSelfTestResult(
            status="passed",
            prompt=prompt,
            model_id=model_id,
            started_at=started_at.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            latency_seconds=latency,
            completion_tokens=result.completion_tokens,
            tokens_per_second=result.completion_tokens / latency,
        )
        logger.info(
            "Startup self-test passed for model '%s': completion_tokens=%s latency=%.3fs tokens_per_second=%.2f",
            model_id,
            result.completion_tokens,
            latency,
            self.result.tokens_per_second,
        )
        logger.info("Startup self-test output for model '%s':\n%s", model_id, result.text)

    async def _start_self_test(self) -> None:
        prompt = self.settings.startup_self_test_prompt.strip()
        model_id = self.settings.effective_default_model_id
        if not self.settings.startup_self_test_enabled:
            self.result = StartupSelfTestResult(status="disabled", prompt=prompt or None)
            logger.info("Startup self-test is disabled.")
            return
        if not prompt:
            self.result = StartupSelfTestResult(status="skipped")
            logger.info("Startup self-test skipped because STARTUP_SELF_TEST_PROMPT is empty.")
            return
        if self.settings.startup_self_test_blocking:
            logger.info("Startup self-test is running in blocking mode for model '%s'.", model_id)
            await self._run_self_test()
            return

        self.result = StartupSelfTestResult(status="queued", prompt=prompt, model_id=model_id)
        logger.info(
            "Startup self-test queued for model '%s' with max_output_tokens=%s and prompt=%r",
            model_id,
            self.settings.startup_self_test_max_output_tokens,
            _preview_text(prompt),
        )
        self._task = asyncio.create_task(self._run_self_test_background())
        self._task.add_done_callback(self._clear_task)

    async def _run_self_test_background(self) -> None:
        try:
            await self._run_self_test()
        except asyncio.CancelledError:
            if self.result.status in {"queued", "running"}:
                self.result = StartupSelfTestResult(
                    status="cancelled",
                    prompt=self.result.prompt,
                    model_id=self.result.model_id,
                    started_at=self.result.started_at,
                )
            raise
        except Exception:
            return

    async def _run_concurrency_test(self) -> None:
        prompt = self.settings.startup_self_test_prompt.strip()
        model_id = self.settings.effective_default_model_id
        max_level = min(
            self.settings.startup_concurrency_test_max_level,
            self.settings.prompt_max_parallel,
        )
        if max_level < 2:
            logger.info(
                "Skipping startup concurrency test: effective max level %s is below minimum 2 "
                "(STARTUP_CONCURRENCY_TEST_MAX_LEVEL=%s, PROMPT_MAX_PARALLEL=%s).",
                max_level,
                self.settings.startup_concurrency_test_max_level,
                self.settings.prompt_max_parallel,
            )
            return
        if not prompt:
            logger.info("Skipping startup concurrency test: no self-test prompt configured.")
            return

        levels = [level for level in (2, 4, 8, 16, 24, 32) if level <= max_level]
        if not levels or levels[-1] != max_level:
            levels.append(max_level)
        logger.info("=== Startup Concurrency Test (model: %s) ===", model_id)
        logger.info(
            "%-7s  %-9s  %-15s  %-20s  %-22s  %-22s",
            "Level",
            "Requests",
            "Wall-clock (s)",
            "Throughput (tok/s)",
            "Per-req tok/s (median)",
            "Per-req tok/s (p95)",
        )
        logger.info("-" * 100)
        for concurrency in sorted(set(levels)):
            await self._run_concurrency_level(model_id, prompt, concurrency)
        logger.info("-" * 100)
        logger.info("=== Concurrency Test Complete ===")

    async def _run_concurrency_level(
        self,
        model_id: str,
        prompt: str,
        concurrency: int,
    ) -> None:
        async def timed_run(request: InferenceRequest):
            started = time.perf_counter()
            result = await self._run_foreground(request)
            return result, max(time.perf_counter() - started, 1e-9)

        requests = [
            InferenceRequest(
                model_id=model_id,
                prompt=prompt,
                max_output_tokens=self.settings.startup_self_test_max_output_tokens,
                temperature=self.settings.default_temperature,
                top_p=self.settings.default_top_p,
                stream=False,
            )
            for _ in range(concurrency)
        ]
        started = time.perf_counter()
        timed = await asyncio.gather(*(timed_run(request) for request in requests))
        elapsed = max(time.perf_counter() - started, 1e-9)
        rates = [result.completion_tokens / latency for result, latency in timed]
        ordered = sorted(rates)
        p95_index = max(0, int(round(0.95 * (len(ordered) - 1))))
        logger.info(
            "%-7d  %-9d  %-15.2f  %-20.2f  %-22.2f  %-22.2f",
            concurrency,
            concurrency,
            elapsed,
            sum(result.completion_tokens for result, _ in timed) / elapsed,
            statistics.median(rates),
            ordered[p95_index],
        )

    def _clear_task(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            if self._task is task:
                self._task = None


def _preview_text(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return "%s..." % text[: limit - 3]
