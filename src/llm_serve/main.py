from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from dataclasses import replace
from pathlib import Path

import uvicorn

from llm_serve.app import create_app
from llm_serve.config import Settings
from llm_serve.runtime.autotune import AutoTunedVLLMBackend


def _configure_app_logging() -> None:
    log = logging.getLogger("llm_serve")
    if log.handlers:
        return
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.propagate = False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-serve")
    subcommands = parser.add_subparsers(dest="command")
    tune = subcommands.add_parser("tune", help="benchmark and persist a vLLM throughput profile")
    tune.add_argument("--force", action="store_true", help="ignore a matching saved profile")
    tune.add_argument("--workload", type=Path, help="JSON workload distribution")
    tune.add_argument("--max-concurrency", type=int)
    tune.add_argument("--repetitions", type=int)
    return parser


async def _run_tune(settings: Settings, args: argparse.Namespace) -> None:
    if settings.inference_backend != "vllm":
        raise ValueError("'llm-serve tune' requires INFERENCE_BACKEND=vllm")
    settings = replace(
        settings,
        vllm_gpu_mode="auto",
        vllm_autotune_workload_file=(args.workload.resolve() if args.workload else settings.vllm_autotune_workload_file),
        vllm_autotune_max_concurrency=args.max_concurrency or settings.vllm_autotune_max_concurrency,
        vllm_autotune_repetitions=args.repetitions or settings.vllm_autotune_repetitions,
    )
    settings.validate()
    backend = AutoTunedVLLMBackend(
        settings.effective_default_model_id,
        settings,
        force_tune=args.force,
    )
    try:
        await backend.start()
        print(json.dumps(backend.tuning_snapshot, indent=2, sort_keys=True))
    finally:
        await backend.shutdown()


def main() -> None:
    _configure_app_logging()
    args = _build_parser().parse_args()
    log = logging.getLogger("llm_serve")
    settings = Settings.load()

    if args.command == "tune":
        asyncio.run(_run_tune(settings, args))
        return

    # Track whether a shutdown signal has been received so we only act once.
    # The first SIGINT/SIGTERM lets uvicorn shut down gracefully (lifespan
    # cleanup runs, vLLM workers are killed).  A second signal force-exits.
    _shutting_down = False

    def _shutdown_handler(signum, frame):
        nonlocal _shutting_down
        sig_name = signal.Signals(signum).name
        if _shutting_down:
            log.warning("Received %s again during shutdown — forcing exit.", sig_name)
            sys.exit(1)
        _shutting_down = True
        log.info("Received %s, shutting down gracefully…", sig_name)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
    )
