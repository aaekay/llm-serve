from __future__ import annotations

import logging
import signal
import sys

import uvicorn

from llm_serve.app import create_app
from llm_serve.config import Settings


def _configure_app_logging() -> None:
    log = logging.getLogger("llm_serve")
    if log.handlers:
        return
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.propagate = False


def main() -> None:
    _configure_app_logging()
    log = logging.getLogger("llm_serve")
    settings = Settings.load()

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
