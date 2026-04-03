from __future__ import annotations

import logging
import sys

import uvicorn

from llm_serve.app import create_app
from llm_serve.config import Settings


def _configure_app_logging() -> None:
    """Uvicorn only configures its own loggers; without this, llm_serve INFO (e.g. startup self-test output) is hidden."""
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
    settings = Settings.load()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
    )
