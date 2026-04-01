from __future__ import annotations

import uvicorn

from llm_serve.app import create_app
from llm_serve.config import Settings


def main() -> None:
    settings = Settings.load()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
    )
