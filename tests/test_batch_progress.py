from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Dict

from llm_serve.batch import BatchManager
from llm_serve.runtime.manager import RuntimeManager
from llm_serve.schemas import BatchCreateRequest, BatchRecord
from llm_serve.storage import StorageManager

from .conftest import make_settings


@dataclass
class FakeProgressBar:
    batch_id: str
    total: int
    position: int
    updates: list[int] = field(default_factory=list)
    closed: bool = False

    def update(self, count: int = 1) -> None:
        self.updates.append(count)

    def close(self) -> None:
        self.closed = True


def test_batch_progress_tracks_successes_and_failures(tmp_path):
    settings = make_settings(tmp_path, STARTUP_SELF_TEST_ENABLED="false")
    progress_bars: Dict[str, FakeProgressBar] = {}

    def progress_factory(batch_id: str, total: int, position: int) -> FakeProgressBar:
        bar = FakeProgressBar(batch_id=batch_id, total=total, position=position)
        progress_bars[batch_id] = bar
        return bar

    batch = asyncio.run(
        _run_batch_job(
            settings=settings,
            input_content=_build_batch_line("req-1") + _build_batch_line("req-2", url="/v1/embeddings"),
            progress_factory=progress_factory,
            progress_enabled=True,
        )
    )

    progress_bar = progress_bars[batch.id]
    assert batch.status == "completed"
    assert batch.request_counts.total == 2
    assert batch.request_counts.completed == 1
    assert batch.request_counts.failed == 1
    assert progress_bar.total == 2
    assert progress_bar.position == 0
    assert progress_bar.updates == [1, 1]
    assert progress_bar.closed is True


def test_batch_progress_uses_distinct_positions_for_concurrent_batches(tmp_path):
    settings = make_settings(
        tmp_path,
        STARTUP_SELF_TEST_ENABLED="false",
        BATCH_MAX_PARALLEL="4",
        MOCK_RESPONSE_DELAY_SECONDS="0.05",
    )
    progress_bars: Dict[str, FakeProgressBar] = {}

    def progress_factory(batch_id: str, total: int, position: int) -> FakeProgressBar:
        bar = FakeProgressBar(batch_id=batch_id, total=total, position=position)
        progress_bars[batch_id] = bar
        return bar

    batch_ids = asyncio.run(
        _run_concurrent_batches(
            settings=settings,
            progress_factory=progress_factory,
            progress_enabled=True,
        )
    )

    positions = {progress_bars[batch_id].position for batch_id in batch_ids}
    assert positions == {0, 1}
    assert all(progress_bars[batch_id].closed for batch_id in batch_ids)


def test_batch_progress_falls_back_to_logs_without_live_bar(tmp_path, caplog):
    settings = make_settings(tmp_path, STARTUP_SELF_TEST_ENABLED="false")
    progress_factory_calls = 0

    def progress_factory(batch_id: str, total: int, position: int) -> FakeProgressBar:
        nonlocal progress_factory_calls
        progress_factory_calls += 1
        return FakeProgressBar(batch_id=batch_id, total=total, position=position)

    with caplog.at_level(logging.INFO, logger="llm_serve.batch"):
        batch = asyncio.run(
            _run_batch_job(
                settings=settings,
                input_content=_build_batch_line("req-1"),
                progress_factory=progress_factory,
                progress_enabled=False,
            )
        )

    assert progress_factory_calls == 0
    assert "Batch %s started with 1 items" % batch.id in caplog.text
    assert "Batch %s finished status=completed completed=1 failed=0 total=1" % batch.id in caplog.text


def _build_batch_line(custom_id: str, url: str = "/v1/chat/completions") -> str:
    return json.dumps(
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": url,
            "body": {
                "messages": [{"role": "user", "content": "Say hello from batch"}],
                "stream": False,
            },
        }
    ) + "\n"


async def _run_batch_job(
    settings,
    input_content: str,
    progress_factory,
    progress_enabled: bool,
) -> BatchRecord:
    storage = StorageManager(settings.storage_root)
    runtime = RuntimeManager(settings)
    manager = BatchManager(
        storage,
        runtime,
        progress_factory=progress_factory,
        progress_enabled=progress_enabled,
    )
    await runtime.startup()
    try:
        input_record = storage.create_file("batch.jsonl", "batch", input_content.encode("utf-8"))
        batch = manager.create_batch(
            BatchCreateRequest(
                input_file_id=input_record.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
        )
        task = manager._tasks[batch.id]
        await task
        return storage.get_batch(batch.id)
    finally:
        await manager.shutdown()
        await runtime.shutdown()


async def _run_concurrent_batches(
    settings,
    progress_factory,
    progress_enabled: bool,
) -> list[str]:
    storage = StorageManager(settings.storage_root)
    runtime = RuntimeManager(settings)
    manager = BatchManager(
        storage,
        runtime,
        progress_factory=progress_factory,
        progress_enabled=progress_enabled,
    )
    await runtime.startup()
    try:
        first_input = storage.create_file(
            "batch-one.jsonl",
            "batch",
            (_build_batch_line("req-1") + _build_batch_line("req-2")).encode("utf-8"),
        )
        second_input = storage.create_file(
            "batch-two.jsonl",
            "batch",
            (_build_batch_line("req-3") + _build_batch_line("req-4")).encode("utf-8"),
        )
        first_batch = manager.create_batch(
            BatchCreateRequest(
                input_file_id=first_input.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
        )
        second_batch = manager.create_batch(
            BatchCreateRequest(
                input_file_id=second_input.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
        )
        first_task = manager._tasks[first_batch.id]
        second_task = manager._tasks[second_batch.id]
        await asyncio.gather(first_task, second_task)
        return [first_batch.id, second_batch.id]
    finally:
        await manager.shutdown()
        await runtime.shutdown()
