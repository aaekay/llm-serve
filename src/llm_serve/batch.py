from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from llm_serve.errors import BadRequestError
from llm_serve.prompting import render_messages_to_prompt
from llm_serve.schemas import BatchCreateRequest, BatchInputLine, BatchRecord, OpenAIChatRequest
from llm_serve.storage import StorageManager, utc_timestamp
from llm_serve.tokenization import estimate_messages_tokens
from llm_serve.types import InferenceRequest

from .runtime.manager import RuntimeManager


class BatchManager:
    def __init__(self, storage: StorageManager, runtime: RuntimeManager) -> None:
        self.storage = storage
        self.runtime = runtime
        self._tasks: Dict[str, "asyncio.Task[None]"] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def startup(self) -> None:
        for batch in self.storage.list_batches():
            if batch.status in {"queued", "validating"}:
                self.enqueue(batch.id)
            elif batch.status in {"in_progress", "cancelling"}:
                batch.status = "failed"
                batch.completed_at = utc_timestamp()
                batch.error = {"message": "Batch interrupted by process restart"}
                self.storage.save_batch(batch)

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        for batch_id, task in list(self._tasks.items()):
            try:
                await task
            except asyncio.CancelledError:
                batch = self.storage.get_batch(batch_id)
                batch.status = "cancelled"
                batch.completed_at = utc_timestamp()
                self.storage.save_batch(batch)

    def create_batch(self, request: BatchCreateRequest) -> BatchRecord:
        if request.endpoint != "/v1/chat/completions":
            raise BadRequestError("Batch endpoint must be /v1/chat/completions")
        input_record = self.storage.get_file(request.input_file_id)
        if input_record.purpose != "batch":
            raise BadRequestError("Input file purpose must be 'batch'")
        if not self.storage.get_file_content(request.input_file_id).decode("utf-8").strip():
            raise BadRequestError("Batch input file must not be empty")

        batch = self.storage.create_batch(
            input_file_id=request.input_file_id,
            endpoint=request.endpoint,
            completion_window=request.completion_window,
        )
        self.enqueue(batch.id)
        return batch

    def enqueue(self, batch_id: str) -> None:
        if batch_id in self._tasks and not self._tasks[batch_id].done():
            return
        self._locks[batch_id] = self._locks.get(batch_id, asyncio.Lock())
        self._tasks[batch_id] = asyncio.create_task(self._run(batch_id))

    async def cancel(self, batch_id: str) -> BatchRecord:
        batch = self.storage.get_batch(batch_id)
        if batch.status in {"completed", "failed", "cancelled"}:
            return batch
        batch.status = "cancelling"
        batch.cancelling_at = utc_timestamp()
        self.storage.save_batch(batch)
        task = self._tasks.get(batch_id)
        if task is not None:
            task.cancel()
        return batch

    async def _run(self, batch_id: str) -> None:
        batch = self.storage.get_batch(batch_id)
        batch.status = "validating"
        self.storage.save_batch(batch)
        input_content = self.storage.get_file_content(batch.input_file_id).decode("utf-8")
        raw_lines = [line for line in input_content.splitlines() if line.strip()]
        batch.request_counts.total = len(raw_lines)
        batch.status = "in_progress"
        batch.in_progress_at = utc_timestamp()
        self.storage.save_batch(batch)

        tasks = []
        for line in raw_lines:
            tasks.append(asyncio.create_task(self._process_line(batch_id, line)))

        try:
            await asyncio.gather(*tasks)
            batch = self.storage.get_batch(batch_id)
            if batch.status == "cancelling":
                batch.status = "cancelled"
            elif batch.request_counts.failed and not batch.request_counts.completed:
                batch.status = "failed"
            else:
                batch.status = "completed"
            batch.completed_at = utc_timestamp()
            self.storage.save_batch(batch)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            batch = self.storage.get_batch(batch_id)
            batch.status = "cancelled"
            batch.completed_at = utc_timestamp()
            self.storage.save_batch(batch)
            raise

    async def _process_line(self, batch_id: str, raw_line: str) -> None:
        batch = self.storage.get_batch(batch_id)
        if batch.status == "cancelling":
            return
        lock = self._locks[batch_id]
        try:
            parsed = BatchInputLine.model_validate(json.loads(raw_line))
            if parsed.method.upper() != "POST":
                raise BadRequestError("Batch line method must be POST")
            if parsed.url != "/v1/chat/completions":
                raise BadRequestError("Batch line url must be /v1/chat/completions")
            if parsed.body.get("stream") is True:
                raise BadRequestError("Batch requests do not support stream=true")

            openai_request = OpenAIChatRequest.model_validate(parsed.body)
            model_id = self.runtime.resolve_model(openai_request.model, openai_request.reasoning_effort)
            estimated_tokens = estimate_messages_tokens([message.model_dump() for message in openai_request.messages])
            if estimated_tokens > self.runtime.settings.max_input_tokens:
                raise BadRequestError(
                    "Prompt too long: estimated %s tokens, max is %s"
                    % (estimated_tokens, self.runtime.settings.max_input_tokens)
                )
            inference_request = InferenceRequest(
                model_id=model_id,
                prompt=render_messages_to_prompt(openai_request.messages),
                max_output_tokens=min(
                    openai_request.resolved_max_tokens(self.runtime.settings.max_output_tokens),
                    self.runtime.settings.max_output_tokens,
                ),
                temperature=openai_request.temperature
                if openai_request.temperature is not None
                else self.runtime.settings.default_temperature,
                top_p=openai_request.top_p if openai_request.top_p is not None else self.runtime.settings.default_top_p,
                stream=False,
                reasoning_effort=openai_request.reasoning_effort,
                include_reasoning=openai_request.include_reasoning,
            )
            result = await self.runtime.run_batch(inference_request)
            response_body = _build_batch_response_body(
                request_id="chatcmpl-" + uuid.uuid4().hex[:12],
                model_id=result.model_id,
                text=result.text,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                reasoning=result.reasoning if openai_request.include_reasoning else None,
            )
            output_line = json.dumps(
                {
                    "custom_id": parsed.custom_id,
                    "response": {
                        "status_code": 200,
                        "request_id": response_body["id"],
                        "body": response_body,
                    },
                    "error": None,
                }
            ) + "\n"
            async with lock:
                self.storage.append_file_content(batch.output_file_id, output_line)
                batch = self.storage.get_batch(batch_id)
                batch.request_counts.completed += 1
                self.storage.save_batch(batch)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_line = json.dumps(
                {
                    "custom_id": _extract_custom_id(raw_line),
                    "error": {"message": str(exc)},
                }
            ) + "\n"
            async with lock:
                batch = self.storage.get_batch(batch_id)
                self.storage.append_file_content(batch.error_file_id, error_line)
                batch.request_counts.failed += 1
                self.storage.save_batch(batch)


def _extract_custom_id(raw_line: str) -> Optional[str]:
    try:
        payload = json.loads(raw_line)
        return payload.get("custom_id")
    except Exception:
        return None


def _build_batch_response_body(
    request_id: str,
    model_id: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning: Optional[str],
) -> Dict[str, object]:
    message = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning"] = reasoning
    created = int(datetime.now(timezone.utc).timestamp())
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
