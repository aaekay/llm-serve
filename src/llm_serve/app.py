from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from llm_serve.batch import BatchManager
from llm_serve.config import Settings
from llm_serve.errors import BadRequestError, ConflictError, ServiceError
from llm_serve.prompting import render_messages_to_prompt
from llm_serve.schemas import (
    BatchCreateRequest,
    OllamaChatRequest,
    OllamaGenerateRequest,
    OpenAIChatRequest,
    PullRequest,
)
from llm_serve.storage import StorageManager
from llm_serve.tokenization import estimate_messages_tokens, estimate_text_tokens
from llm_serve.types import InferenceRequest, InferenceResult, LoadStatus

from .runtime.manager import RuntimeManager


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.load()
    storage = StorageManager(settings.storage_root)
    runtime = RuntimeManager(settings)
    batch_manager = BatchManager(storage, runtime)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await runtime.startup()
        await batch_manager.startup()
        try:
            yield
        finally:
            await batch_manager.shutdown()
            await runtime.shutdown()

    app = FastAPI(title="llm-serve", lifespan=lifespan)
    app.state.settings = settings
    app.state.storage = storage
    app.state.runtime = runtime
    app.state.batch_manager = batch_manager

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=exc.status_code, content=exc.to_ollama_error())
        return JSONResponse(status_code=exc.status_code, content=exc.to_openai_error())

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        detail = "; ".join(error["msg"] for error in exc.errors())
        service_error = BadRequestError(detail or "Invalid request payload")
        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=service_error.status_code, content=service_error.to_ollama_error())
        return JSONResponse(status_code=service_error.status_code, content=service_error.to_openai_error())

    @app.get("/healthz")
    async def healthz():
        snapshot = runtime.health_snapshot()
        batch_total, batch_in_progress = batch_manager.batch_counts()
        return {
            "status": "ok",
            "loaded": snapshot["loaded"],
            "model_id": snapshot["model_id"],
            "inference_backend": snapshot["inference_backend"],
            "foreground_active": snapshot["foreground_active"],
            "foreground_capacity": snapshot["foreground_capacity"],
            "queue_depth": snapshot["queue_depth"],
            "batch_jobs_total": batch_total,
            "batch_jobs_in_progress": batch_in_progress,
            "batch_active": snapshot["batch_active"],
            "batch_capacity": snapshot["batch_capacity"],
            "batch_queue_depth": snapshot["batch_queue_depth"],
            "switch_in_progress": snapshot["switch_in_progress"],
            "switch_target_model": snapshot["switch_target_model"],
            "startup_self_test": snapshot["startup_self_test"],
        }

    @app.get("/v1/models")
    async def list_openai_models():
        return {"object": "list", "data": runtime.list_models()}

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(payload: OpenAIChatRequest):
        model_id = runtime.resolve_model(payload.model, payload.reasoning_effort)
        prompt_tokens = estimate_messages_tokens([message.model_dump() for message in payload.messages])
        _validate_input_tokens(prompt_tokens, settings.max_input_tokens)
        output_tokens = min(payload.resolved_max_tokens(settings.max_output_tokens), settings.max_output_tokens)
        request_model = InferenceRequest(
            model_id=model_id,
            prompt=render_messages_to_prompt(payload.messages),
            max_output_tokens=output_tokens,
            temperature=payload.temperature if payload.temperature is not None else settings.default_temperature,
            top_p=payload.top_p if payload.top_p is not None else settings.default_top_p,
            stream=payload.stream,
            reasoning_effort=payload.reasoning_effort,
            include_reasoning=payload.include_reasoning,
            messages=[message.model_dump() for message in payload.messages],
        )

        load_status = runtime.check_readiness(model_id)
        if not load_status.ready:
            if load_status.state == "conflict":
                raise ConflictError("Runtime is switching models", extra={"current_model": load_status.current_model})
            return _load_pending_json_response(load_status, _openai_load_response(load_status))

        if payload.stream:
            stream = _openai_stream(runtime, request_model)
            return StreamingResponse(stream, media_type="text/event-stream")

        result = await runtime.run_foreground(request_model)
        return _openai_completion_response(result, payload.include_reasoning)

    @app.post("/v1/files")
    async def create_file(purpose: str = Form(...), file: UploadFile = File(...)):
        content = await file.read()
        record = storage.create_file(file.filename or "upload.jsonl", purpose, content)
        return record.model_dump()

    @app.get("/v1/files/{file_id}")
    async def get_file(file_id: str):
        return storage.get_file(file_id).model_dump()

    @app.get("/v1/files/{file_id}/content")
    async def get_file_content(file_id: str):
        content = storage.get_file_content(file_id)
        return PlainTextResponse(content.decode("utf-8"))

    @app.post("/v1/batches")
    async def create_batch(payload: BatchCreateRequest):
        batch = batch_manager.create_batch(payload)
        return batch.model_dump(exclude_none=True)

    @app.get("/v1/batches/{batch_id}")
    async def get_batch(batch_id: str):
        return storage.get_batch(batch_id).model_dump(exclude_none=True)

    @app.post("/v1/batches/{batch_id}/cancel")
    async def cancel_batch(batch_id: str):
        batch = await batch_manager.cancel(batch_id)
        return batch.model_dump(exclude_none=True)

    @app.post("/api/chat")
    async def ollama_chat(payload: OllamaChatRequest):
        options = payload.options
        reasoning_effort = options.reasoning_effort if options is not None else None
        model_id = runtime.resolve_model(payload.model, reasoning_effort)
        prompt_tokens = estimate_messages_tokens([message.model_dump() for message in payload.messages])
        _validate_input_tokens(prompt_tokens, settings.max_input_tokens)
        output_tokens = _resolve_ollama_output_tokens(payload, settings.max_output_tokens)

        request_model = InferenceRequest(
            model_id=model_id,
            prompt=render_messages_to_prompt(payload.messages),
            max_output_tokens=output_tokens,
            temperature=_resolve_ollama_temperature(payload, settings.default_temperature),
            top_p=_resolve_ollama_top_p(payload, settings.default_top_p),
            stream=payload.stream,
            reasoning_effort=reasoning_effort,
            include_reasoning=bool(options.include_reasoning) if options is not None else False,
            messages=[message.model_dump() for message in payload.messages],
        )

        load_status = runtime.check_readiness(model_id)
        if not load_status.ready:
            if load_status.state == "conflict":
                raise ConflictError("Runtime is switching models", extra={"current_model": load_status.current_model})
            return _load_pending_json_response(load_status, _ollama_load_response(load_status))

        if payload.stream:
            return StreamingResponse(_ollama_chat_stream(runtime, request_model), media_type="application/x-ndjson")

        result = await runtime.run_foreground(request_model)
        return _ollama_chat_response(result, request_model.include_reasoning)

    @app.post("/api/generate")
    async def ollama_generate(payload: OllamaGenerateRequest):
        options = payload.options
        reasoning_effort = options.reasoning_effort if options is not None else None
        model_id = runtime.resolve_model(payload.model, reasoning_effort)
        prompt_tokens = estimate_text_tokens(payload.prompt)
        _validate_input_tokens(prompt_tokens, settings.max_input_tokens)

        request_model = InferenceRequest(
            model_id=model_id,
            prompt=payload.prompt,
            max_output_tokens=_resolve_ollama_output_tokens(payload, settings.max_output_tokens),
            temperature=_resolve_ollama_temperature(payload, settings.default_temperature),
            top_p=_resolve_ollama_top_p(payload, settings.default_top_p),
            stream=payload.stream,
            reasoning_effort=reasoning_effort,
            include_reasoning=bool(options.include_reasoning) if options is not None else False,
        )

        load_status = runtime.check_readiness(model_id)
        if not load_status.ready:
            if load_status.state == "conflict":
                raise ConflictError("Runtime is switching models", extra={"current_model": load_status.current_model})
            return _load_pending_json_response(load_status, _ollama_load_response(load_status))

        if payload.stream:
            return StreamingResponse(
                _ollama_generate_stream(runtime, request_model),
                media_type="application/x-ndjson",
            )

        result = await runtime.run_foreground(request_model)
        return _ollama_generate_response(result, request_model.include_reasoning)

    @app.get("/api/tags")
    async def ollama_tags():
        return {"models": await runtime.list_tags()}

    @app.post("/api/pull")
    async def ollama_pull(payload: PullRequest):
        model_id = runtime.resolve_model(payload.name)
        status = await runtime.pull_model(model_id)
        if status.ready:
            return {"status": "ready", "model": model_id}
        return _load_pending_json_response(status, _ollama_load_response(status))

    return app


def _validate_input_tokens(estimated_tokens: int, max_input_tokens: int) -> None:
    if estimated_tokens > max_input_tokens:
        raise BadRequestError("Prompt too long: estimated %s tokens, max is %s" % (estimated_tokens, max_input_tokens))


def _openai_load_response(load_status: LoadStatus):
    return {
        "object": "model.load",
        "status": "spinning_up",
        "model": load_status.model_id,
        "current_model": load_status.current_model,
        "retry_after_seconds": load_status.retry_after_seconds,
        "eta_seconds": None,
    }


def _ollama_load_response(load_status: LoadStatus):
    return {
        "status": "spinning_up",
        "model": load_status.model_id,
        "current_model": load_status.current_model,
        "retry_after_seconds": load_status.retry_after_seconds,
    }


def _load_pending_json_response(load_status: LoadStatus, payload: dict) -> JSONResponse:
    return JSONResponse(
        status_code=202,
        content=payload,
        headers={"Retry-After": str(load_status.retry_after_seconds)},
    )


def _openai_completion_response(result: InferenceResult, include_reasoning: bool) -> JSONResponse:
    created = int(datetime.now(timezone.utc).timestamp())
    message: dict = {"role": "assistant", "content": result.text}
    if include_reasoning and result.reasoning:
        message["reasoning"] = result.reasoning
    payload = {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": created,
        "model": result.model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        },
    }
    return JSONResponse(content=payload)


async def _openai_stream(runtime: RuntimeManager, request_model: InferenceRequest) -> AsyncIterator[str]:
    stream_id = "chatcmpl-" + uuid.uuid4().hex[:12]
    created = int(datetime.now(timezone.utc).timestamp())
    role_chunk = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request_model.model_id,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield "data: %s\n\n" % json.dumps(role_chunk)
    async for chunk in runtime.stream_foreground(request_model):
        payload = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request_model.model_id,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        }
        yield "data: %s\n\n" % json.dumps(payload)
    final_payload = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request_model.model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield "data: %s\n\n" % json.dumps(final_payload)
    yield "data: [DONE]\n\n"


def _ollama_chat_response(result: InferenceResult, include_reasoning: bool):
    payload = {
        "model": result.model_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": {"role": "assistant", "content": result.text},
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "eval_count": result.completion_tokens,
        "prompt_eval_count": result.prompt_tokens,
    }
    if include_reasoning and result.reasoning:
        payload["message"]["reasoning"] = result.reasoning
    return JSONResponse(content=payload)


async def _ollama_chat_stream(runtime: RuntimeManager, request_model: InferenceRequest) -> AsyncIterator[str]:
    async for chunk in runtime.stream_foreground(request_model):
        payload = {
            "model": request_model.model_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": {"role": "assistant", "content": chunk},
            "done": False,
        }
        yield json.dumps(payload) + "\n"
    yield json.dumps(
        {
            "model": request_model.model_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        }
    ) + "\n"


def _ollama_generate_response(result: InferenceResult, include_reasoning: bool):
    payload = {
        "model": result.model_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "response": result.text,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": result.prompt_tokens,
        "eval_count": result.completion_tokens,
    }
    if include_reasoning and result.reasoning:
        payload["reasoning"] = result.reasoning
    return JSONResponse(content=payload)


async def _ollama_generate_stream(runtime: RuntimeManager, request_model: InferenceRequest) -> AsyncIterator[str]:
    full_text = ""
    async for chunk in runtime.stream_foreground(request_model):
        full_text += chunk
        payload = {
            "model": request_model.model_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response": chunk,
            "done": False,
        }
        yield json.dumps(payload) + "\n"
    yield json.dumps(
        {
            "model": request_model.model_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response": "",
            "done": True,
            "done_reason": "stop",
            "total_response": full_text,
        }
    ) + "\n"


def _resolve_ollama_output_tokens(payload, default_value: int) -> int:
    options = payload.options
    if options is None or options.num_predict is None:
        return default_value
    return min(options.num_predict, default_value)


def _resolve_ollama_temperature(payload, default_value: float) -> float:
    options = payload.options
    if options is None or options.temperature is None:
        return default_value
    return options.temperature


def _resolve_ollama_top_p(payload, default_value: float) -> float:
    options = payload.options
    if options is None or options.top_p is None:
        return default_value
    return options.top_p
