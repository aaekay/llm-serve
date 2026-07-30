from __future__ import annotations

from llm_serve.config import Settings
from llm_serve.errors import BadRequestError
from llm_serve.prompting import render_messages_to_prompt
from llm_serve.schemas import OllamaChatRequest, OllamaGenerateRequest, OpenAIChatRequest
from llm_serve.tokenization import estimate_messages_tokens, estimate_text_tokens
from llm_serve.types import InferenceRequest


def build_openai_chat_request(
    payload: OpenAIChatRequest,
    model_id: str,
    settings: Settings,
    *,
    stream: bool | None = None,
    timeout_retry_enabled: bool = True,
) -> InferenceRequest:
    messages = [message.model_dump() for message in payload.messages]
    _validate_input_tokens(estimate_messages_tokens(messages), settings.max_input_tokens)
    return InferenceRequest(
        model_id=model_id,
        prompt=render_messages_to_prompt(payload.messages),
        max_output_tokens=min(
            payload.resolved_max_tokens(settings.max_output_tokens),
            settings.max_output_tokens,
        ),
        temperature=_default(payload.temperature, settings.default_temperature),
        top_p=_default(payload.top_p, settings.default_top_p),
        stream=payload.stream if stream is None else stream,
        reasoning_effort=payload.reasoning_effort,
        include_reasoning=payload.include_reasoning,
        messages=messages,
        timeout_retry_enabled=timeout_retry_enabled,
    )


def build_ollama_chat_request(
    payload: OllamaChatRequest,
    model_id: str,
    settings: Settings,
) -> InferenceRequest:
    messages = [message.model_dump() for message in payload.messages]
    _validate_input_tokens(estimate_messages_tokens(messages), settings.max_input_tokens)
    options = payload.options
    return InferenceRequest(
        model_id=model_id,
        prompt=render_messages_to_prompt(payload.messages),
        max_output_tokens=_ollama_output_tokens(payload, settings),
        temperature=_default(options.temperature if options else None, settings.default_temperature),
        top_p=_default(options.top_p if options else None, settings.default_top_p),
        stream=payload.stream,
        reasoning_effort=options.reasoning_effort if options else None,
        include_reasoning=bool(options.include_reasoning) if options else False,
        messages=messages,
    )


def build_ollama_generate_request(
    payload: OllamaGenerateRequest,
    model_id: str,
    settings: Settings,
) -> InferenceRequest:
    _validate_input_tokens(estimate_text_tokens(payload.prompt), settings.max_input_tokens)
    options = payload.options
    return InferenceRequest(
        model_id=model_id,
        prompt=payload.prompt,
        max_output_tokens=_ollama_output_tokens(payload, settings),
        temperature=_default(options.temperature if options else None, settings.default_temperature),
        top_p=_default(options.top_p if options else None, settings.default_top_p),
        stream=payload.stream,
        reasoning_effort=options.reasoning_effort if options else None,
        include_reasoning=bool(options.include_reasoning) if options else False,
    )


def _validate_input_tokens(estimated_tokens: int, max_input_tokens: int) -> None:
    if estimated_tokens > max_input_tokens:
        raise BadRequestError(
            "Prompt too long: estimated %s tokens, max is %s"
            % (estimated_tokens, max_input_tokens)
        )


def _ollama_output_tokens(
    payload: OllamaChatRequest | OllamaGenerateRequest,
    settings: Settings,
) -> int:
    options = payload.options
    if options is None or options.num_predict is None:
        return settings.max_output_tokens
    return min(options.num_predict, settings.max_output_tokens)


def _default(value, default):
    return default if value is None else value
