from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from llm_serve.config import Settings
from llm_serve.errors import (
    NotReadyError,
    UpstreamAPIError,
    UpstreamRuntimeError,
    UpstreamTimeoutError,
)
from llm_serve.tokenization import estimate_text_tokens
from llm_serve.types import InferenceRequest, InferenceResult

from .base import ModelBackend

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


logger = logging.getLogger(__name__)


class LlamaCppAPIClient:
    """Small client for a separately managed llama-server OpenAI API."""

    def __init__(
        self,
        settings: Settings,
        transport: Optional["httpx.AsyncBaseTransport"] = None,
    ) -> None:
        if httpx is None:  # pragma: no cover
            raise NotReadyError(
                "httpx is not installed. Install project dependencies before using "
                "INFERENCE_BACKEND=llamacpp."
            )
        headers = {}
        if settings.llamacpp_api_key:
            headers["Authorization"] = "Bearer %s" % settings.llamacpp_api_key
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.llamacpp_base_url + "/",
            headers=headers,
            timeout=settings.request_timeout_seconds,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> List[Dict[str, Any]]:
        payload = await self._request_json("GET", "models")
        models = payload.get("data")
        if not isinstance(models, list):
            raise NotReadyError(
                "llama-server at %s returned an invalid model list."
                % self._settings.llamacpp_base_url
            )
        return [model for model in models if isinstance(model, dict)]

    async def assert_model_available(self, model_id: str) -> Dict[str, Any]:
        for model in await self.list_models():
            if model.get("id") == model_id:
                return model
        raise NotReadyError(
            "llama-server does not expose model alias '%s' at %s. Start it with "
            "--alias '%s', or set LLAMACPP_DEFAULT_MODEL_ID and MODEL_ALLOWLIST "
            "to the id returned by GET /v1/models."
            % (model_id, self._settings.llamacpp_base_url, model_id)
        )

    async def openai_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request_json("POST", "chat/completions", payload)

    async def openai_chat_stream(
        self,
        payload: Dict[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        async for event in self._stream_json("chat/completions", payload):
            yield event

    async def complete(self, request: InferenceRequest, *, chat: bool) -> InferenceResult:
        path = "chat/completions" if chat else "completions"
        payload = self._inference_payload(request, chat=chat, stream=False)
        response = await self._request_json("POST", path, payload)
        text, reasoning = self._completion_text(response, chat=chat)
        usage = response.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return InferenceResult(
            model_id=request.model_id,
            text=text,
            prompt_tokens=(
                prompt_tokens
                if isinstance(prompt_tokens, int)
                else estimate_text_tokens(request.prompt)
            ),
            completion_tokens=(
                completion_tokens
                if isinstance(completion_tokens, int)
                else estimate_text_tokens(text)
            ),
            reasoning=reasoning,
        )

    async def complete_stream(
        self,
        request: InferenceRequest,
        *,
        chat: bool,
    ) -> AsyncIterator[str]:
        path = "chat/completions" if chat else "completions"
        payload = self._inference_payload(request, chat=chat, stream=True)
        async for event in self._stream_json(path, payload):
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            if chat:
                delta = choice.get("delta")
                chunk = delta.get("content") if isinstance(delta, dict) else None
            else:
                chunk = choice.get("text")
            if isinstance(chunk, str) and chunk:
                yield chunk

    async def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            kwargs: Dict[str, Any] = {}
            if payload is not None:
                kwargs["json"] = payload
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            value = response.json()
        except httpx.HTTPStatusError as exc:
            raise self._upstream_api_error(exc) from exc
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(
                "Timed out waiting for llama-server at %s."
                % self._settings.llamacpp_base_url
            ) from exc
        except httpx.RequestError as exc:
            raise UpstreamRuntimeError(
                "Failed to reach llama-server at %s: %s"
                % (self._settings.llamacpp_base_url, exc)
            ) from exc
        except ValueError as exc:
            raise UpstreamRuntimeError(
                "llama-server at %s returned invalid JSON."
                % self._settings.llamacpp_base_url
            ) from exc
        if not isinstance(value, dict):
            raise UpstreamRuntimeError(
                "llama-server at %s returned an invalid JSON payload."
                % self._settings.llamacpp_base_url
            )
        return value

    async def _stream_json(
        self,
        path: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        try:
            async with self._client.stream("POST", path, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        return
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise UpstreamRuntimeError(
                            "llama-server at %s returned invalid streaming JSON."
                            % self._settings.llamacpp_base_url
                        ) from exc
                    if isinstance(item, dict):
                        yield item
        except httpx.HTTPStatusError as exc:
            await exc.response.aread()
            raise self._upstream_api_error(exc) from exc
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(
                "Timed out streaming from llama-server at %s."
                % self._settings.llamacpp_base_url
            ) from exc
        except httpx.RequestError as exc:
            raise UpstreamRuntimeError(
                "llama-server stream at %s failed: %s"
                % (self._settings.llamacpp_base_url, exc)
            ) from exc

    def _upstream_api_error(self, exc: "httpx.HTTPStatusError") -> UpstreamAPIError:
        try:
            payload = exc.response.json()
        except ValueError:
            payload = {
                "error": {
                    "message": exc.response.text or "llama-server rejected the request",
                    "type": "upstream_error",
                }
            }
        if not isinstance(payload, dict):
            payload = {
                "error": {
                    "message": "llama-server rejected the request",
                    "type": "upstream_error",
                }
            }
        return UpstreamAPIError(exc.response.status_code, payload)

    @staticmethod
    def _completion_text(
        payload: Dict[str, Any],
        *,
        chat: bool,
    ) -> tuple[str, Optional[str]]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise UpstreamRuntimeError("llama-server returned no completion choice.")
        choice = choices[0]
        if not chat:
            text = choice.get("text")
            if not isinstance(text, str):
                raise UpstreamRuntimeError("llama-server returned an invalid text completion.")
            return text, None
        message = choice.get("message")
        if not isinstance(message, dict):
            raise UpstreamRuntimeError("llama-server returned an invalid chat completion.")
        text = message.get("content")
        reasoning = message.get("reasoning_content")
        return (
            text if isinstance(text, str) else "",
            reasoning if isinstance(reasoning, str) else None,
        )

    @staticmethod
    def _inference_payload(
        request: InferenceRequest,
        *,
        chat: bool,
        stream: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": request.model_id,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if chat:
            payload["messages"] = request.messages or [
                {"role": "user", "content": request.prompt}
            ]
            if request.reasoning_effort:
                payload["chat_template_kwargs"] = {
                    "reasoning_strength": request.reasoning_effort
                }
        else:
            payload["prompt"] = request.prompt
        return payload


class LlamaCppModelBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        settings: Settings,
        client: Optional[LlamaCppAPIClient] = None,
    ) -> None:
        super().__init__(model_id)
        self._client = client or LlamaCppAPIClient(settings)

    async def start(self) -> None:
        logger.info("Checking llama-server model alias '%s'.", self.model_id)
        await self._client.assert_model_available(self.model_id)
        logger.info("llama-server model alias '%s' is ready.", self.model_id)

    async def shutdown(self) -> None:
        await self._client.close()

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        return await self._client.complete(request, chat=False)

    async def generate_chat(self, request: InferenceRequest) -> InferenceResult:
        return await self._client.complete(request, chat=True)

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        async for chunk in self._client.complete_stream(request, chat=False):
            yield chunk

    async def generate_chat_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        async for chunk in self._client.complete_stream(request, chat=True):
            yield chunk

    async def openai_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._client.openai_chat(payload)

    async def openai_chat_stream(
        self,
        payload: Dict[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        async for event in self._client.openai_chat_stream(payload):
            yield event
