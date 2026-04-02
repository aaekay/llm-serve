from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from llm_serve.config import Settings
from llm_serve.errors import NotReadyError, UpstreamRuntimeError, UpstreamTimeoutError
from llm_serve.tokenization import estimate_text_tokens
from llm_serve.types import InferenceRequest, InferenceResult

from .base import ModelBackend

try:
    import httpx
except ImportError:  # pragma: no cover - runtime dependency guard
    httpx = None


class OllamaAPIClient:
    def __init__(self, settings: Settings, transport: Optional["httpx.AsyncBaseTransport"] = None) -> None:
        if httpx is None:  # pragma: no cover - runtime dependency guard
            raise NotReadyError(
                "httpx is not installed. Install the project dependencies before using INFERENCE_BACKEND=ollama."
            )
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=None,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> List[Dict[str, Any]]:
        response = await self._request(
            "GET",
            "/api/tags",
            timeout=self._settings.request_timeout_seconds,
            action="list models from",
            error_cls=NotReadyError,
        )
        payload = self._parse_json(response, action="list models from", error_cls=NotReadyError)
        models = payload.get("models")
        if not isinstance(models, list):
            raise NotReadyError(
                "Ollama at %s returned an invalid model list response." % self._settings.ollama_base_url
            )
        return models

    async def assert_model_installed(self, model_id: str) -> Dict[str, Any]:
        for model in await self.list_models():
            if self.extract_model_name(model) == model_id:
                return model
        raise NotReadyError(
            "Ollama model '%s' is not installed at %s. Pull it with /api/pull before requesting it."
            % (model_id, self._settings.ollama_base_url)
        )

    async def pull_model(self, model_id: str) -> Dict[str, Any]:
        response = await self._request(
            "POST",
            "/api/pull",
            json_body={"model": model_id, "stream": False},
            timeout=self._settings.switch_timeout_seconds,
            action="pull model '%s' from" % model_id,
            error_cls=NotReadyError,
        )
        return self._parse_json(
            response,
            action="pull model '%s' from" % model_id,
            error_cls=NotReadyError,
        )

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        timeouts = self._request_timeouts_for_request(request)
        for attempt_index, timeout_seconds in enumerate(timeouts):
            try:
                response = await self._request(
                    "POST",
                    "/api/generate",
                    json_body=self._build_generate_payload(request, stream=False),
                    timeout=timeout_seconds,
                    action="generate text from",
                    error_cls=UpstreamRuntimeError,
                    timeout_error_cls=UpstreamTimeoutError,
                )
            except UpstreamTimeoutError:
                if attempt_index == len(timeouts) - 1:
                    raise
                continue

            payload = self._parse_json(response, action="generate text from", error_cls=UpstreamRuntimeError)
            response_text = payload.get("response")
            if not isinstance(response_text, str):
                raise UpstreamRuntimeError(
                    "Ollama at %s returned an invalid generation payload." % self._settings.ollama_base_url
                )
            prompt_tokens = payload.get("prompt_eval_count")
            completion_tokens = payload.get("eval_count")
            reasoning = payload.get("thinking")
            return InferenceResult(
                model_id=request.model_id,
                text=response_text,
                prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else estimate_text_tokens(request.prompt),
                completion_tokens=completion_tokens
                if isinstance(completion_tokens, int)
                else estimate_text_tokens(response_text),
                reasoning=reasoning if isinstance(reasoning, str) else None,
            )

        raise UpstreamTimeoutError("Timed out while generating from Ollama at %s." % self._settings.ollama_base_url)

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        payload = self._build_generate_payload(request, stream=True)
        timeouts = self._request_timeouts_for_request(request)
        for attempt_index, timeout_seconds in enumerate(timeouts):
            emitted_chunk = False
            try:
                async with self._client.stream(
                    "POST",
                    "/api/generate",
                    json=payload,
                    timeout=timeout_seconds,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise UpstreamRuntimeError(
                                "Ollama at %s returned invalid streaming JSON." % self._settings.ollama_base_url
                            ) from exc
                        if item.get("error"):
                            raise UpstreamRuntimeError(str(item["error"]))
                        chunk = item.get("response")
                        if isinstance(chunk, str) and chunk:
                            emitted_chunk = True
                            yield chunk
                        if item.get("done") is True:
                            return
                return
            except httpx.HTTPStatusError as exc:
                raise UpstreamRuntimeError(
                    "Ollama request to %s failed with status %s: %s"
                    % (self._settings.ollama_base_url, exc.response.status_code, exc.response.text)
                ) from exc
            except httpx.TimeoutException as exc:
                if emitted_chunk or attempt_index == len(timeouts) - 1:
                    raise UpstreamTimeoutError(
                        "Timed out while streaming from Ollama at %s." % self._settings.ollama_base_url
                    ) from exc
                continue
            except httpx.HTTPError as exc:
                raise UpstreamRuntimeError(
                    "Failed to reach Ollama at %s: %s" % (self._settings.ollama_base_url, exc)
                ) from exc

    @staticmethod
    def extract_model_name(model: Dict[str, Any]) -> str:
        raw_name = model.get("model") or model.get("name") or ""
        return raw_name if isinstance(raw_name, str) else ""

    async def _request(
        self,
        method: str,
        path: str,
        timeout: float,
        action: str,
        error_cls,
        timeout_error_cls=None,
        json_body: Optional[Dict[str, Any]] = None,
    ):
        timeout_error_cls = timeout_error_cls or error_cls
        try:
            response = await self._client.request(method, path, json=json_body, timeout=timeout)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            raise error_cls(
                "Ollama request to %s failed with status %s while trying to %s: %s"
                % (
                    self._settings.ollama_base_url,
                    exc.response.status_code,
                    action,
                    exc.response.text,
                )
            ) from exc
        except httpx.TimeoutException as exc:
            raise timeout_error_cls(
                "Timed out while trying to %s Ollama at %s."
                % (action, self._settings.ollama_base_url)
            ) from exc
        except httpx.HTTPError as exc:
            raise error_cls(
                "Failed to reach Ollama at %s while trying to %s: %s"
                % (self._settings.ollama_base_url, action, exc)
            ) from exc

    def _parse_json(self, response, action: str, error_cls) -> Dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise error_cls(
                "Ollama at %s returned invalid JSON while trying to %s."
                % (self._settings.ollama_base_url, action)
            ) from exc
        if not isinstance(payload, dict):
            raise error_cls(
                "Ollama at %s returned an invalid JSON payload while trying to %s."
                % (self._settings.ollama_base_url, action)
            )
        if payload.get("error"):
            raise error_cls(str(payload["error"]))
        return payload

    def _build_generate_payload(self, request: InferenceRequest, stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": request.model_id,
            "prompt": request.prompt,
            "stream": stream,
            "options": {
                "num_predict": request.max_output_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
        }
        if request.reasoning_effort:
            payload["think"] = request.reasoning_effort
        elif request.include_reasoning:
            payload["think"] = True
        return payload

    def _request_timeouts_for_request(self, request: InferenceRequest) -> List[float]:
        base_timeout = request.upstream_timeout_seconds or self._settings.ollama_request_timeout_seconds
        timeouts = [base_timeout]
        if request.timeout_retry_enabled and self._settings.ollama_request_timeout_retry_enabled:
            timeouts.append(base_timeout * self._settings.ollama_request_timeout_retry_multiplier)
        return timeouts


class OllamaModelBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        settings: Settings,
        client_factory: Optional[Callable[[Settings], OllamaAPIClient]] = None,
    ) -> None:
        super().__init__(model_id)
        self._settings = settings
        self._client_factory = client_factory or OllamaAPIClient
        self._client: Optional[OllamaAPIClient] = None

    async def start(self) -> None:
        client = self._get_client()
        await client.assert_model_installed(self.model_id)

    async def shutdown(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.close()

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        return await self._get_client().generate(request)

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        async for chunk in self._get_client().generate_stream(request):
            yield chunk

    def _get_client(self) -> OllamaAPIClient:
        if self._client is None:
            self._client = self._client_factory(self._settings)
        return self._client
