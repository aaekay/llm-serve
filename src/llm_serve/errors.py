from __future__ import annotations

from typing import Any, Dict, Optional


class ServiceError(Exception):
    status_code = 500
    error_type = "server_error"

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code or self.status_code
        self.error_type = error_type or self.error_type
        self.extra = extra or {}

    def to_openai_error(self) -> Dict[str, Any]:
        payload = {
            "error": {
                "message": self.message,
                "type": self.error_type,
            }
        }
        payload["error"].update(self.extra)
        return payload

    def to_ollama_error(self) -> Dict[str, Any]:
        payload = {"error": self.message}
        payload.update(self.extra)
        return payload


class BadRequestError(ServiceError):
    status_code = 400
    error_type = "invalid_request_error"


class ConflictError(ServiceError):
    status_code = 409
    error_type = "conflict_error"


class NotFoundError(ServiceError):
    status_code = 404
    error_type = "not_found_error"


class TooManyRequestsError(ServiceError):
    status_code = 429
    error_type = "rate_limit_error"


class NotReadyError(ServiceError):
    status_code = 503
    error_type = "service_unavailable_error"


class GatewayTimeoutError(ServiceError):
    status_code = 504
    error_type = "gateway_timeout_error"


class UpstreamTimeoutError(ServiceError):
    status_code = 504
    error_type = "upstream_timeout_error"


class UpstreamRuntimeError(ServiceError):
    status_code = 502
    error_type = "upstream_error"
