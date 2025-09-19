"""Minimal FastAPI stub for offline test environments."""
from __future__ import annotations

from typing import Any, Callable, Optional


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str | None = None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail or ""


class Depends:  # pragma: no cover - trivial shim
    def __init__(self, dependency: Callable[..., Any]):
        self.dependency = dependency


class Query:  # pragma: no cover - trivial shim
    def __init__(self, default: Any = None, description: str | None = None, pattern: str | None = None):
        self.default = default
        self.description = description
        self.pattern = pattern


class ORJSONResponse:
    def __init__(self, content: Any):
        self.content = content


class CORSMiddleware:  # pragma: no cover - configuration only
    def __init__(self, app: "FastAPI", **kwargs: Any):
        self.app = app
        self.kwargs = kwargs


class FastAPI:
    def __init__(self, default_response_class: type | None = None, title: str | None = None, version: str | None = None):
        self.default_response_class = default_response_class
        self.title = title
        self.version = version
        self._events: dict[str, list[Callable[..., Any]]] = {}

    def add_middleware(self, middleware_class: type, **kwargs: Any):  # pragma: no cover - no-op
        return None

    def on_event(self, event: str):
        def decorator(func: Callable[..., Any]):
            self._events.setdefault(event, []).append(func)
            return func

        return decorator

    def post(self, path: str, response_model: type | None = None):
        def decorator(func: Callable[..., Any]):
            return func

        return decorator

    def get(self, path: str, response_model: type | None = None):
        def decorator(func: Callable[..., Any]):
            return func

        return decorator


__all__ = [
    "FastAPI",
    "Depends",
    "HTTPException",
    "Query",
    "ORJSONResponse",
    "CORSMiddleware",
]
