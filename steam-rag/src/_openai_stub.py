"""Minimal OpenAI client stub for offline tests."""
from __future__ import annotations


class _StubResponse:
    def __init__(self, data=None, content: str = ""):
        self.data = data or []
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})})()]


class OpenAI:  # pragma: no cover - simple stub
    def __init__(self, *args, **kwargs):
        pass

    class embeddings:  # type: ignore
        @staticmethod
        def create(*args, **kwargs):
            raise RuntimeError("OpenAI embeddings stub used without patching")

    class chat:  # type: ignore
        class completions:  # type: ignore
            @staticmethod
            def create(*args, **kwargs):
                raise RuntimeError("OpenAI chat stub used without patching")


__all__ = ["OpenAI"]
