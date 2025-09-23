"""Application configuration helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List

try:  # pragma: no cover - fallback for offline tests
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False


# Load environment variables from a .env file if present at project root.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


@dataclass
class Settings:
    openai_api_key: str
    db_dsn: str
    llm_model: str = "gpt-5-turbo"
    embed_model: str = "text-embedding-3-small"
    cors_allow_origins: List[str] = field(default_factory=list)
    rerank_model: str | None = None


def _parse_origins(raw: str | None) -> List[str]:
    if not raw:
        return DEFAULT_CORS_ORIGINS.copy()
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        return DEFAULT_CORS_ORIGINS.copy()
    if "*" in items:
        return ["*"]
    return items


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env = os.environ
    openai_api_key = env.get("OPENAI_API_KEY")
    db_dsn = env.get("DB_DSN")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    if not db_dsn:
        raise RuntimeError("DB_DSN is required")
    return Settings(
        openai_api_key=openai_api_key,
        db_dsn=db_dsn,
        llm_model=env.get("LLM_MODEL", "gpt-5-turbo"),
        embed_model=env.get("EMBED_MODEL", "text-embedding-3-small"),
        cors_allow_origins=_parse_origins(env.get("CORS_ALLOW_ORIGINS")),
        rerank_model=env.get("RERANK_MODEL"),
    )


__all__ = ["get_settings", "Settings"]
