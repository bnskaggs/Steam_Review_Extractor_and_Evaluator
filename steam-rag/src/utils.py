"""Utility helpers shared across the code base."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Iterable, Iterator, List, Sequence, Tuple, TypeVar

try:  # pragma: no cover - offline fallback
    from dateutil import parser
except ModuleNotFoundError:  # pragma: no cover
    from datetime import datetime as _dt

    class _Parser:
        @staticmethod
        def parse(text: str):
            try:
                return _dt.fromisoformat(text)
            except ValueError:
                return None

    parser = _Parser()  # type: ignore

logger = logging.getLogger(__name__)

T = TypeVar("T")


def chunked(iterable: Iterable[T], size: int) -> Iterator[List[T]]:
    """Yield lists of *size* from *iterable* until it is exhausted."""

    batch: List[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_bool(value) -> bool | None:
    """Parse truthy strings into booleans."""

    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    return None


def safe_int(value, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_float(value, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def parse_datetime(value) -> datetime | None:
    """Parse a timestamp or return ``None`` when parsing fails."""

    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except (OverflowError, ValueError):
            return None
    text = str(value)
    try:
        return parser.parse(text)
    except (parser.ParserError, ValueError, TypeError):
        return None


def normalize_text(value: str | None) -> str | None:
    """Return trimmed text or ``None`` for empty inputs."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedupe_texts(items: Sequence[Tuple[str, str]], threshold: float = 0.95) -> List[Tuple[str, str]]:
    """Remove near duplicate review texts.

    The implementation keeps the first occurrence of a review id/text pair and
    removes later entries whose hashes match exactly. When we need more advanced
    deduplication we can plug in SimHash or cosine similarity in this function
    without touching callers.
    """

    seen: set[str] = set()
    unique: List[Tuple[str, str]] = []
    for review_id, text in items:
        digest = hash_text(text)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append((review_id, text))
    return unique


def select_top_snippets(snippets: Sequence[Tuple[str, dict]], limit: int = 10) -> List[Tuple[str, dict]]:
    """Return top ``limit`` snippets prioritising helpful reviews."""

    sorted_snippets = sorted(
        snippets,
        key=lambda pair: (
            -pair[1].get("helpful_count", 0),
            -pair[1].get("score", 0.0),
            pair[1].get("created_at"),
        ),
    )
    seen_ids: set[str] = set()
    deduped: List[Tuple[str, dict]] = []
    for review_id, payload in sorted_snippets:
        if review_id in seen_ids:
            continue
        seen_ids.add(review_id)
        deduped.append((review_id, payload))
        if len(deduped) >= limit:
            break
    return deduped


__all__ = [
    "chunked",
    "parse_bool",
    "safe_int",
    "safe_float",
    "parse_datetime",
    "normalize_text",
    "hash_text",
    "dedupe_texts",
    "select_top_snippets",
]
