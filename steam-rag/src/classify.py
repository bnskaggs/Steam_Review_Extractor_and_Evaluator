"""Topic and sentiment classification pipeline."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from typing import Mapping, Sequence

try:  # pragma: no cover - offline fallback
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover
    from ._openai_stub import OpenAI

from .config import get_settings
from .db import Database
from .prompts import SENTIMENT_LABELS, TOPIC_DEFINITIONS, build_classifier_messages
from .utils import parse_datetime

logger = logging.getLogger(__name__)


def _fetch_reviews(db: Database, since: datetime | None, limit: int | None) -> list[dict]:
    if db.backend == "sqlite":
        params_list: list[object] = []
        filters = ["r.lang = 'english'", "length(trim(r.review_text)) >= 10"]
        if since is not None:
            filters.append("r.updated_at >= ?")
            params_list.append(since)
        sql = (
            "SELECT r.review_id, r.review_text FROM reviews r "
            "WHERE " + " AND ".join(filters) +
            " AND NOT EXISTS (SELECT 1 FROM review_topics t WHERE t.review_id = r.review_id)"
            " ORDER BY r.updated_at ASC"
        )
        if limit:
            sql += " LIMIT ?"
            params_list.append(limit)
        return db.fetchall(sql, params_list or None)
    params: dict[str, object] = {}
    filters = ["r.lang = 'english'", "length(trim(r.review_text)) >= 10"]
    if since is not None:
        filters.append("r.updated_at >= %(since)s")
        params["since"] = since
    sql = (
        "SELECT r.review_id, r.review_text FROM reviews r "
        "WHERE " + " AND ".join(filters) +
        " AND NOT EXISTS (SELECT 1 FROM review_topics t WHERE t.review_id = r.review_id)"
        " ORDER BY r.updated_at ASC"
    )
    if limit:
        sql += " LIMIT %(limit)s"
        params["limit"] = limit
    return db.fetchall(sql, params if params else None)


def _call_classifier(client: OpenAI, reviews: Sequence[Mapping[str, str]], topics: Sequence[str], model: str) -> dict:
    messages = build_classifier_messages(reviews, topics)
    response = client.chat.completions.create(model=model, temperature=0.2, messages=messages)
    content = response.choices[0].message.content
    return _parse_json(content)


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise
        snippet = text[start : end + 1]
        return json.loads(snippet)


def _prepare_topic_rows(review_id: str, payload: Mapping[str, object]) -> list[dict]:
    topics = payload.get("topics") or []
    sentiment_map: Mapping[str, str] = payload.get("sentiment_by_topic") or {}
    confidence_map: Mapping[str, float] = payload.get("confidence_by_topic") or {}
    rows: list[dict] = []
    for topic in topics:
        sentiment = sentiment_map.get(topic, "neutral")
        if sentiment not in SENTIMENT_LABELS:
            sentiment = "neutral"
        confidence = confidence_map.get(topic, 0.0) or 0.0
        if confidence < 0.5:
            sentiment = "neutral"
        rows.append(
            {
                "review_id": review_id,
                "topic": topic,
                "sentiment": sentiment,
                "confidence": float(confidence),
            }
        )
    return rows


def classify_reviews(since: datetime | None = None, limit: int | None = None, dry_run: bool = False) -> int:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for classification")

    db = Database()
    topics = list(TOPIC_DEFINITIONS.keys())
    rows = _fetch_reviews(db, since=since, limit=limit)
    if not rows:
        logger.info("No reviews pending classification")
        return 0

    client = OpenAI(api_key=settings.openai_api_key)
    total = 0
    for batch in [rows[i : i + 5] for i in range(0, len(rows), 5)]:
        data = _call_classifier(client, batch, topics, settings.llm_model)
        reviews_payload = data.get("reviews") or []
        prepared: list[dict] = []
        for item in reviews_payload:
            review_id = item.get("review_id")
            if not review_id:
                continue
            prepared.extend(_prepare_topic_rows(review_id, item))
        if dry_run:
            logger.info("Preview classification: %s", prepared)
        else:
            db.upsert_topics(prepared)
        total += len(prepared)
    logger.info("Classified %s topic rows", total)
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify reviews by topic and sentiment")
    parser.add_argument("--since", type=str, default=None, help="Only classify reviews updated since date")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of reviews")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    since_dt = parse_datetime(args.since) if args.since else None
    classify_reviews(since=since_dt, limit=args.limit, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
