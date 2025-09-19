from __future__ import annotations

import importlib
import sys
from datetime import datetime
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def api_env(tmp_path, monkeypatch):
    db_path = tmp_path / "counts.db"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")
    monkeypatch.setenv("EMBED_MODEL", "embed-test")
    monkeypatch.setenv("DB_DSN", f"sqlite:///{db_path}")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "")

    from src import config

    config.get_settings.cache_clear()

    if "src.retrieval" in importlib.sys.modules:
        importlib.reload(importlib.sys.modules["src.retrieval"])
    else:
        importlib.import_module("src.retrieval")

    if "src.rag_api" in importlib.sys.modules:
        importlib.reload(importlib.sys.modules["src.rag_api"])
    else:
        importlib.import_module("src.rag_api")

    from src import rag_api

    rag_api.db_instance.run_schema()
    yield rag_api
    rag_api.db_instance.close()


def test_counts_endpoint(api_env):
    rag_api = api_env
    db = rag_api.db_instance

    base = datetime(2024, 1, 15)
    rows = []
    for idx in range(6):
        review_id = f"c{idx}"
        rows.append(
            {
                "review_id": review_id,
                "app_id": "app-1",
                "lang": "english",
                "created_at": base.replace(month=((idx % 3) + 1)),
                "recommended": True,
                "playtime_hours": 5.0,
                "helpful_count": 10 + idx,
                "funny_count": 0,
                "purchase_type": "steam",
                "review_url": None,
                "review_text": "Sample text",
                "embedding": None,
            }
        )
    db.upsert_reviews(rows)

    sentiments = ["positive", "negative", "very_negative"]
    topic_rows = []
    for idx, row in enumerate(rows):
        topic = "netcode" if idx % 2 == 0 else "story"
        topic_rows.append(
            {
                "review_id": row["review_id"],
                "topic": topic,
                "sentiment": sentiments[idx % len(sentiments)],
                "confidence": 0.9,
            }
        )
    db.upsert_topics(topic_rows)

    counts = rag_api.counts_endpoint(
        app_id="app-1",
        topic="netcode",
        min_helpful=10,
        group_by="month",
        db=db,
    )
    assert counts, "expected counts result"
    entry = counts[0]
    assert entry.topic == "netcode"
    assert entry.total > 0
    assert "negative" in entry.by_sentiment
    assert entry.buckets, "expected monthly buckets"
