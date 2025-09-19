from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def api_env(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
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


def test_smoke_flow(api_env, monkeypatch):
    rag_api = api_env
    db = rag_api.db_instance

    now = datetime.utcnow()
    rows = []
    for idx in range(20):
        review_id = f"r{idx}"
        text = "Netcode lag is terrible" if idx % 2 == 0 else "Story mode is amazing"
        rows.append(
            {
                "review_id": review_id,
                "app_id": "app-1",
                "lang": "english",
                "created_at": now - timedelta(days=idx),
                "recommended": idx % 3 == 0,
                "playtime_hours": float(idx),
                "helpful_count": idx * 2,
                "funny_count": idx,
                "purchase_type": "steam",
                "review_url": f"https://example.com/{review_id}",
                "review_text": text,
                "embedding": None,
            }
        )
    db.upsert_reviews(rows)

    for idx, row in enumerate(rows):
        review_id = row["review_id"]
        vector = [1.0, 0.0, 0.0] if "Netcode" in row["review_text"] else [0.0, 1.0, 0.0]
        db.execute(
            "UPDATE reviews SET embedding = ? WHERE review_id = ?",
            (json.dumps(vector), review_id),
        )
        db.upsert_embedding_meta(
            [{"review_id": review_id, "text_checksum": f"chk-{review_id}"}]
        )
        topic = "netcode" if idx % 2 == 0 else "story"
        sentiment = "negative" if topic == "netcode" else "positive"
        db.upsert_topics(
            [
                {
                    "review_id": review_id,
                    "topic": topic,
                    "sentiment": sentiment,
                    "confidence": 0.9,
                }
            ]
        )

    from src import retrieval

    def fake_embed_query(text: str, client=None, model=None):
        return [1.0, 0.0, 0.0] if "netcode" in text.lower() else [0.0, 1.0, 0.0]

    monkeypatch.setattr(retrieval, "embed_query", fake_embed_query)
    monkeypatch.setattr(rag_api, "_generate_answer", lambda q, s, d: f"• Summary [{s[0]['review_id']}]" if s else "")

    ask_request = rag_api.AskRequest(
        query="netcode issues",
        app_id="app-1",
        k=10,
        topics=["netcode"],
        sentiments=["negative"],
    )
    ask_response = rag_api.ask_reviews(ask_request, db=db)
    assert ask_response.citations, "expected citations"
    assert f"[{ask_response.citations[0]}]" in ask_response.answer

    search_request = rag_api.SearchRequest(query="story", app_id="app-1", k=5, topics=["story"])
    search_response = rag_api.search_reviews(search_request, db=db)
    assert search_response.results, "search should return results"
