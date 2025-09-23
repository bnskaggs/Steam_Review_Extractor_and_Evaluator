"""Semantic retrieval utilities."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Sequence

from math import sqrt
try:  # pragma: no cover - offline fallback
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover
    from ._openai_stub import OpenAI

from .config import get_settings
from .db import Database

logger = logging.getLogger(__name__)

_RERANKER = None


def get_reranker(model_name: str):  # pragma: no cover - heavy dependency
    global _RERANKER
    if model_name is None:
        return None
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        _RERANKER = CrossEncoder(model_name)
    return _RERANKER


def embed_query(text: str, client: OpenAI | None = None, model: str | None = None) -> list[float]:
    settings = get_settings()
    client = client or OpenAI(api_key=settings.openai_api_key)
    model = model or settings.embed_model
    response = client.embeddings.create(model=model, input=[text])
    return response.data[0].embedding


def _vector_search(
    db: Database,
    query_vector: Sequence[float],
    app_id: str,
    lang: str,
    start: datetime | None,
    end: datetime | None,
    limit: int,
) -> list[dict]:
    if db.backend == "postgres":
        sql = (
            "SELECT review_id, review_text, review_url, helpful_count, funny_count, created_at,"
            " 1 - (embedding <#> %(vec)s) AS score"
            " FROM reviews"
            " WHERE app_id = %(app_id)s AND lang = %(lang)s AND embedding IS NOT NULL"
        )
        params: dict[str, object] = {
            "vec": query_vector,
            "app_id": app_id,
            "lang": lang,
        }
        if start is not None:
            sql += " AND created_at >= %(start)s"
            params["start"] = start
        if end is not None:
            sql += " AND created_at <= %(end)s"
            params["end"] = end
        sql += " ORDER BY embedding <#> %(vec)s, helpful_count DESC LIMIT %(limit)s"
        params["limit"] = limit
        rows = db.fetchall(sql, params)
        return rows
    # sqlite fallback - fetch candidates and compute similarity in python
    sql = (
        "SELECT review_id, review_text, review_url, helpful_count, funny_count, created_at, embedding"
        " FROM reviews WHERE app_id = ? AND lang = ? AND embedding IS NOT NULL"
    )
    params = [app_id, lang]
    if start is not None:
        sql += " AND created_at >= ?"
        params.append(start)
    if end is not None:
        sql += " AND created_at <= ?"
        params.append(end)
    rows = db.fetchall(sql, params)
    results = []
    q = [float(x) for x in query_vector]
    for row in rows:
        emb = row.get("embedding")
        if isinstance(emb, str):
            emb_vec = [float(x) for x in json.loads(emb)]
        else:
            emb_vec = [float(x) for x in (emb or [])]
        if not emb_vec:
            continue
        score = _cosine_similarity(q, emb_vec)
        row = dict(row)
        row["score"] = float(score)
        results.append(row)
    results.sort(key=lambda item: (-item.get("score", 0.0), -(item.get("helpful_count") or 0)))
    return results[:limit]


def _bm25_candidates(db: Database, query: str, app_id: str, lang: str, limit: int) -> list[dict]:
    words = [w for w in query.split() if len(w) > 2]
    if not words:
        return []
    if db.backend == "postgres":
        sql = (
            "SELECT review_id, review_text, review_url, helpful_count, funny_count, created_at"
            " FROM reviews"
            " WHERE app_id = %(app_id)s AND lang = %(lang)s"
            " AND to_tsvector('english', review_text) @@ plainto_tsquery(%(q)s)"
            " ORDER BY helpful_count DESC LIMIT %(limit)s"
        )
        params = {"app_id": app_id, "lang": lang, "q": " ".join(words), "limit": limit}
        return db.fetchall(sql, params)
    # sqlite fallback using LIKE
    clauses = ["review_text LIKE ?" for _ in words]
    sql = (
        "SELECT review_id, review_text, review_url, helpful_count, funny_count, created_at"
        f" FROM reviews WHERE app_id = ? AND lang = ? AND ({' OR '.join(clauses)})"
        " ORDER BY helpful_count DESC LIMIT ?"
    )
    params = [app_id, lang] + [f"%{w}%" for w in words] + [limit]
    return db.fetchall(sql, params)


def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    numerator = sum(a * b for a, b in zip(vec1, vec2))
    denom = sqrt(sum(a * a for a in vec1)) * sqrt(sum(b * b for b in vec2))
    if denom == 0:
        return 0.0
    return float(numerator / denom)


def _apply_topic_filters(
    db: Database,
    candidates: list[dict],
    topics: Sequence[str] | None,
    sentiments: Sequence[str] | None,
) -> list[dict]:
    if not candidates:
        return []
    if not topics and not sentiments:
        return candidates
    ids = [item["review_id"] for item in candidates]
    if db.backend == "postgres":
        sql = "SELECT review_id, topic, sentiment FROM review_topics WHERE review_id = ANY(%(ids)s)"
        rows = db.fetchall(sql, {"ids": ids})
    else:
        placeholders = ",".join("?" for _ in ids)
        sql = f"SELECT review_id, topic, sentiment FROM review_topics WHERE review_id IN ({placeholders})"
        rows = db.fetchall(sql, ids)
    by_review: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        by_review.setdefault(row["review_id"], []).append((row["topic"], row["sentiment"]))
    filtered: list[dict] = []
    for item in candidates:
        labels = by_review.get(item["review_id"], [])
        if not labels:
            if topics:
                continue
            if sentiments:
                continue
            filtered.append(item)
            continue
        if topics:
            labels = [label for label in labels if label[0] in topics]
            if not labels:
                continue
        if sentiments:
            if not any(label[1] in sentiments for label in labels):
                continue
        filtered.append(item)
    return filtered


def semantic_search(
    query: str,
    app_id: str,
    lang: str = "english",
    start: datetime | None = None,
    end: datetime | None = None,
    k: int = 64,
    use_reranker: bool = False,
    topics: Sequence[str] | None = None,
    sentiments: Sequence[str] | None = None,
    db: Database | None = None,
) -> list[dict]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for retrieval")

    db = db or Database()
    query_vector = embed_query(query)
    vector_hits = _vector_search(db, query_vector, app_id, lang, start, end, k)
    bm25_hits = _bm25_candidates(db, query, app_id, lang, k // 2)

    combined: dict[str, dict] = {}
    for source, hits in (("vector", vector_hits), ("bm25", bm25_hits)):
        for rank, item in enumerate(hits):
            review_id = item["review_id"]
            existing = combined.get(review_id)
            score = item.get("score", 0.0) or 0.0
            if existing:
                existing["score"] = max(existing["score"], score)
            else:
                combined[review_id] = {
                    **item,
                    "score": score if score else max(0.0, 1 - rank * 0.01),
                }
    candidates = list(combined.values())
    candidates.sort(key=lambda item: (-item.get("score", 0.0), -(item.get("helpful_count") or 0)))
    candidates = candidates[: max(k, 16)]
    candidates = _apply_topic_filters(db, candidates, topics, sentiments)

    if use_reranker and candidates:
        reranker = get_reranker(settings.rerank_model)
        if reranker is not None:
            pairs = [(query, item["review_text"]) for item in candidates]
            scores = reranker.predict(pairs)
            for item, score in zip(candidates, scores):
                item["score"] = float(score)
            candidates.sort(key=lambda item: (-item["score"], -(item.get("helpful_count") or 0)))
            candidates = candidates[: min(len(candidates), 16)]
    return candidates[:k]


def fetch_reviews_by_ids(db: Database, review_ids: Sequence[str]) -> list[dict]:
    if not review_ids:
        return []
    if db.backend == "postgres":
        sql = "SELECT review_id, review_text, review_url, helpful_count, funny_count, created_at FROM reviews WHERE review_id = ANY(%(ids)s)"
        rows = db.fetchall(sql, {"ids": list(review_ids)})
    else:
        placeholders = ",".join("?" for _ in review_ids)
        sql = f"SELECT review_id, review_text, review_url, helpful_count, funny_count, created_at FROM reviews WHERE review_id IN ({placeholders})"
        rows = db.fetchall(sql, review_ids)
    return rows


def counts_by_topic(
    app_id: str,
    topic: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    min_helpful: int = 0,
    group_by_month: bool = False,
    db: Database | None = None,
) -> list[dict]:
    db = db or Database()
    params: dict[str, object] = {"app_id": app_id, "min_helpful": min_helpful}
    clauses = ["r.app_id = %(app_id)s", "r.lang = 'english'", "r.helpful_count >= %(min_helpful)s"]
    if topic:
        clauses.append("t.topic = %(topic)s")
        params["topic"] = topic
    if start is not None:
        clauses.append("r.created_at >= %(start)s")
        params["start"] = start
    if end is not None:
        clauses.append("r.created_at <= %(end)s")
        params["end"] = end
    if group_by_month:
        bucket = "date_trunc('month', r.created_at)"
    else:
        bucket = "NULL"
    if db.backend == "postgres":
        where_clause = " AND ".join(clauses)
        sql = (
            "SELECT {bucket} AS bucket, t.topic, t.sentiment, COUNT(*) AS count "
            "FROM review_topics t "
            "JOIN reviews r ON r.review_id = t.review_id "
            "WHERE {where_clause} "
            "GROUP BY bucket, t.topic, t.sentiment ORDER BY bucket NULLS FIRST"
        ).format(bucket=bucket, where_clause=where_clause)
        rows = db.fetchall(sql, params)
    else:
        bucket_expr = "strftime('%Y-%m-01', r.created_at)" if group_by_month else "NULL"
        clauses_sql: list[str] = ["r.app_id = ?", "r.lang = 'english'", "r.helpful_count >= ?"]
        params_list: list = [app_id, min_helpful]
        if topic:
            clauses_sql.append("t.topic = ?")
            params_list.append(topic)
        if start is not None:
            clauses_sql.append("r.created_at >= ?")
            params_list.append(start)
        if end is not None:
            clauses_sql.append("r.created_at <= ?")
            params_list.append(end)
        sql = (
            f"SELECT {bucket_expr} AS bucket, t.topic, t.sentiment, COUNT(*) AS count "
            "FROM review_topics t JOIN reviews r ON r.review_id = t.review_id "
            "WHERE " + " AND ".join(clauses_sql)
            + " GROUP BY bucket, t.topic, t.sentiment ORDER BY bucket"
        )
        rows = db.fetchall(sql, params_list)
    return rows


__all__ = [
    "semantic_search",
    "counts_by_topic",
    "fetch_reviews_by_ids",
    "embed_query",
]
