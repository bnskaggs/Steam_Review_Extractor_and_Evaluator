"""FastAPI application exposing search, ask and counts endpoints."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

try:  # pragma: no cover - fallback for offline tests
    from fastapi import Depends, FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import ORJSONResponse
except ModuleNotFoundError:  # pragma: no cover
    from ._fastapi_stub import Depends, FastAPI, HTTPException, Query, CORSMiddleware, ORJSONResponse
try:  # pragma: no cover - offline fallback
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover
    from ._openai_stub import OpenAI
try:  # pragma: no cover - offline fallback
    from pydantic import BaseModel, field_validator
except ModuleNotFoundError:  # pragma: no cover
    class BaseModel:  # type: ignore
        def __init__(self, **data):
            for name, value in self.__class__.__dict__.items():
                if name.startswith("_") or callable(value):
                    continue
                if name not in data:
                    setattr(self, name, value)
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self):
            return self.__dict__.copy()

    def field_validator(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator

from .config import get_settings
from .db import Database
from .prompts import build_rag_messages
from .retrieval import counts_by_topic, semantic_search
from .utils import dedupe_texts, select_top_snippets

logger = logging.getLogger(__name__)

settings = get_settings()
db_instance = Database()
app = FastAPI(default_response_class=ORJSONResponse, title="Steam RAG API", version="1.0.0")

if settings.cors_allow_origins:
    allow_credentials = "*" not in settings.cors_allow_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class ReviewDoc(BaseModel):
    review_id: str
    review_text: str
    review_url: Optional[str] = None
    helpful_count: int = 0
    funny_count: int = 0
    created_at: datetime
    score: float | None = None

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


class SearchRequest(BaseModel):
    query: str
    app_id: str
    lang: str = "english"
    date_start: datetime | None = None
    date_end: datetime | None = None
    k: int = 20
    rerank: bool = False
    topics: List[str] | None = None
    sentiments: List[str] | None = None

    @field_validator("date_start", "date_end", mode="before")
    @classmethod
    def _coerce_dates(cls, value):
        if value in (None, ""):
            return None
        return value

    @field_validator("topics", "sentiments", mode="before")
    @classmethod
    def _coerce_list(cls, value):
        if value in (None, ""):
            return None
        if isinstance(value, list):
            return value
        return [value]


class SearchResponse(BaseModel):
    results: List[ReviewDoc]


class AskRequest(BaseModel):
    query: str
    app_id: str
    date_start: datetime | None = None
    date_end: datetime | None = None
    k: int = 64
    rerank: bool = False
    topics: List[str] | None = None
    sentiments: List[str] | None = None

    @field_validator("date_start", "date_end", mode="before")
    @classmethod
    def _coerce_dates(cls, value):
        if value in (None, ""):
            return None
        return value

    @field_validator("topics", "sentiments", mode="before")
    @classmethod
    def _coerce_list(cls, value):
        if value in (None, ""):
            return None
        if isinstance(value, list):
            return value
        return [value]


class AskResponse(BaseModel):
    answer: str
    citations: List[str]
    snippets: List[ReviewDoc]


class CountsBucket(BaseModel):
    bucket: Optional[str] = None
    total: int
    by_sentiment: dict[str, int]


class CountsResponse(BaseModel):
    topic: Optional[str]
    total: int
    by_sentiment: dict[str, int]
    buckets: List[CountsBucket] | None = None


def get_db() -> Database:
    return db_instance


@app.on_event("startup")
async def ensure_schema():  # pragma: no cover - executed in runtime env
    db_instance.run_schema()


def _serialize_review(row: dict) -> ReviewDoc:
    return ReviewDoc(
        review_id=row["review_id"],
        review_text=row.get("review_text", ""),
        review_url=row.get("review_url"),
        helpful_count=row.get("helpful_count") or 0,
        funny_count=row.get("funny_count") or 0,
        created_at=row.get("created_at"),
        score=row.get("score"),
    )


def _unwrap_query(value):
    return getattr(value, "default", value)


@app.post("/search", response_model=SearchResponse)
def search_reviews(request: SearchRequest, db: Database = Depends(get_db)):
    hits = semantic_search(
        query=request.query,
        app_id=request.app_id,
        lang=request.lang,
        start=request.date_start,
        end=request.date_end,
        k=request.k,
        use_reranker=request.rerank,
        topics=request.topics,
        sentiments=request.sentiments,
        db=db,
    )
    docs = [_serialize_review(row) for row in hits]
    return SearchResponse(results=docs)


def _prepare_snippets(hits: list[dict]) -> list[dict]:
    text_pairs = [(item["review_id"], item.get("review_text", "")) for item in hits]
    unique_pairs = dedupe_texts(text_pairs)
    ordered_ids = [review_id for review_id, _ in unique_pairs]
    filtered: list[dict] = []
    seen: set[str] = set()
    for review_id in ordered_ids:
        for item in hits:
            if item["review_id"] == review_id and review_id not in seen:
                filtered.append(item)
                seen.add(review_id)
                break
    ranked = select_top_snippets([(item["review_id"], item) for item in filtered], limit=12)
    snippets: list[dict] = []
    for review_id, payload in ranked:
        snippet = dict(payload)
        snippet["review_id"] = review_id
        snippets.append(snippet)
    return snippets


def _generate_answer(query: str, snippets: list[dict], date_range: tuple[datetime | None, datetime | None]):
    client = OpenAI(api_key=settings.openai_api_key)
    messages = build_rag_messages(query, snippets, settings.llm_model, date_range=date_range)
    response = client.chat.completions.create(model=settings.llm_model, temperature=0.2, messages=messages)
    return response.choices[0].message.content.strip()


@app.post("/ask", response_model=AskResponse)
def ask_reviews(request: AskRequest, db: Database = Depends(get_db)):
    hits = semantic_search(
        query=request.query,
        app_id=request.app_id,
        start=request.date_start,
        end=request.date_end,
        k=request.k,
        use_reranker=request.rerank,
        topics=request.topics,
        sentiments=request.sentiments,
        db=db,
    )
    if not hits:
        raise HTTPException(status_code=404, detail="No reviews found for the given filters")
    snippets = _prepare_snippets(hits)
    answer = _generate_answer(request.query, snippets, (request.date_start, request.date_end))
    citations = [snippet["review_id"] for snippet in snippets]
    docs = [_serialize_review(snippet) for snippet in snippets]
    return AskResponse(answer=answer, citations=citations, snippets=docs)


@app.get("/counts", response_model=List[CountsResponse])
def counts_endpoint(
    app_id: str = Query(..., description="Steam app identifier"),
    topic: str | None = Query(default=None),
    date_start: datetime | None = Query(default=None),
    date_end: datetime | None = Query(default=None),
    min_helpful: int = Query(default=0),
    group_by: str | None = Query(default=None, pattern="^(month)?$"),
    db: Database = Depends(get_db),
):
    topic = _unwrap_query(topic)
    date_start = _unwrap_query(date_start)
    date_end = _unwrap_query(date_end)
    min_helpful = _unwrap_query(min_helpful)
    group_by = _unwrap_query(group_by)
    rows = counts_by_topic(
        app_id=app_id,
        topic=topic,
        start=date_start,
        end=date_end,
        min_helpful=min_helpful,
        group_by_month=group_by == "month",
        db=db,
    )
    if not rows:
        return [
            CountsResponse(
                topic=topic,
                total=0,
                by_sentiment={},
                buckets=[] if group_by == "month" else None,
            )
        ]
    topics_map: dict[str | None, CountsResponse] = {}
    for row in rows:
        topic_key = row.get("topic")
        entry = topics_map.setdefault(
            topic_key,
            CountsResponse(topic=topic_key, total=0, by_sentiment={}, buckets=[] if group_by == "month" else None),
        )
        sentiment = row.get("sentiment")
        count = int(row.get("count") or 0)
        entry.total += count
        entry.by_sentiment[sentiment] = entry.by_sentiment.get(sentiment, 0) + count
        if group_by == "month":
            bucket_value = row.get("bucket")
            if isinstance(bucket_value, datetime):
                bucket_str = bucket_value.date().isoformat()
            elif bucket_value is None:
                bucket_str = None
            else:
                bucket_str = str(bucket_value)
            buckets = entry.buckets or []
            bucket_entry = next((b for b in buckets if b.bucket == bucket_str), None)
            if bucket_entry is None:
                bucket_entry = CountsBucket(bucket=bucket_str, total=0, by_sentiment={})
                buckets.append(bucket_entry)
            bucket_entry.total += count
            bucket_entry.by_sentiment[sentiment] = bucket_entry.by_sentiment.get(sentiment, 0) + count
            entry.buckets = buckets
    results = list(topics_map.values())
    if group_by == "month":
        for entry in results:
            if entry.buckets:
                entry.buckets.sort(key=lambda b: b.bucket or "")
    results.sort(key=lambda item: item.topic or "")
    return results


__all__ = ["app"]
