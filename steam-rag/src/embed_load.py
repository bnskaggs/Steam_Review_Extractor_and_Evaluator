"""Populate pgvector embeddings for reviews."""
from __future__ import annotations

import argparse
import json
import logging
from typing import Sequence

try:  # pragma: no cover - offline fallback
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover
    from ._openai_stub import OpenAI

from .config import get_settings
from .db import Database
from .utils import chunked, hash_text

logger = logging.getLogger(__name__)

BATCH_SIZE = 128


def _fetch_candidates(db: Database, limit: int | None = None) -> list[dict]:
    params: Sequence = ()
    limit_clause = ""
    if limit:
        limit_clause = " LIMIT %s" if db.backend != "sqlite" else " LIMIT ?"
        params = (limit,)
    sql = (
        "SELECT r.review_id, r.review_text, r.embedding, e.text_checksum "
        "FROM reviews r LEFT JOIN review_embeddings e ON e.review_id = r.review_id "
        "WHERE r.lang = 'english' ORDER BY r.updated_at ASC" + limit_clause
    )
    rows = db.fetchall(sql, params if params else None)
    candidates: list[dict] = []
    for row in rows:
        text: str = row.get("review_text") or ""
        if len(text.strip()) < 10:
            continue
        digest = hash_text(text)
        if row.get("embedding") is None:
            candidates.append({"review_id": row["review_id"], "text": text, "checksum": digest})
        elif row.get("text_checksum") != digest:
            candidates.append({"review_id": row["review_id"], "text": text, "checksum": digest})
    return candidates


def batch_embed_texts(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in response.data]
    return vectors


def _store_embeddings(db: Database, payloads: list[tuple[str, list[float], str]]):
    if not payloads:
        return
    if db.backend == "sqlite":
        sql = "UPDATE reviews SET embedding = ? WHERE review_id = ?"
        with db.cursor() as cur:
            for review_id, vector, _checksum in payloads:
                cur.execute(sql, (json.dumps(vector), review_id))
    else:
        sql = "UPDATE reviews SET embedding = %s WHERE review_id = %s"
        params = [(vector, review_id) for review_id, vector, _checksum in payloads]
        db.executemany(sql, params)
    meta = [
        {"review_id": review_id, "text_checksum": checksum}
        for review_id, _vector, checksum in payloads
    ]
    db.upsert_embedding_meta(meta)


def embed(limit: int | None = None) -> int:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embedding")

    db = Database()
    db.run_schema()

    candidates = _fetch_candidates(db, limit=limit)
    if not candidates:
        logger.info("No reviews require embedding")
        return 0

    client = OpenAI(api_key=settings.openai_api_key)
    processed = 0

    for batch in chunked(candidates, BATCH_SIZE):
        texts = [item["text"] for item in batch]
        vectors = batch_embed_texts(client, texts, settings.embed_model)
        payloads = [
            (item["review_id"], vector, item["checksum"])
            for item, vector in zip(batch, vectors)
        ]
        _store_embeddings(db, payloads)
        processed += len(batch)
        logger.info("Embedded %s / %s", processed, len(candidates))

    return processed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed reviews with OpenAI")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for testing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    total = embed(limit=args.limit)
    logger.info("Embedded %s reviews", total)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
