"""CSV ingestion pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .db import Database
from .utils import hash_text, normalize_text, parse_bool, parse_datetime, safe_float, safe_int

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"review_id", "review_text", "lang"}

def _iter_csv_rows(csv_path: Path, limit: int | None = None) -> Iterable[dict]:
    chunksize = 5000
    read_rows = 0
    for chunk in pd.read_csv(
        csv_path,
        dtype=str,
        chunksize=chunksize,
        keep_default_na=False,
        encoding="utf-8",
        on_bad_lines="skip",
        engine="python",
    ):
        for _, row in chunk.iterrows():
            data = row.to_dict()
            read_rows += 1
            yield data
            if limit is not None and read_rows >= limit:
                return


def _fetch_existing_checksums(db: Database, ids: Sequence[str]) -> dict[str, str]:
    if not ids:
        return {}
    if db.backend == "sqlite":
        placeholders = ",".join("?" for _ in ids)
        sql = f"SELECT review_id, text_checksum FROM review_embeddings WHERE review_id IN ({placeholders})"
        rows = db.fetchall(sql, ids)
    else:
        sql = "SELECT review_id, text_checksum FROM review_embeddings WHERE review_id = ANY(%s)"
        rows = db.fetchall(sql, (list(ids),))
    return {row["review_id"]: row["text_checksum"] for row in rows}


def _reset_embeddings(db: Database, review_ids: Sequence[str]):
    if not review_ids:
        return
    if db.backend == "sqlite":
        placeholders = ",".join("?" for _ in review_ids)
        db.execute(f"UPDATE reviews SET embedding=NULL WHERE review_id IN ({placeholders})", review_ids)
        db.execute(f"DELETE FROM review_embeddings WHERE review_id IN ({placeholders})", review_ids)
    else:
        db.execute("UPDATE reviews SET embedding=NULL WHERE review_id = ANY(%s)", (list(review_ids),))
        db.execute("DELETE FROM review_embeddings WHERE review_id = ANY(%s)", (list(review_ids),))


def ingest(csv_path: Path, app_id: str | None = None, limit: int | None = None) -> int:
    db = Database()
    db.run_schema()

    seen_ids: set[str] = set()
    upsert_count = 0
    batch: list[dict] = []

    for row in _iter_csv_rows(csv_path, limit=limit):
        review_id = normalize_text(row.get("review_id"))
        review_text = normalize_text(row.get("review_text"))
        if not review_id or not review_text:
            continue
        if review_id in seen_ids:
            continue
        seen_ids.add(review_id)

        lang = (row.get("lang") or "").lower()
        if lang and lang != "english":
            continue

        created_at = parse_datetime(row.get("created_at"))
        if created_at is None:
            continue

        recommended = parse_bool(row.get("recommended"))
        playtime_hours = safe_float(row.get("playtime_hours"))
        helpful_count = safe_int(row.get("helpful"), default=0) or 0
        funny_count = safe_int(row.get("funny"), default=0) or 0
        purchase_type = normalize_text(row.get("purchase_type"))
        review_url = normalize_text(row.get("review_url"))

        row_app_id = normalize_text(row.get("app_id")) or app_id
        if not row_app_id:
            continue

        batch.append(
            {
                "review_id": review_id,
                "app_id": row_app_id,
                "lang": "english",
                "created_at": created_at,
                "recommended": recommended,
                "playtime_hours": playtime_hours,
                "helpful_count": helpful_count,
                "funny_count": funny_count,
                "purchase_type": purchase_type,
                "review_url": review_url,
                "review_text": review_text,
                "embedding": None,
            }
        )

        if len(batch) >= 1000:
            upsert_count += _flush_batch(db, batch)
            batch.clear()
            if upsert_count and upsert_count % 5000 == 0:
                logger.info("Ingested %s reviews", upsert_count)

    if batch:
        upsert_count += _flush_batch(db, batch)

    logger.info("Ingest complete: %s reviews", upsert_count)
    return upsert_count


def _flush_batch(db: Database, batch: list[dict]) -> int:
    ids = [item["review_id"] for item in batch]
    existing = _fetch_existing_checksums(db, ids)

    db.upsert_reviews(batch)

    changed: list[str] = []
    for item in batch:
        digest = hash_text(item["review_text"] or "")
        previous = existing.get(item["review_id"])
        if previous is not None and previous != digest:
            changed.append(item["review_id"])
        elif previous is None:
            # New review -> ensure embedding is null
            changed.append(item["review_id"])
    _reset_embeddings(db, changed)
    return len(batch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Steam review CSV into Postgres")
    parser.add_argument("--csv", required=True, type=Path, help="Path to steam_reviews_en.csv")
    parser.add_argument("--app-id", required=False, help="Default app id when CSV lacks one")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for smoke testing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    total = ingest(args.csv, app_id=args.app_id, limit=args.limit)
    logger.info("Ingested %s rows", total)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
