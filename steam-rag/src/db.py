"""Database utilities for Postgres/pgvector with a sqlite fallback for tests."""
from __future__ import annotations

import argparse
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - offline fallback
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg.rows import dict_row
except ModuleNotFoundError:  # pragma: no cover
    psycopg = None  # type: ignore

    def register_vector(conn):  # type: ignore
        return None

    dict_row = None  # type: ignore

from .config import get_settings

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    """Lightweight DB wrapper used throughout the application."""

    def __init__(self, dsn: str | None = None):
        settings = get_settings()
        self.dsn = dsn or settings.db_dsn
        self.backend = "sqlite" if self.dsn.startswith("sqlite") else "postgres"
        self._conn: Any | None = None

    # -- connection management -------------------------------------------------
    def connect(self):
        if self._conn is not None:
            return self._conn
        if self.backend == "sqlite":
            path = self.dsn.split("sqlite:///")[-1]
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            self._conn = conn
        else:
            if psycopg is None:
                raise RuntimeError("psycopg is required for Postgres connections")
            conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
            register_vector(conn)
            self._conn = conn
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def cursor(self):
        conn = self.connect()
        if self.backend == "sqlite":
            cur = conn.cursor()
            try:
                yield cur
                conn.commit()
            finally:
                cur.close()
        else:
            with conn.cursor() as cur:
                yield cur

    # -- executing helpers -----------------------------------------------------
    def execute(self, sql: str, params: Mapping[str, Any] | Sequence[Any] | None = None):
        with self.cursor() as cur:
            cur.execute(sql, params or ())

    def executemany(self, sql: str, params_seq: Iterable[Sequence[Any] | Mapping[str, Any]]):
        with self.cursor() as cur:
            cur.executemany(sql, list(params_seq))

    def fetchall(self, sql: str, params: Mapping[str, Any] | Sequence[Any] | None = None) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
        if self.backend == "sqlite":
            return [dict(row) for row in rows]
        return rows

    def fetchone(self, sql: str, params: Mapping[str, Any] | Sequence[Any] | None = None) -> dict | None:
        with self.cursor() as cur:
            cur.execute(sql, params or ())
            row = cur.fetchone()
        if row is None:
            return None
        if self.backend == "sqlite":
            return dict(row)
        return row

    # -- schema ----------------------------------------------------------------
    def run_schema(self):
        sql_text = _SCHEMA_PATH.read_text(encoding="utf-8")
        if self.backend == "sqlite":
            statements = _prepare_sqlite_statements(sql_text)
            for statement in statements:
                self.execute(statement)
        else:
            with self.cursor() as cur:
                cur.execute(sql_text)
        logger.info("Schema ensured for backend %s", self.backend)

    # -- bulk helpers ----------------------------------------------------------
    def upsert_reviews(self, rows: Sequence[Mapping[str, Any]]):
        if not rows:
            return
        if self.backend == "sqlite":
            sql = (
                "INSERT INTO reviews (review_id, app_id, lang, created_at, recommended, "
                "playtime_hours, helpful_count, funny_count, purchase_type, review_url, "
                "review_text, embedding, updated_at)"
                " VALUES (:review_id, :app_id, :lang, :created_at, :recommended, :playtime_hours,"
                " :helpful_count, :funny_count, :purchase_type, :review_url, :review_text,"
                " :embedding, CURRENT_TIMESTAMP)"
                " ON CONFLICT(review_id) DO UPDATE SET "
                "app_id=excluded.app_id, lang=excluded.lang, created_at=excluded.created_at,"
                "recommended=excluded.recommended, playtime_hours=excluded.playtime_hours,"
                "helpful_count=excluded.helpful_count, funny_count=excluded.funny_count,"
                "purchase_type=excluded.purchase_type, review_url=excluded.review_url,"
                "review_text=excluded.review_text, updated_at=CURRENT_TIMESTAMP"
            )
        else:
            sql = (
                "INSERT INTO reviews (review_id, app_id, lang, created_at, recommended, "
                "playtime_hours, helpful_count, funny_count, purchase_type, review_url, "
                "review_text, embedding, updated_at)"
                " VALUES (%(review_id)s, %(app_id)s, %(lang)s, %(created_at)s, %(recommended)s,"
                " %(playtime_hours)s, %(helpful_count)s, %(funny_count)s, %(purchase_type)s,"
                " %(review_url)s, %(review_text)s, %(embedding)s, NOW())"
                " ON CONFLICT (review_id) DO UPDATE SET "
                "app_id=EXCLUDED.app_id, lang=EXCLUDED.lang, created_at=EXCLUDED.created_at,"
                "recommended=EXCLUDED.recommended, playtime_hours=EXCLUDED.playtime_hours,"
                "helpful_count=EXCLUDED.helpful_count, funny_count=EXCLUDED.funny_count,"
                "purchase_type=EXCLUDED.purchase_type, review_url=EXCLUDED.review_url,"
                "review_text=EXCLUDED.review_text, updated_at=NOW()"
            )
        self.executemany(sql, rows)

    def upsert_embedding_meta(self, entries: Sequence[Mapping[str, Any]]):
        if not entries:
            return
        if self.backend == "sqlite":
            sql = (
                "INSERT INTO review_embeddings (review_id, text_checksum, embedded_at)"
                " VALUES (:review_id, :text_checksum, CURRENT_TIMESTAMP)"
                " ON CONFLICT(review_id) DO UPDATE SET text_checksum=excluded.text_checksum,"
                " embedded_at=CURRENT_TIMESTAMP"
            )
        else:
            sql = (
                "INSERT INTO review_embeddings (review_id, text_checksum, embedded_at)"
                " VALUES (%(review_id)s, %(text_checksum)s, NOW())"
                " ON CONFLICT (review_id) DO UPDATE SET text_checksum=EXCLUDED.text_checksum,"
                " embedded_at=NOW()"
            )
        self.executemany(sql, entries)

    def upsert_topics(self, entries: Sequence[Mapping[str, Any]]):
        if not entries:
            return
        if self.backend == "sqlite":
            sql = (
                "INSERT INTO review_topics (review_id, topic, sentiment, confidence)"
                " VALUES (:review_id, :topic, :sentiment, :confidence)"
                " ON CONFLICT(review_id, topic) DO UPDATE SET sentiment=excluded.sentiment,"
                " confidence=excluded.confidence"
            )
        else:
            sql = (
                "INSERT INTO review_topics (review_id, topic, sentiment, confidence)"
                " VALUES (%(review_id)s, %(topic)s, %(sentiment)s, %(confidence)s)"
                " ON CONFLICT (review_id, topic) DO UPDATE SET sentiment=EXCLUDED.sentiment,"
                " confidence=EXCLUDED.confidence"
            )
        self.executemany(sql, entries)


def _prepare_sqlite_statements(sql_text: str) -> list[str]:
    sanitized_lines = []
    skip_block = False
    for raw_line in sql_text.splitlines():
        line = raw_line.strip()
        if line.upper().startswith("DO $$"):
            skip_block = True
            continue
        if skip_block and line.upper().startswith("END $$"):
            skip_block = False
            continue
        if skip_block:
            continue
        if line.upper().startswith("CREATE EXTENSION"):
            continue
        raw_line = raw_line.replace("TIMESTAMPTZ", "TIMESTAMP")
        raw_line = raw_line.replace("VECTOR(1536)", "TEXT")
        raw_line = raw_line.replace("NOW()", "CURRENT_TIMESTAMP")
        sanitized_lines.append(raw_line)
    sanitized_sql = "\n".join(sanitized_lines)
    statements = [stmt.strip() for stmt in sanitized_sql.split(";") if stmt.strip()]
    return statements


# -- CLI ----------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Steam RAG database helper")
    parser.add_argument("--init", action="store_true", help="Initialise schema")
    parser.add_argument("--check", action="store_true", help="Test connectivity")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    db = Database()
    if args.init:
        db.run_schema()
        logger.info("Schema initialised")
    if args.check:
        row = db.fetchone("SELECT NOW() AS ts" if db.backend != "sqlite" else "SELECT datetime('now') AS ts")
        logger.info("Database reachable: %s", row["ts"] if row else "unknown")
    if not (args.init or args.check):
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())


__all__ = ["Database"]
