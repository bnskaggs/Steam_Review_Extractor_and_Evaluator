# Steam Review Retrieval-Augmented Generation Stack

This project ingests Steam reviews from CSV, stores them in Postgres + pgvector, annotates them with topic/sentiment labels, and serves a FastAPI RAG interface for semantic search, analytics, and summarisation.

## Features

- Postgres 16 with pgvector for ANN search and metadata tables
- CSV ingestion with robust parsing and deduplication
- OpenAI embeddings (`text-embedding-3-small`) with checksum tracking
- Prompt-based multi-label topic + sentiment classification (ready to swap for fine-tuned models)
- Hybrid semantic/BM25 retrieval with optional cross-encoder reranker
- FastAPI endpoints for `/search`, `/ask`, `/counts`
- Docker Compose for the database, Windows-friendly CLI commands
- Example SQL queries for analytics

## Prerequisites

- Python 3.11
- Docker Desktop (for Postgres/pgvector)
- OpenAI API key with access to the requested models

## Quickstart

All commands assume a PowerShell session at the project root (`steam-rag`).

```powershell
# 1) Install Python dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) start DB
docker compose up -d

# 3) create schema
python -m src.db --init

# 4) ingest
python -m src.ingest_csv --csv steam_reviews_en.csv --app-id 1364780

# 5) embed
python -m src.embed_load --limit 1000000

# 6) classify (batch)
python -m src.classify --since 2023-01-01

# 7) run API
uvicorn src.rag_api:app --host 0.0.0.0 --port 8000
```

On macOS/Linux substitute the virtualenv activation line with `source .venv/bin/activate`.

Create a `.env` file based on `.env.example` before running scripts:

```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-5-turbo
EMBED_MODEL=text-embedding-3-small
DB_DSN=postgresql://postgres:postgres@localhost:5432/steam_rag
CORS_ALLOW_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## CSV Expectations

- UTF-8 encoded CSV with at least `review_id`, `review_text`, `lang`, `created_at` columns.
- Additional columns are tolerated; missing optional columns default sensibly.
- Only `lang=english` rows are ingested.
- Duplicate `review_id` entries are ignored.
- When review text changes, embeddings are cleared and re-generated on the next embed run.

## Endpoints

All endpoints accept/return JSON.

### `POST /search`

Request body:

```json
{
  "query": "rollback netcode",
  "app_id": "1364780",
  "k": 10,
  "topics": ["netcode"],
  "sentiments": ["negative"],
  "rerank": true
}
```

Response:

```json
{
  "results": [
    {
      "review_id": "123",
      "review_text": "...",
      "review_url": "https://...",
      "helpful_count": 42,
      "created_at": "2024-03-15T12:34:00",
      "score": 0.78
    }
  ]
}
```

### `POST /ask`

Request body:

```json
{
  "query": "Summarise negative feedback on monetization",
  "app_id": "1364780",
  "date_start": "2024-01-01",
  "date_end": "2024-06-01",
  "topics": ["monetization"],
  "sentiments": ["negative", "very_negative"],
  "rerank": true
}
```

Response:

```json
{
  "answer": "• Many players dislike the seasonal pass [abc123]\n• ...",
  "citations": ["abc123", "def456"],
  "snippets": [{"review_id": "abc123", "review_text": "...", ...}]
}
```

### `GET /counts`

Query parameters: `app_id`, optional `topic`, `date_start`, `date_end`, `min_helpful`, `group_by=month`.

Example response:

```json
[
  {
    "topic": "netcode",
    "total": 721,
    "by_sentiment": {
      "very_positive": 34,
      "positive": 198,
      "neutral": 87,
      "negative": 301,
      "very_negative": 101
    },
    "buckets": [
      {
        "bucket": "2024-01-01",
        "total": 120,
        "by_sentiment": {"negative": 50, "very_negative": 20}
      }
    ]
  }
]
```

## Tests

The unit tests run against a sqlite fallback with patched OpenAI calls:

```bash
pytest
```

## Troubleshooting

- **pgvector index missing**: rerun `python -m src.db --init` to ensure schema. The init script is idempotent.
- **OpenAI rate limits**: use `--limit` flags on embedding/classification scripts for incremental processing.
- **CSV parse errors**: rows with malformed quotes or missing required fields are skipped with a log message; inspect logs for details.
- **Windows path issues**: ensure Docker Desktop exposes port 5432 and the repository path has no spaces to simplify scripts.

## Project Layout

```
steam-rag/
  docker-compose.yml
  requirements.txt
  src/
    config.py
    db.py
    ingest_csv.py
    embed_load.py
    classify.py
    retrieval.py
    rag_api.py
    prompts.py
    utils.py
    schema.sql
    queries.sql
  tests/
    test_smoke.py
    test_counts.py
```
