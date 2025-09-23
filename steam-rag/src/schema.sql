-- Database schema for Steam review RAG system
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    app_id TEXT NOT NULL,
    lang TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    recommended BOOLEAN,
    playtime_hours REAL,
    helpful_count INT,
    funny_count INT,
    purchase_type TEXT,
    review_url TEXT,
    review_text TEXT NOT NULL,
    embedding VECTOR(1536),
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

CREATE TABLE IF NOT EXISTS review_topics (
    review_id TEXT REFERENCES reviews(review_id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    sentiment TEXT NOT NULL CHECK (sentiment IN ('very_positive','positive','neutral','negative','very_negative')),
    confidence REAL,
    PRIMARY KEY (review_id, topic)
);

CREATE TABLE IF NOT EXISTS review_embeddings (
    review_id TEXT PRIMARY KEY REFERENCES reviews(review_id) ON DELETE CASCADE,
    text_checksum TEXT NOT NULL,
    embedded_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reviews_app_lang_created ON reviews (app_id, lang, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_review_topics_topic ON review_topics (topic);
CREATE INDEX IF NOT EXISTS idx_review_topics_sentiment ON review_topics (sentiment);

-- Vector index for ANN search
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_reviews_embedding_ivfflat'
    ) THEN
        CREATE INDEX idx_reviews_embedding_ivfflat ON reviews USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    END IF;
END $$;
