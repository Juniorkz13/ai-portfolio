CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    uploaded_at TIMESTAMP NOT NULL DEFAULT NOW(),
    document_type VARCHAR(100) NOT NULL DEFAULT 'unknown',
    version VARCHAR(50) NOT NULL DEFAULT '1.0'
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
