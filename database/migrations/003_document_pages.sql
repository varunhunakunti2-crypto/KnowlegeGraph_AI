-- Migration 003: store the extracted page snapshot used by the API.
-- The application writes documents.pages, while the initial schema only
-- defined documents.metadata. Keep this idempotent for existing databases.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS pages JSONB;
