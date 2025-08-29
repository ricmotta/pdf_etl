-- Schema
CREATE SCHEMA IF NOT EXISTS etl;

-- Documents: one row per document (invoice or receipt)
CREATE TABLE IF NOT EXISTS etl.documents (
  document_id           TEXT PRIMARY KEY,
  doc_type              TEXT NOT NULL,  -- 'invoice' or 'receipt'

  -- Generic metadata
  document_number        TEXT,
  party_name             TEXT,           -- who receives (invoice) or customer (receipt if available)
  supplier_name          TEXT,           -- who issues (supplier for invoices, merchant for receipts)
  currency               TEXT,
  items_count            INTEGER,
  sum_line_total         NUMERIC,

  -- Invoice-style fields
  subtotal_meta          NUMERIC,
  tax_meta_pct           NUMERIC,

  -- Generic totals
  total_meta             NUMERIC,
  diff_sum_vs_subtotal   NUMERIC,
  diff_sum_vs_total      NUMERIC,

  -- Provenance
  source_path            TEXT,
  extraction_flavor      TEXT,
  updated_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Line items: many rows per document
CREATE TABLE IF NOT EXISTS etl.line_items (
  id           BIGSERIAL PRIMARY KEY,
  document_id  TEXT NOT NULL REFERENCES etl.documents(document_id) ON DELETE CASCADE,
  item         TEXT,
  description  TEXT,
  quantity     NUMERIC(14,3),
  unit_price   NUMERIC(14,2),
  line_total   NUMERIC(14,2)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_line_items_document_id  ON etl.line_items(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_party_name    ON etl.documents(party_name);
CREATE INDEX IF NOT EXISTS idx_documents_supplier_name ON etl.documents(supplier_name);
CREATE INDEX IF NOT EXISTS idx_documents_doc_type      ON etl.documents(doc_type);

-- Normalize defaults and stamp updated_at
CREATE OR REPLACE FUNCTION etl.fn_documents_normalize()
RETURNS trigger AS $$
BEGIN
  IF NEW.document_number IS NULL OR NEW.document_number = '' THEN
    NEW.document_number := 'Not Found';
  END IF;

  IF NEW.doc_type = 'receipt' AND (NEW.party_name IS NULL OR NEW.party_name = '') THEN
    NEW.party_name := 'End Customer';
  END IF;

  NEW.updated_at := now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_biu_norm ON etl.documents;
CREATE TRIGGER trg_documents_biu_norm
BEFORE INSERT OR UPDATE ON etl.documents
FOR EACH ROW
EXECUTE FUNCTION etl.fn_documents_normalize();
