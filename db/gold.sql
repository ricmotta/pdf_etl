-- Overview (single row)
CREATE OR REPLACE VIEW etl.v_summary_overview AS
SELECT
  COUNT(*)                         AS n_docs,
  COALESCE(SUM(sum_line_total),0)  AS sum_value,
  SUM(CASE WHEN diff_sum_vs_subtotal IS DISTINCT FROM 0 THEN 1 ELSE 0 END) AS n_with_diff_subtotal,
  NOW()::timestamptz AS generated_ts
FROM etl.documents;

-- Top parties by value
CREATE OR REPLACE VIEW etl.v_top_parties AS
SELECT
  party_name,
  SUM(sum_line_total) AS sum_value,
  COUNT(*)            AS count_docs
FROM etl.documents
GROUP BY party_name
ORDER BY SUM(sum_line_total) DESC, COUNT(*) DESC;

-- Top items by frequency (sum of quantities)
CREATE OR REPLACE VIEW etl.v_top_items_by_frequency AS
SELECT
  item,
  COALESCE(SUM(COALESCE(quantity, 1)), 0) AS total_quantity,
  COALESCE(SUM(line_total),0)  AS sum_value
FROM etl.line_items
GROUP BY item
ORDER BY total_quantity DESC, sum_value DESC;

-- Top items by value
CREATE OR REPLACE VIEW etl.v_top_items_by_value AS
SELECT
  item,
  COALESCE(SUM(line_total),0) AS sum_value,
  COUNT(*)                    AS count_rows
FROM etl.line_items
GROUP BY item
ORDER BY SUM(line_total) DESC, COUNT(*) DESC;

-- Consistency diffs
CREATE OR REPLACE VIEW etl.v_consistency_diffs AS
SELECT
  d.document_id,
  d.party_name,
  d.sum_line_total,
  d.subtotal_meta,
  d.diff_sum_vs_subtotal,
  d.extraction_flavor,
  d.source_path
FROM etl.documents d;

-- Outliers by subtotal difference (absolute and relative thresholds)
CREATE OR REPLACE FUNCTION etl.f_consistency_outliers(
  abs_threshold NUMERIC DEFAULT 0.01,
  rel_threshold NUMERIC DEFAULT 0.005
)
RETURNS TABLE (
  document_id TEXT,
  party_name TEXT,
  sum_line_total NUMERIC,
  subtotal_meta NUMERIC,
  diff_sum_vs_subtotal NUMERIC,
  total_meta NUMERIC,
  extraction_flavor TEXT,
  source_path TEXT
)
LANGUAGE sql STABLE AS
$$
  SELECT
    d.document_id,
    d.party_name,
    d.sum_line_total,
    d.subtotal_meta,
    d.diff_sum_vs_subtotal,
    d.total_meta,
    d.extraction_flavor,
    d.source_path
  FROM etl.documents d
  WHERE
    d.subtotal_meta IS NOT NULL
    AND ABS(d.diff_sum_vs_subtotal) > abs_threshold
    AND ABS(d.diff_sum_vs_subtotal) / GREATEST(ABS(d.subtotal_meta), 1) > rel_threshold
  ORDER BY ABS(d.diff_sum_vs_subtotal) DESC;
$$;
