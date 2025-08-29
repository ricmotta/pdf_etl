# src/db/load_pg.py
from __future__ import annotations
from pathlib import Path
from io import StringIO
from typing import Tuple, List
import pandas as pd
import psycopg


def _pick_silver_pair(sdir: Path) -> Tuple[Path, Path]:
    """
    Return (items_csv, docs_csv) for either invoices or receipts.
    Priority: invoice_* → receipt_* → glob fallback.
    """
    candidates: List[Tuple[str, str]] = [
        ("invoice_items.csv", "invoice_documents.csv"),
        ("receipt_items.csv", "receipt_documents.csv"),
    ]
    for a, b in candidates:
        ia, ib = sdir / a, sdir / b
        if ia.exists() and ib.exists():
            return ia, ib

    items = sorted(sdir.glob("*_items.csv"))
    docs = sorted(sdir.glob("*_documents.csv"))
    if items and docs:
        return items[0], docs[0]

    found = sorted(p.name for p in sdir.glob("*.csv"))
    raise FileNotFoundError(
        f"No matching silver CSVs in {sdir}. Expected invoice_* or receipt_* pairs. Found: {found}"
    )


def _copy_csv(conn: psycopg.Connection, df: pd.DataFrame, table: str, columns: list[str]) -> None:
    """
    Fast load using COPY FROM STDIN. NaN -> empty cell (NULL).
    """
    csv_buf = StringIO()
    # Ensure column order
    df = df[columns].copy()
    df.to_csv(csv_buf, index=False, header=False)
    csv_buf.seek(0)
    cols = ", ".join(columns)
    sql = f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT CSV)"
    with conn.cursor() as cur, cur.copy(sql) as copy:
        copy.write(csv_buf.getvalue())


def load_postgres_and_gold(
    silver_dir: Path,
    dsn: str,
    ddl_path: Path = Path("db/ddl.sql"),
    gold_sql_path: Path = Path("db/gold.sql"),
    replace: bool = True,
) -> dict:
    """
    Create schema/tables (DDL), load silver CSVs into Postgres, then run GOLD SQL.
    Works for either invoices or receipts (auto-detection by filename).
    """
    silver_dir = Path(silver_dir)

    # Detect invoice_* or receipt_* pair
    items_csv, docs_csv = _pick_silver_pair(silver_dir)

    items = pd.read_csv(items_csv)
    docs = pd.read_csv(docs_csv)

    # Superset of document columns (invoices + receipts). Missing ones will be added as NaN.
    docs_cols = [
    "document_id","doc_type", "document_number",
    "party_name","supplier_name","currency",
    "items_count","sum_line_total",
    "subtotal_meta","tax_meta_pct",
    "total_meta","diff_sum_vs_subtotal","diff_sum_vs_total",
    "source_path", "extraction_flavor"
]

    items_cols = ["document_id", "item", "description", "quantity", "unit_price", "line_total"]

    # Add any missing columns as NaN to keep COPY column order consistent
    for c in docs_cols:
        if c not in docs.columns:
            docs[c] = pd.NA
    for c in items_cols:
        if c not in items.columns:
            items[c] = pd.NA

    # Load SQL
    ddl_sql = Path(ddl_path).read_text(encoding="utf-8")
    gold_sql = Path(gold_sql_path).read_text(encoding="utf-8")

    with psycopg.connect(dsn) as conn:
        # Create schema & tables
        conn.execute(ddl_sql)

        # Replace strategy: delete existing rows for these document_ids
        if replace and not docs.empty:
            doc_ids = docs["document_id"].astype(str).dropna().unique().tolist()
            if doc_ids:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM etl.line_items WHERE document_id = ANY(%s)", (doc_ids,))
                    cur.execute("DELETE FROM etl.documents  WHERE document_id = ANY(%s)", (doc_ids,))

        # COPY (fast load)
        _copy_csv(conn, docs[docs_cols], "etl.documents", docs_cols)
        _copy_csv(conn, items[items_cols], "etl.line_items", items_cols)

        # GOLD
        conn.execute(gold_sql)
        conn.commit()

    return {
        "loaded_docs": int(len(docs)),
        "loaded_items": int(len(items)),
    }
