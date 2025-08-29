# src/ingest.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Invoices
from src.pipelines.invoices.extract import extract_invoice
from src.pipelines.invoices.silver import bronze_to_silver
from src.db.load_pg import load_postgres_and_gold
from src.pipelines.invoices.report import make_report as make_invoice_report

# Receipts
from src.pipelines.receipts.extract import extract_receipts_dir
from src.pipelines.receipts.silver import aggregate_receipts_silver
from src.pipelines.receipts.report import make_receipt_report


def _safe_stem(pdf: Path) -> str:
    """
    Produce a filesystem-safe stem (used for per-PDF subfolders).
    """
    s = pdf.stem.lower()
    s = re.sub(r"[^a-z0-9_\-]+", "-", s)
    return s or "doc"


def ingest_dir_invoices(
    in_dir: Path,
    bronze_root: Path,
    silver_dir: Path,
    dsn: str,
    ddl_path: Path,
    gold_sql_path: Path,
    report_out: Path,
    threshold: float = 0.01,
    top_n: int = 5,
    replace: bool = True,
) -> Dict[str, Any]:
    """
    Batch pipeline for INVOICES:

      For each PDF in `in_dir`:
        1) Extract bronze into its own subfolder: {bronze_root}/{pdf_stem}/
        2) Build silver for THIS document into:   {bronze_root}/{pdf_stem}/_silver/

      After the loop:
        3) Concatenate all per-doc silver files into the GLOBAL silver_dir:
             - silver_dir/invoice_items.csv
             - silver_dir/invoice_documents.csv
        4) Load into Postgres (DDL + GOLD SQL)
        5) Generate Markdown report

    Returns a summary dict with processed/failed counts and paths.
    """
    in_dir = Path(in_dir)
    bronze_root = Path(bronze_root)
    silver_dir = Path(silver_dir)
    ddl_path = Path(ddl_path)
    gold_sql_path = Path(gold_sql_path)
    report_out = Path(report_out)

    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDFs found in {in_dir}")

    bronze_root.mkdir(parents=True, exist_ok=True)
    silver_dir.mkdir(parents=True, exist_ok=True)

    processed: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []

    # Track per-document silver outputs for aggregation
    perdoc_items: List[pd.DataFrame] = []
    perdoc_docs: List[pd.DataFrame] = []

    for pdf in pdfs:
        try:
            # 1) Bronze (per-document)
            bronze_sub = bronze_root / _safe_stem(pdf)
            bronze_sub.mkdir(parents=True, exist_ok=True)
            ext = extract_invoice(pdf_path=pdf, out_dir=bronze_sub)

            # 2) Silver (per-document)
            silver_sub = silver_dir / _safe_stem(pdf)
            silver_sub.mkdir(parents=True, exist_ok=True)
            _ = bronze_to_silver(bronze_dir=bronze_sub, out_dir=silver_sub)

            # Read back for aggregation
            p_items = silver_sub / "invoice_items.csv"
            p_docs  = silver_sub / "invoice_documents.csv"
            if p_items.exists():
                perdoc_items.append(pd.read_csv(p_items))
            if p_docs.exists():
                perdoc_docs.append(pd.read_csv(p_docs))

            processed.append({
                "pdf": str(pdf),
                "bronze_rows": ext.get("rows", 0),
                "extraction_flavor": ext.get("extraction_flavor"),
            })
        except Exception as e:
            failed.append({"pdf": str(pdf), "error": str(e)})

    # 3) Aggregate into global silver/
    items_out = silver_dir / "invoice_items.csv"
    docs_out = silver_dir / "invoice_documents.csv"

    if perdoc_items:
        items_agg = pd.concat(perdoc_items, ignore_index=True)
    else:
        items_agg = pd.DataFrame(
            columns=["document_id", "item", "description", "quantity", "unit_price", "line_total"]
        )

    if perdoc_docs:
        docs_agg = pd.concat(perdoc_docs, ignore_index=True)
    else:
        docs_agg = pd.DataFrame(
            columns=[
                "document_id", "doc_type", "document_number_raw", "document_number",
                "party_name", "currency", "items_count",
                "sum_line_total", "subtotal_meta", "tax_meta_pct", "amount_due_meta",
                "total_meta", "diff_sum_vs_subtotal", "diff_sum_vs_total",
                "source_path", "extraction_flavor"
            ]
        )

    items_agg.to_csv(items_out, index=False)
    docs_agg.to_csv(docs_out, index=False)

    # 4) Load into Postgres + GOLD
    load_out = load_postgres_and_gold(
        silver_dir=silver_dir,
        dsn=dsn,
        ddl_path=ddl_path,
        gold_sql_path=gold_sql_path,
        replace=replace,
    )

    # 5) Report (Markdown)
    report_path = make_invoice_report(
        dsn=dsn,
        out_path=report_out,
        threshold=threshold,
        top_n=top_n,
    )

    return {
        "processed_count": len(processed),
        "failed_count": len(failed),
        "processed": processed,
        "failed": failed,
        "loaded_docs": load_out.get("loaded_docs"),
        "loaded_items": load_out.get("loaded_items"),
        "report_path": str(report_path),
    }


def ingest_dir_receipts(
    in_dir: Path,
    bronze_root: Path,
    silver_dir: Path,
    dsn: str,
    ddl_path: Path,
    gold_sql_path: Path,
    report_out: Path,
    threshold: float = 0.03,
    top_n: int = 5,
    replace: bool = True,
) -> Dict[str, Any]:
    """
    Batch pipeline for RECEIPTS:

      1) Bronze:
         - Run OCR extraction for all images in `in_dir` into {bronze_root}/{stem}/
           (uses src.pipelines.receipts.extract.extract_receipts_dir)

      2) Silver:
         - Promote all per-receipt bronze folders to per-document silver and aggregate into:
             - silver_dir/receipt_items.csv
             - silver_dir/receipt_documents.csv

      3) Load into Postgres (DDL + GOLD SQL)

      4) Generate Markdown report

    Returns a summary dict with processed/failed counts and paths.
    """
    in_dir = Path(in_dir)
    bronze_root = Path(bronze_root)
    silver_dir = Path(silver_dir)
    ddl_path = Path(ddl_path)
    gold_sql_path = Path(gold_sql_path)
    report_out = Path(report_out)

    bronze_root.mkdir(parents=True, exist_ok=True)
    silver_dir.mkdir(parents=True, exist_ok=True)

    # 1) Bronze (batch OCR)
    results = extract_receipts_dir(in_dir=str(in_dir), out_root=str(bronze_root))

    processed: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []
    for r in results:
        if "error" in r:
            failed.append({"img_path": r.get("img_path", ""), "error": r["error"]})
        else:
            processed.append(r)

    # 2) Silver (aggregate)
    silver_summary = aggregate_receipts_silver(bronze_root=bronze_root, silver_dir=silver_dir)

    # 3) Load into Postgres + GOLD
    load_out = load_postgres_and_gold(
        silver_dir=silver_dir,
        dsn=dsn,
        ddl_path=ddl_path,
        gold_sql_path=gold_sql_path,
        replace=replace,
    )

    # 4) Report (Markdown)
    report_path = make_receipt_report(
        dsn=dsn,
        out_path=report_out,
        threshold=threshold,
        top_n=top_n,
    )

    return {
        "processed_count": len(processed),
        "failed_count": len(failed),
        "processed": processed,
        "failed": failed,
        "silver": silver_summary,
        "loaded_docs": load_out.get("loaded_docs"),
        "loaded_items": load_out.get("loaded_items"),
        "report_path": str(report_path),
    }


# Backward-compatible alias (kept for any existing imports)
ingest_dir = ingest_dir_invoices
