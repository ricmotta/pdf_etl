# src/pipelines/receipts/silver.py
# -*- coding: utf-8 -*-
"""
Receipts → Silver
- Read bronze outputs (receipt_line_items.csv + receipt_meta.json) per-document
- Normalize numbers and merchant name
- Compute sane fallbacks for totals
- Write per-document silver CSVs: _silver/receipt_items.csv, _silver/receipt_documents.csv
- Provide an aggregate helper to build data/output/silver/receipts/*
"""

from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd

MONEY_RX = re.compile(r"[-+]?\d{1,3}(\.\d{3})*(,\d{2})|[-+]?\d+(\.\d+)?")

# Final set of columns that WILL go to DB (etl.documents) for receipts
DOCS_KEEP = [
    "document_id", "doc_type",
    "document_number_raw", "document_number",
    "party_name", "supplier_name", "currency", "extraction_flavor",
    "items_count", "sum_line_total",
    "subtotal_meta", "tax_meta_pct",
    "total_meta", "diff_sum_vs_subtotal", "diff_sum_vs_total",
    "source_path", "ingest_ts", "silver_ts",
]

def _parse_money(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    s = s.replace(" ", "")
    # normalize thousands/decimal: supports "1.234,56" and "1,234.56"
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def _parse_int(x: Any) -> Optional[int]:
    try:
        return int(float(str(x)))
    except Exception:
        return None

def _norm_merchant(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9\s\-\&\'\.]", " ", name)).strip().upper()

def _parse_dt(dt_raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (txn_datetime_iso, txn_date_iso)."""
    if not dt_raw:
        return (None, None)
    s = dt_raw.strip()
    fmts = [
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            return (dt.isoformat(timespec="seconds"), dt.date().isoformat())
        except Exception:
            continue
    return (None, None)

def _vendor_slug(name: str | None) -> str:
    """Return an 8-char A–Z0–9 slug from supplier/merchant name."""
    if not name:
        return "UNKNOWN"
    return (re.sub(r"[^A-Z0-9]", "", str(name).upper())[:8] or "UNKNOWN")

def _yyyymmdd_from_iso(dt_iso: str | None) -> str:
    """From 'YYYY-MM-DDTHH:MM:SS' return 'YYYYMMDD'; else 'NA'."""
    if not dt_iso:
        return "NA"
    try:
        return dt_iso.split("T", 1)[0].replace("-", "")
    except Exception:
        return "NA"

def bronze_to_silver(
    bronze_dir: Path,
    out_dir: Path,
    currency_default: str = "USD",
    subtotal_tolerance: float = 0.02,  # 2%
) -> Dict[str, Any]:
    """
    Promote one receipt bronze folder to per-document silver CSVs.
    Input:
      bronze_dir / 'receipt_line_items.csv'
      bronze_dir / 'receipt_meta.json'
    Output:
      out_dir / 'receipt_items.csv'
      out_dir / 'receipt_documents.csv'
    """
    items_csv = bronze_dir / "receipt_line_items.csv"
    meta_json = bronze_dir / "receipt_meta.json"
    assert items_csv.exists() and meta_json.exists(), f"Missing bronze outputs in {bronze_dir}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- meta ---
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    merchant = _norm_merchant(meta.get("merchant_name"))
    # Keep parsing datetime only to build a stable document_id (date part). We won't store it in DB.
    _, txn_dt_iso = _parse_dt(meta.get("txn_datetime_raw"))
    items_detected = _parse_int(meta.get("items_sold_meta"))

    subtotal_meta = _parse_money(meta.get("subtotal_meta"))
    total_meta = _parse_money(meta.get("total_meta"))
    tendered_meta = _parse_money(meta.get("tendered_meta"))
    change_due_meta = _parse_money(meta.get("change_due_meta"))
    discount_val = _parse_money(meta.get("discount_meta_val"))
    tax_val = _parse_money(meta.get("tax_meta_val"))
    tax_pct = meta.get("tax_meta_pct")
    tax_pct = float(tax_pct) if isinstance(tax_pct, (int, float)) else None

    # --- items ---
    df = pd.read_csv(items_csv)
    # Normalize numbers
    if "quantity" in df.columns:
        df["quantity"] = df["quantity"].apply(lambda v: _parse_money(v) or 0.0)
    for col in ("unit_price", "line_total"):
        if col in df.columns:
            df[col] = df[col].apply(_parse_money)

    # Compute line_total when missing
    if {"line_total", "unit_price", "quantity"}.issubset(df.columns):
        mask_missing = df["line_total"].isna() | (df["line_total"] == 0)
        df.loc[mask_missing, "line_total"] = (
            df.loc[mask_missing, "unit_price"].fillna(0) * df.loc[mask_missing, "quantity"].fillna(0)
        )

    # Safety on negatives
    for c in ("quantity", "unit_price", "line_total"):
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
            df.loc[df[c] < 0, c] = 0.0

    sum_line_total = float(df["line_total"].sum()) if "line_total" in df.columns else 0.0

    # Infer totals if needed
    if total_meta is None and tendered_meta is not None and change_due_meta is not None:
        total_meta = max(0.0, tendered_meta - change_due_meta)
    if subtotal_meta is None:
        subtotal_meta = max(0.0, sum_line_total - (discount_val or 0.0))

    if tax_val is None and tax_pct is not None and subtotal_meta is not None:
        tax_val = round(subtotal_meta * float(tax_pct), 2)
    if total_meta is None and subtotal_meta is not None:
        total_meta = round(subtotal_meta + (tax_val or 0.0) - (discount_val or 0.0), 2)

    # Diff metrics
    diff_sum_vs_subtotal = None
    if subtotal_meta is not None:
        diff_sum_vs_subtotal = sum_line_total - float(subtotal_meta)

    diff_sum_vs_total = None
    if total_meta is not None:
        diff_sum_vs_total = sum_line_total - float(total_meta)

    # Minimal validations
    validations = {
        "items_detected_gt_zero": (items_detected or 0) > 0,
        "subtotal_nonneg": (subtotal_meta or 0) >= 0,
        "total_nonneg": (total_meta or 0) >= 0,
        "subtotal_close_to_sum": (abs(diff_sum_vs_subtotal or 0) <= subtotal_tolerance),
    }

     # -------------------------------
    # Harmonize document_number & document_id with invoice pattern
    # document_number: try file stem; else "Not Found"
    # document_id: VENDOR8-<document_number|NA>-<YYYYMMDD|NA>
    # -------------------------------
    stem = bronze_dir.name  # e.g., "00", "01", ...
    document_number = stem if stem else "Not Found"

    vendor8 = _vendor_slug(merchant)
    yyyymmdd = _yyyymmdd_from_iso(txn_dt_iso)
    dn_part = document_number if document_number != "Not Found" else "NA"
    document_id = f"{vendor8}-{dn_part}-{yyyymmdd}"

    # --- write per-document silver ---
    items_out = out_dir / "receipt_items.csv"
    docs_out = out_dir / "receipt_documents.csv"

    # items
    items_df = df.copy()
    items_df.insert(0, "document_id", document_id)
    items_df.to_csv(items_out, index=False)

    # documents
    doc_row = [{
        "document_id": document_id,
        "doc_type": "receipt",
        "document_number_raw": None,
        "document_number": None,
        "party_name": None,
        "supplier_name": merchant,
        "currency": meta.get("currency") or currency_default,
        "extraction_flavor": meta.get("extraction_flavor"),

        "items_count": int(len(df)),
        "sum_line_total": round(sum_line_total, 2),

        "subtotal_meta": subtotal_meta,
        "tax_meta_pct": tax_pct,
        "total_meta": total_meta,

        "diff_sum_vs_subtotal": diff_sum_vs_subtotal,
        "diff_sum_vs_total": diff_sum_vs_total,

        "source_path": meta.get("source_path"),
        "ingest_ts": meta.get("ingest_ts"),
        "silver_ts": datetime.utcnow().isoformat(timespec="seconds"),
    }]
    docs_df = pd.DataFrame(doc_row).reindex(columns=DOCS_KEEP)
    docs_df.to_csv(docs_out, index=False)

    return {
        "document_id": document_id,
        "items_path": str(items_out),
        "documents_path": str(docs_out),
        "validations": validations,
    }

def aggregate_receipts_silver(bronze_root: Path, silver_dir: Path) -> Dict[str, Any]:
    """
    Walk all per-image bronze/receipts/<stem> folders, promote to _silver,
    then concatenate into data/output/silver/receipts/{receipt_items.csv, receipt_documents.csv}
    """
    silver_dir.mkdir(parents=True, exist_ok=True)
    items_all: List[pd.DataFrame] = []
    docs_all: List[pd.DataFrame] = []

    for stem_dir in sorted(bronze_root.iterdir()):
        if not stem_dir.is_dir():
            continue
        if not (stem_dir / "receipt_meta.json").exists():
            continue

        per_doc_silver = silver_dir / stem_dir.name
        per_doc_silver.mkdir(parents=True, exist_ok=True)

        _ = bronze_to_silver(bronze_dir=stem_dir, out_dir=per_doc_silver)

        items_df = pd.read_csv(per_doc_silver / "receipt_items.csv")
        docs_df  = pd.read_csv(per_doc_silver / "receipt_documents.csv")

        docs_df = docs_df.reindex(columns=DOCS_KEEP, fill_value=pd.NA)

        items_all.append(items_df)
        docs_all.append(docs_df)

    if not items_all or not docs_all:
        return {"status": "no_receipts"}

    items_nonempty = [df for df in items_all if not df.empty]
    docs_nonempty  = [df for df in docs_all  if not df.empty]
    if not items_nonempty or not docs_nonempty:
        return {"status": "no_receipts"}

    items_cat = pd.concat(items_nonempty, ignore_index=True)
    docs_cat  = pd.concat(docs_nonempty,  ignore_index=True)

    items_out = silver_dir / "receipt_items.csv"
    docs_out = silver_dir / "receipt_documents.csv"
    items_cat.to_csv(items_out, index=False)
    docs_cat.to_csv(docs_out, index=False)

    return {
        "items_path": str(items_out),
        "documents_path": str(docs_out),
        "items_count": int(len(items_cat)),
        "documents_count": int(len(docs_cat)),
    }
