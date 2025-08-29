from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import re
import hashlib


# -----------------------------
# Helpers: parse, ids, money
# -----------------------------

def _money_to_float(x):
    """Convert money-like strings to float. Supports '1,234.56' and '1.234,56'."""
    if x is None or pd.isna(x):
        return pd.NA
    t = str(x).strip()
    t = re.sub(r"[^\d,.\-]", "", t)
    if "," in t and "." in t:
        t = t.replace(",", "")                 # US style -> 1,234.56
    elif "," in t and "." not in t:
        t = t.replace(".", "").replace(",", ".")  # 1.234,56 -> 1234.56
    try:
        return float(t)
    except Exception:
        return pd.NA


def _is_number(x) -> bool:
    """True only if x is not None and not NA."""
    return (x is not None) and (not pd.isna(x))


def _sanitize_id(s: str) -> str:
    """Uppercase and keep only A–Z, 0–9 and dashes."""
    return re.sub(r"[^A-Z0-9\-]+", "", s.upper())


def _token_with_digit(number_raw: str) -> str | None:
    """
    Return the first token containing at least one digit (e.g. '1', 'INV-001').
    Works for '1 ABC ...' or 'INV-2025-001 ACME ...'.
    """
    if not number_raw:
        return None
    tokens = re.findall(r"[A-Za-z0-9\-]+", number_raw.upper())
    for tok in tokens:
        if any(ch.isdigit() for ch in tok):
            return tok
    return None


def _clean_party_name(name: str) -> str:
    """Remove noise like 'Invoice #:' and leading pure numeric tokens."""
    if not name:
        return ""
    t = str(name)
    t = re.sub(r"^\s*invoice\s*#?\s*:?\s*", "", t, flags=re.I)
    t = re.sub(r"^\s*\d+\s+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _vendor_slug(name: str) -> str:
    """Compact vendor/client name to an 8-char slug (A–Z0–9)."""
    if not name:
        return "UNKNOWN"
    slug = re.sub(r"[^A-Z0-9]", "", name.upper())
    return (slug[:8] or "UNKNOWN")


def _date_to_yyyymmdd(s: str) -> str | None:
    """Parse dates like DD/MM/YYYY, MM/DD/YYYY, DD-MM-YYYY, into YYYYMMDD."""
    if not s:
        return None
    m = re.search(r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})", s)
    if not m:
        return None
    a, b, y = m.groups()
    a = int(a); b = int(b); y = int(y)
    if y < 100:
        y += 2000
    d, mo = (a, b)  # prefer DD/MM over MM/DD for invoices
    return f"{y:04d}{mo:02d}{d:02d}"


def _hash12(*parts: str) -> str:
    """Deterministic 12-char SHA1 hex for stable fallback IDs."""
    base = "|".join(p.strip() for p in parts if p)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12].upper()


def make_document_id(meta: dict, source_path: str) -> tuple[str, str, str | None]:
    """
    Build a robust, unique document_id.

    Returns:
      (document_id, document_number_raw, document_number_norm)

    Strategy:
      - number_raw := meta['document_number_raw'] if present; else meta['document_id']; else "".
      - document_number_norm := first token in raw that has digits (e.g., '1', 'INV-001').
      - vendor := cleaned 'bill_from_name' (preferred) else 'client_or_supplier'.
      - If document_number_norm exists:
          doc_id = "{VENDOR8}-{document_number_norm}" [+ "-YYYYMMDD" if invoice_date available]
        Else:
          doc_id = "{VENDOR8}-{HASH12(stem|vendor|date)}"
    """
    number_raw = (meta.get("document_number_raw") or meta.get("document_id") or "").strip()
    number_norm = _token_with_digit(number_raw)

    party_raw = (meta.get("bill_from_name") or meta.get("client_or_supplier") or "").strip()
    party_clean = _clean_party_name(party_raw) or (meta.get("client_or_supplier") or "").strip()
    vendor8 = _vendor_slug(party_clean)

    inv_date = (meta.get("invoice_date") or "").strip()
    ymd = _date_to_yyyymmdd(inv_date)
    stem = Path(source_path or "unknown").stem

    if number_norm:
        core = f"{vendor8}-{number_norm}"
        if ymd:
            core = f"{core}-{ymd}"
        return _sanitize_id(core), number_raw, number_norm

    h = _hash12(stem, vendor8, ymd or "")
    return _sanitize_id(f"{vendor8}-{h}"), number_raw, None


def _supplier_from_number_raw(number_raw: str | None) -> str | None:
    """
    From a string like '1 ABC Construction Supplies Ltd.' return 'ABC Construction Supplies Ltd.'.
    Generic rule: drop the *first* token and return the rest.
    """
    if not number_raw:
        return None
    s = str(number_raw).strip()
    # drop a leading 'Invoice #:' if somehow present here
    s = re.sub(r"^\s*(invoice\s*(no\.?|number|#)\s*[:#]?)\s*", "", s, flags=re.I).strip()
    m = re.match(r"^\s*\S+\s+(.+)$", s)
    if m:
        out = re.sub(r"\s+", " ", m.group(1)).strip()
        return out or None
    return None


# -----------------------------
# Silver builder
# -----------------------------

def bronze_to_silver(bronze_dir: Path, out_dir: Path) -> dict:
    """
    Transform bronze artifacts (invoice_line_items.csv + invoice_meta.json) into silver:
      - invoice_items.csv  (WITH document_id column)
      - invoice_documents.csv

    NOTE: If you run this multiple times pointing to the same out_dir, files will be overwritten.
    For batch runs, use the ingest script with staging/concatenation.
    """
    bronze_dir = Path(bronze_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load bronze
    items_bronze = pd.read_csv(bronze_dir / "invoice_line_items.csv")
    meta = json.loads((bronze_dir / "invoice_meta.json").read_text(encoding="utf-8"))

    # Robust document_id
    doc_id, document_number_raw, document_number_norm = make_document_id(
        meta, meta.get("source_path", "unknown")
    )

    # Supplier name: bill_from_name (clean) > document_number_raw tail > client_or_supplier (clean)
    supplier_name = meta.get("bill_from_name")
    supplier_name = _clean_party_name(supplier_name) if supplier_name else None
    if not supplier_name:
        supplier_name = _supplier_from_number_raw(document_number_raw)
    if not supplier_name:
        supplier_name = _clean_party_name(meta.get("client_or_supplier"))


    # Normalize numeric columns and ensure consistent schema
    items = items_bronze.copy()
    for c in ["quantity", "unit_price", "line_total"]:
        if c not in items.columns:
            items[c] = pd.NA

    items["quantity"] = pd.to_numeric(items["quantity"], errors="coerce")
    items["unit_price"] = items["unit_price"].map(_money_to_float)
    items["line_total"] = items["line_total"].map(_money_to_float)

    # Compute line_total when missing but quantity & unit_price exist
    need_total = items["line_total"].isna() & items["unit_price"].notna() & items["quantity"].notna()
    items.loc[need_total, "line_total"] = items.loc[need_total, "unit_price"] * items.loc[need_total, "quantity"]

    # Prepend document_id column
    items.insert(0, "document_id", doc_id)

    # Write silver items with exact column order expected by the loader
    items_cols = ["document_id", "item", "description", "quantity", "unit_price", "line_total"]
    items = items[items_cols]
    items_csv = out_dir / "invoice_items.csv"
    items.to_csv(items_csv, index=False)

    # Build silver documents row
    sum_line_total = float(items["line_total"].fillna(0).sum())

    # Read raw meta values
    subtotal_meta   = _money_to_float(meta.get("subtotal"))
    amount_due_meta = _money_to_float(meta.get("amount_due"))
    total_meta      = _money_to_float(meta.get("total"))  # may be missing in your PDFs

    # Tax: accept "6%", "6,00%", 6, 6.0, or 0.06 (normalize to 6.0)
    tax_meta_pct = None
    tax_raw = meta.get("tax")
    if tax_raw is not None:
        if isinstance(tax_raw, str):
            t = tax_raw.replace("%", "").strip().replace(",", ".")
            try:
                tax_meta_pct = float(t)
            except Exception:
                tax_meta_pct = None
        elif isinstance(tax_raw, (int, float)):
            tax_meta_pct = float(tax_raw)

    # If value looks like a fraction (<= 1), scale to percent
    if tax_meta_pct is not None and not pd.isna(tax_meta_pct) and abs(tax_meta_pct) <= 1:
        tax_meta_pct = tax_meta_pct * 100.0


    # ---- Fallbacks for total_meta ----
    def _is_num(x):  # helper to dodge "boolean value of NA is ambiguous"
        return x is not None and not pd.isna(x)

    if not _is_num(total_meta):
        if _is_num(subtotal_meta) and _is_num(tax_meta_pct):
            total_meta = float(subtotal_meta) * (1.0 + float(tax_meta_pct)/100)
        elif _is_num(amount_due_meta):
            total_meta = float(amount_due_meta)

    # Diffs
    diff_sum_vs_subtotal = sum_line_total - subtotal_meta if _is_num(subtotal_meta) else None
    diff_sum_vs_total = sum_line_total - total_meta if _is_num(total_meta) else None


    documents = pd.DataFrame([{
        "document_id": doc_id,
        "doc_type": "invoice",
        "document_number_raw": document_number_raw,     # raw text for transparency
        "document_number": document_number_norm,        # normalized token with digits (if any)
        "party_name": meta.get("client_or_supplier"),
        "supplier_name": supplier_name,
        "currency": meta.get("currency"),
        "extraction_flavor": meta.get("extraction_flavor"),
        "items_count": int(len(items)),
        "sum_line_total": sum_line_total,
        "subtotal_meta": subtotal_meta if _is_number(subtotal_meta) else None,
        "tax_meta_pct": tax_meta_pct,
        # "amount_due_meta": amount_due_meta if _is_number(amount_due_meta) else None,
        "total_meta": total_meta if _is_number(total_meta) else None,
        "diff_sum_vs_subtotal": diff_sum_vs_subtotal,
        "diff_sum_vs_total": diff_sum_vs_total,
        "source_path": meta.get("source_path"),
        "ingest_ts": meta.get("ingest_ts"),
        "silver_ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }], columns=[
        "document_id", "doc_type", "document_number_raw", "document_number",
        "party_name", "supplier_name", "currency", "extraction_flavor",
        "items_count", "sum_line_total", "subtotal_meta", "tax_meta_pct",
        "total_meta", "diff_sum_vs_subtotal", "diff_sum_vs_total",
        "source_path", "ingest_ts", "silver_ts"
    ])

    docs_csv = out_dir / "invoice_documents.csv"
    documents.to_csv(docs_csv, index=False)

    return {"items_csv": str(items_csv), "documents_csv": str(docs_csv)}
