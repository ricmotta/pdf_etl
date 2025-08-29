# src/pipelines/invoices/extract.py
from __future__ import annotations

import re
import json
import warnings
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import camelot
import pdfplumber

# Quiet down noisy libs on some PDFs
warnings.filterwarnings("ignore", message="CropBox missing from /Page")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# ----------------------------- heuristics & tokens -----------------------------
SUMMARY_WORDS = ("subtotal", "tax", "amount due", "total:")
META_WORDS = (
    "invoice", "invoice #", "invoice no", "invoice number", "invoice date",
    "po #", "purchase order", "due date",
    "bill to", "bill from", "email", "@", "phone", "tel", "address"
)

TOKENS = {
    "item": ["item", "item name", "product", "code"],
    "description": ["description", "desc", "details"],
    "quantity": ["qty", "quantity", "amount"],
    "unit_price": ["unit price", "price", "rate"],
    "line_total": ["total", "amount"],
}

MONEY_RE = re.compile(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})")
NUM_RE   = re.compile(r"\b\d+(?:\.\d+)?\b")


# ----------------------------- numeric cleaners -------------------------------
def _money_to_float(x):
    """Convert money-like tokens to float, handling $1,234.56 and 1.234,56 formats."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NA
    t = re.sub(r"[^\d,.\-]", "", str(x))
    if "," in t and "." in t:
        t = t.replace(",", "")
    elif "," in t and "." not in t:
        t = t.replace(".", "").replace(",", ".")
    try:
        return float(t)
    except Exception:
        return pd.NA

def _clean_money_series(s: pd.Series) -> pd.Series:
    return s.map(_money_to_float)


# ----------------------------- small helpers ----------------------------------
def _is_meta(text: str) -> bool:
    """Heuristic: row looks like header/billing metadata (not a line item)."""
    t = (text or "").lower()
    return any(k in t for k in META_WORDS)

def _is_summary(text: str) -> bool:
    """Heuristic: row looks like totals/subtotal/summary."""
    t = (text or "").lower()
    return any(k in t for k in SUMMARY_WORDS)

def _score_numeric(df: pd.DataFrame) -> int:
    """Count how many numeric cells (qty/unit/total) are present; used to pick the best parse."""
    if df is None or df.empty:
        return 0
    cols = [c for c in ["quantity", "unit_price", "line_total"] if c in df.columns]
    return int(df[cols].notna().sum().sum()) if cols else 0

def _extract_same_row_numbers(row_text: str):
    """
    Parse numbers from a single text line:
      - first money token       -> unit_price
      - plain number            -> quantity (prefer one between unit and total)
      - last money token        -> line_total (if there are two)
    Returns (unit_price, quantity, line_total); any of them can be pd.NA.
    """
    money = [(m.group(), m.start(), m.end()) for m in MONEY_RE.finditer(row_text)]
    nums  = [(n.group(), n.start(), n.end()) for n in NUM_RE.finditer(row_text)]

    # remove numbers that are part of a money token
    def inside(a_start, a_end, b_start, b_end):
        return a_start >= b_start and a_end <= b_end
    nums = [n for n in nums if not any(inside(n[1], n[2], m[1], m[2]) for m in money)]

    unit_price = pd.NA
    quantity   = pd.NA
    line_total = pd.NA

    if money:
        unit_price = _money_to_float(money[0][0])
        if len(money) >= 2:
            line_total = _money_to_float(money[-1][0])

        # choose quantity
        candidates = []
        if nums:
            if len(money) >= 2:
                # prefer a number BETWEEN unit and total
                left = money[0][2]; right = money[-1][1]
                candidates = [n for n in nums if n[1] >= left and n[2] <= right]
            if not candidates:
                # else first number to the right of unit; else first number anywhere
                candidates = [n for n in nums if n[1] >= money[0][2]] or nums

            try:
                quantity = float(candidates[0][0])
            except Exception:
                quantity = pd.NA

    return unit_price, quantity, line_total


# ----------------------------- header text / meta ------------------------------
def _extract_header_text(pdf_path: Path) -> str:
    """Concatenate plain text from all pages (works for text-based PDFs)."""
    chunks = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for p in pdf.pages:
            chunks.append(p.extract_text() or "")
    return "\n".join(chunks)

def _parse_invoice_meta(text: str) -> dict:
    """
    Regex-based parsing of common invoice metadata.

    Important: we DO NOT set 'document_id' in bronze to avoid propagating bad IDs like 'ABC'.
    Instead we capture:
      - 'document_number_raw' (full text after the label)
      - 'document_number' (parsed token when confidently detected)
    The silver layer decides the final 'document_id' (sanitized or hashed fallback).
    """
    meta = {}

    def _extract_doc_number_token(raw: str):
        """
        Extract a leading ID-like token from a captured line:
          - Case A: starts with digits (allow spaces/hyphens) -> join digits (e.g., "123 456-789" -> "123456789")
          - Case B: starts with alnum/hyphen token -> return that token (e.g., "INV-2025-001")
          - Fallback: first alnum/hyphen token anywhere in the line.
        """
        s = (raw or "").strip()

        # A) starts with digits
        m = re.match(r"^\s*([0-9][0-9\-\s]{0,80})", s)
        if m:
            num = re.sub(r"[\s\-]", "", m.group(1))
            if 1 <= len(num) <= 40:
                return num.upper()

        # B) starts with alnum/hyphen token
        m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9\-]{1,80})", s)
        if m:
            return m.group(1).upper()

        # Fallback
        m = re.search(r"([A-Za-z0-9][A-Za-z0-9\-]{1,80})", s)
        if m:
            return m.group(1).upper()

        return None

    # --- Document / PO numbers: capture to EOL, store raw + parsed token ---
    m = re.search(r"invoice\s*(?:no\.?|number|#)\s*[:#]?\s*([^\n\r]+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        meta["document_number_raw"] = raw
        token = _extract_doc_number_token(raw)
        if token:
            meta["document_number"] = token  # <-- keep as helper for silver

    m = re.search(r"\bpo\s*(?:no\.?|number|#)?\s*[:#]?\s*([^\n\r]+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        meta["po_number_raw"] = raw
        token = _extract_doc_number_token(raw)
        if token:
            meta["po_number"] = token

    # --- Dates (DD/MM/YYYY or MM/DD/YYYY) ---
    m = re.search(r"invoice\s*date[:\s]*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})", text, re.IGNORECASE)
    if m:
        meta["invoice_date"] = m.group(1)
    m = re.search(r"due\s*date[:\s]*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})", text, re.IGNORECASE)
    if m:
        meta["due_date"] = m.group(1)

    # --- Parties (capture block after label until next known label/blank line) ---
    def capture_block(label):
        mm = re.search(
            label + r"\s*:?\s*(.+?)(?:\n\s*\n|bill\s*from|bill\s*to|invoice|po\s*#?|due\s*date|invoice\s*date|$)",
            text, re.IGNORECASE | re.DOTALL
        )
        if not mm:
            return None
        block = "\n".join([ln.strip() for ln in mm.group(1).splitlines() if ln.strip()])
        name = block.splitlines()[0] if block else None
        return name, block

    bt = capture_block(r"bill\s*to")
    bf = capture_block(r"bill\s*from")
    if bt:
        meta["bill_to_name"] = bt[0]
        meta["bill_to_block"] = bt[1]
        meta["client_or_supplier"] = bt[0]
    if bf:
        meta["bill_from_name"] = bf[0]
        meta["bill_from_block"] = bf[1]
        meta.setdefault("client_or_supplier", bf[0])

    # --- Money-like fields (keep strings; silver normalizes) ---
    for key, patt in [
        ("amount_due", r"amount\s*due\s*[:\s]*\$?([0-9,.\-]+)"),
        ("subtotal",   r"\bsubtotal\s*[:\s]*\$?([0-9,.\-]+)"),
        ("tax",        r"\btax\s*[:\s]*([0-9,.\-%]+)"),
        ("total",      r"\btotal\s*[:\s]*\$?([0-9,.\-]+)"),
    ]:
        m = re.search(patt, text, re.IGNORECASE)
        if m:
            meta[key] = m.group(1)

    # --- Currency heuristic ---
    low = text.lower()
    if "€" in text or re.search(r"\beur\b", low):
        meta["currency"] = "EUR"
    elif "£" in text or re.search(r"\bgbp\b", low):
        meta["currency"] = "GBP"
    elif "r$" in low or re.search(r"\bbrl\b", low):
        meta["currency"] = "BRL"
    elif "$" in text or re.search(r"\busd\b", low):
        meta["currency"] = "USD"
    else:
        meta["currency"] = "UNKNOWN"

    return meta


# ----------------------------- header-based normalization ----------------------
def _find_header_index(df: pd.DataFrame, scan_rows: int = 60):
    """Find header row index; also detect if header spans two rows."""
    n = len(df)
    if n <= 0:
        return None, False

    def score(txt: str) -> int:
        low = txt.lower()
        groups = [
            any(k in low for k in TOKENS["item"]),
            any(k in low for k in TOKENS["description"]),
            any(k in low for k in TOKENS["unit_price"]),
            any(k in low for k in TOKENS["quantity"]),
            any(k in low for k in TOKENS["line_total"]),
        ]
        return sum(groups)

    best = (-1, False, -1)  # (score, two_line, idx)
    scan = min(scan_rows, max(n - 1, 0))
    for i in range(scan + 1):
        row_i = " ".join(str(x) for x in df.iloc[i].tolist())
        s1 = score(row_i)
        if s1 > best[0]:
            best = (s1, False, i)
        if i + 1 < n:
            row_2 = " ".join(str(x) for x in df.iloc[i + 1].tolist())
            s2 = score(row_i + " " + row_2)
            if s2 > best[0]:
                best = (s2, True, i)

    if best[0] >= 2 and best[2] >= 0:
        return best[2], best[1]

    # fallback: look for "item" and "description" nearby
    for i in range(n - 1):
        row_i = " ".join(str(x) for x in df.iloc[i].tolist()).lower()
        if ("item" in row_i and "description" in row_i) or ("item name" in row_i):
            two = False
            if i + 1 < n:
                row_2 = " ".join(str(x) for x in df.iloc[i + 1].tolist()).lower()
                if ("price" in row_2 or "rate" in row_2) or ("qty" in row_2 or "quantity" in row_2):
                    two = True
            return i, two
    return None, False

def _build_header(df: pd.DataFrame, idx: int, two_line: bool):
    """Create the DataFrame header from row idx (optionally merged with next row)."""
    cols1 = [str(x).strip() for x in df.iloc[idx].tolist()]
    if two_line and idx + 1 < len(df):
        cols2 = [str(x).strip() for x in df.iloc[idx + 1].tolist()]
        if len(cols2) < len(cols1):
            cols2 += [""] * (len(cols1) - len(cols2))
        cols = [(a if a else b) if (a or b) else "" for a, b in zip(cols1, cols2)]
        new_df = df.iloc[idx + 2:].reset_index(drop=True)
    else:
        cols = cols1
        new_df = df.iloc[idx + 1:].reset_index(drop=True)

    cols = [re.sub(r"\s+", " ", c).strip() for c in cols]
    new_df.columns = cols
    return new_df

def _pick_col(cols_lower, *candidates):
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None

def _map_columns(df: pd.DataFrame):
    """Map (case-insensitive) source column names to the target schema."""
    cols_lower = {c.lower(): c for c in df.columns}
    c_item       = _pick_col(cols_lower, *TOKENS["item"])
    c_desc       = _pick_col(cols_lower, *TOKENS["description"])
    c_qty        = _pick_col(cols_lower, *TOKENS["quantity"])
    c_unit_price = _pick_col(cols_lower, *TOKENS["unit_price"])
    c_total      = _pick_col(cols_lower, *TOKENS["line_total"])
    return c_item, c_desc, c_qty, c_unit_price, c_total

def _normalize_via_header(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a table using a detected header row."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["item", "description", "quantity", "unit_price", "line_total"])
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    idx, two = _find_header_index(df)
    if idx is None:
        return pd.DataFrame(columns=["item", "description", "quantity", "unit_price", "line_total"])

    df = _build_header(df, idx, two)
    c_item, c_desc, c_qty, c_unit_price, c_total = _map_columns(df)
    n = len(df)

    def s(col, default_str=False, numeric=False, money=False):
        if col and col in df:
            series = df[col]
            if numeric:
                return pd.to_numeric(series, errors="coerce")
            if money:
                return _clean_money_series(series)
            return series
        return pd.Series([""] * n, index=df.index) if default_str else pd.Series([pd.NA] * n, index=df.index)

    out = pd.DataFrame({
        "item":        s(c_item, default_str=True).astype(str).str.strip(),
        "description": s(c_desc, default_str=True).astype(str).str.strip(),
        "quantity":    s(c_qty, numeric=True),
        "unit_price":  s(c_unit_price, money=True),
        "line_total":  s(c_total, money=True),
    })

    # remove headers/summary/empty rows
    is_summary = out[["item", "description"]].fillna("").apply(
        lambda col: col.astype(str).str.lower().map(_is_summary)
    ).any(axis=1)
    empty_text = (out[["item", "description"]].fillna("").apply(lambda c: c.astype(str).str.strip()) == "").all(axis=1)
    empty_nums = out[["quantity", "unit_price", "line_total"]].isna().all(axis=1)
    out = out[~(is_summary | (empty_text & empty_nums))].reset_index(drop=True)

    # compute total when missing and possible
    need_total = out["line_total"].isna() & out["unit_price"].notna() & out["quantity"].notna()
    out.loc[need_total, "line_total"] = out.loc[need_total, "unit_price"] * out.loc[need_total, "quantity"]
    return out


# ----------------------------- regex fallback (2-line) -------------------------
def _normalize_via_regex_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    No header path. First try to extract numbers on the SAME LINE; if something is missing,
    look into the NEXT LINE (common pattern in stream-extracted tables).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["item", "description", "quantity", "unit_price", "line_total"])

    g = df.fillna("").astype(str)
    rows = []
    i = 0
    n = len(g)

    while i < n:
        row_vals = g.iloc[i].tolist()
        row_text = " ".join(row_vals).strip()

        # skip metadata/header-like rows
        if _is_meta(row_text) or ("item" in row_text.lower() and "description" in row_text.lower()):
            i += 1
            continue

        # item/description = first two columns when available
        item = row_vals[0].strip() if len(row_vals) > 0 else ""
        desc = row_vals[1].strip() if len(row_vals) > 1 else ""

        if _is_meta(item) or _is_meta(desc) or _is_summary(item) or _is_summary(desc):
            i += 1
            continue

        # 1) same-line extraction
        unit_price, quantity, line_total = _extract_same_row_numbers(row_text)

        # 2) complement from next line if needed
        if (pd.isna(quantity) or pd.isna(line_total)) and (i + 1 < n):
            nxt_vals = g.iloc[i + 1].tolist()
            nxt_text = " ".join(nxt_vals).strip()
            if not _is_meta(nxt_text):
                money_nxt = MONEY_RE.findall(nxt_text)
                nums_nxt  = [q for q in NUM_RE.findall(nxt_text) if not MONEY_RE.search(q)]
                if pd.isna(quantity) and nums_nxt:
                    try:
                        quantity = float(nums_nxt[0])
                    except Exception:
                        quantity = pd.NA
                if pd.isna(line_total) and money_nxt:
                    line_total = _money_to_float(money_nxt[0])

                # consume next row if it contributed values
                if money_nxt or nums_nxt:
                    i += 1

        rows.append({
            "item": item, "description": desc,
            "quantity": quantity, "unit_price": unit_price, "line_total": line_total
        })
        i += 1

    out = pd.DataFrame(rows, columns=["item", "description", "quantity", "unit_price", "line_total"])

    # final cleanup
    is_summary = out[["item", "description"]].fillna("").apply(
        lambda col: col.astype(str).str.lower().map(_is_summary)
    ).any(axis=1)
    empty_text = (out[["item", "description"]].fillna("").apply(lambda c: c.astype(str).str.strip()) == "").all(axis=1)
    empty_nums = out[["quantity", "unit_price", "line_total"]].isna().all(axis=1)
    out = out[~(is_summary | (empty_text & empty_nums))].reset_index(drop=True)

    need_total = out["line_total"].isna() & out["unit_price"].notna() & out["quantity"].notna()
    out.loc[need_total, "line_total"] = out.loc[need_total, "unit_price"] * out.loc[need_total, "quantity"]
    return out


# ----------------------------- table reading -----------------------------------
def _read_tables(pdf_path: Path):
    """Try Camelot lattice first (bordered tables); fallback to stream."""
    try:
        t_lat = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice", line_scale=40)
        if t_lat and len(t_lat) > 0:
            return "lattice", t_lat
    except Exception:
        pass
    try:
        t_str = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream", strip_text="\n")
        if t_str and len(t_str) > 0:
            return "stream", t_str
    except Exception:
        pass
    return "none", []


# ----------------------------- public API --------------------------------------
def extract_invoice(pdf_path: Path, out_dir: Path) -> dict:
    """
    High-level bronze extractor:
      1) Read tables with Camelot.
      2) Normalize items via header path; fallback to regex pairs if needed.
      3) Write bronze CSV + meta JSON. We do NOT set 'document_id' here.
         We store 'document_number_raw' (and 'document_number' if detected).
         The silver layer will compute the final 'document_id'.
    """
    flavor, tables = _read_tables(pdf_path)

    best_df = None
    best_score = (-1, -1)  # (numeric_cells, rows)

    for t in (tables or []):
        raw = t.df.dropna(how="all")
        # path 1: header-based
        tidy = _normalize_via_header(raw)
        score = (_score_numeric(tidy), len(tidy))
        # fallback: lightweight regex (two-line) if weak/empty
        if tidy.empty or score[0] == 0:
            fallback = _normalize_via_regex_pairs(raw)
            score_fb = (_score_numeric(fallback), len(fallback))
            if score_fb > score:
                tidy, score = fallback, score_fb

        if score > best_score:
            best_df, best_score = tidy, score

    items = best_df if best_df is not None else pd.DataFrame(
        columns=["item", "description", "quantity", "unit_price", "line_total"]
    )

    # persist bronze
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_csv = out_dir / "invoice_line_items.csv"
    meta_json  = out_dir / "invoice_meta.json"
    items.to_csv(tables_csv, index=False)

    header_text = _extract_header_text(pdf_path)
    meta = _parse_invoice_meta(header_text)
    meta.update({
        "source_path": str(pdf_path),
        "extraction_flavor": flavor,
        "ingest_ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "line_items_rows": int(len(items)),
    })
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "tables_csv": str(tables_csv),
        "meta_json": str(meta_json),
        "rows": int(len(items)),
        "extraction_flavor": flavor,
    }
