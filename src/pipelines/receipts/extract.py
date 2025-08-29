# src/pipelines/receipts/extract.py
# -*- coding: utf-8 -*-

"""
Receipt OCR extractor (Bronze)
- Stronger OCR: try multiple input variants (gray / binarized / final from OpenCV preprocess),
  then run multi-pass Tesseract (psm 4/6/11), score by keywords and pick the best.
- Outputs (per image):
    data/output/bronze/receipts/<stem>/receipt_line_items.csv
    data/output/bronze/receipts/<stem>/receipt_meta.json
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import pytesseract
import csv

# OpenCV-based preprocess (writes _debug images)
from src.pipelines.receipts.preprocess import preprocess_receipt, PreprocessConfig


# ---------------------------
# Tesseract configuration
# ---------------------------

def _configure_tesseract_from_env() -> None:
    """Allow explicit Tesseract binary via OCR_TESSERACT_CMD or common Windows path."""
    cmd: Optional[str] = os.getenv("OCR_TESSERACT_CMD")
    if cmd and os.path.exists(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd
        return
    common_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and os.path.exists(common_win):
        pytesseract.pytesseract.tesseract_cmd = common_win

_configure_tesseract_from_env()


# ---------------------------
# OCR helpers
# ---------------------------

_KEYWORDS = [
    "subtotal", "total", "tax", "change due", "cash tend", "debit tend",
    "items sold", "thank you", "manager", "terminal", "account", "tc#"
]

def _score_receipt_text(text: str) -> int:
    """Score OCR text by presence of typical receipt keywords."""
    low = text.lower()
    return sum(1 for kw in _KEYWORDS if kw in low)


def _run_ocr_best_single_image(img: Image.Image) -> Tuple[str, str, int]:
    """
    Run multiple Tesseract configs over a single image.
    Returns (merged_text, best_config, score).
    """
    configs = [
        "--oem 3 --psm 4 -l eng",   # columns/varied text
        "--oem 3 --psm 6 -l eng",   # uniform block
        "--oem 3 --psm 11 -l eng",  # sparse text
    ]
    results = []  # (score, cfg, text)

    # Collect texts for each config
    texts: List[Tuple[str, str]] = []
    for cfg in configs:
        try:
            txt = pytesseract.image_to_string(img, config=cfg)
        except (pytesseract.pytesseract.TesseractNotFoundError, OSError):
            raise RuntimeError(
                "tesseract is not installed or it's not in your PATH. "
                "Install Tesseract and/or set OCR_TESSERACT_CMD."
            )
        texts.append((cfg, txt))
        results.append((_score_receipt_text(txt), cfg, txt))

    # Pick best cfg by keyword score
    results.sort(reverse=True, key=lambda x: x[0])
    best_cfg = results[0][1]

    # Merge unique non-empty lines across all cfgs
    seen = set()
    merged_lines = []
    for _, txt in texts:
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            key = re.sub(r"\s+", " ", ln)
            if key.lower() not in seen:
                seen.add(key.lower())
                merged_lines.append(ln)
    merged = "\n".join(merged_lines)

    # Score the merged version as well
    merged_score = _score_receipt_text(merged)
    return merged, best_cfg, merged_score


def _run_ocr_multi_inputs(gray_img: Optional[Image.Image],
                          bin_img: Optional[Image.Image],
                          final_img: Optional[Image.Image]) -> Tuple[str, str, str]:
    """
    Try OCR on multiple input variants and pick the best-scoring text.
    Returns (best_text, flavor_label, input_variant).
    """
    candidates: List[Tuple[str, Image.Image]] = []
    if gray_img is not None:
        candidates.append(("gray", gray_img))
    if bin_img is not None:
        candidates.append(("bin", bin_img))
    if final_img is not None:
        candidates.append(("final", final_img))

    if not candidates:
        raise RuntimeError("No image candidates available for OCR.")

    scored: List[Tuple[int, str, str, str]] = []  # (score, variant, text, cfg)
    for variant, im in candidates:
        text, cfg, sc = _run_ocr_best_single_image(im)
        scored.append((sc, variant, text, cfg))

    scored.sort(reverse=True, key=lambda x: x[0])
    best_sc, best_var, best_text, best_cfg = scored[0]
    return best_text, f"{best_var}:{best_cfg}", best_var


# ---------------------------
# Parsing helpers
# ---------------------------

_CURRENCY_RE = r"[$€£R\$]"
_NUMBER_ANY = r"[+-]?(?:\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2})|\d+)"

def _parse_number(s: str) -> Optional[float]:
    """Parse numbers tolerant to US/EU formats."""
    if s is None:
        return None
    s = s.strip()
    s = re.sub(rf"({_CURRENCY_RE})", "", s)
    s = s.replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        if "," in s and "." not in s:
            s = s.replace(".", "").replace(",", ".")
        elif s.count(",") > 0 and s.count(".") == 1:
            s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _extract_header(text: str) -> Dict[str, Optional[str]]:
    """
    Extract merchant, TC code, txn datetime (raw), phone, and items_sold.
    Tolerant regexes for noisy OCR.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    merchant = None

    # Merchant: scan whole text for known brands
    if re.search(r"\bwal[-\s\*]*mart\b", text, flags=re.I):
        merchant = "WALMART"
    elif re.search(r"\btrader\s*joe'?s?\b", text, flags=re.I):
        merchant = "TRADER JOE'S"
    else:
        # fallback: first meaningful line
        for ln in lines[:10]:
            if len(ln) >= 3 and not re.search(r"survey|feedback|thank|manager|mgr", ln, flags=re.I):
                merchant = re.sub(r"[^A-Za-z0-9 '&\-/\.]", "", ln).strip()
                if merchant:
                    break

    # TC code near bottom
    tc_code = None
    for ln in lines[-40:]:
        m = re.search(r"\bTC[#:\s]+([0-9\s]{10,})", ln, flags=re.I)
        if m:
            tc_code = re.sub(r"\s+", " ", m.group(1)).strip()
            break

    # Phone (NANP-like), reject clearly invalid area codes (0/1xx)
    phone = None
    candidates = []
    for ln in lines[:40]:
        for m in re.finditer(r"\(?\s*(\d{3})\s*\)?\D*\s*(\d{3})\D*\s*(\d{4})", ln):
            candidates.append((m.group(1), m.group(2), m.group(3)))
    if candidates:
        def score(p):
            a,b,c = p
            bad = (a.startswith(("0","1")) or a == "000" or b == "000" or c == "0000")
            zeros = (a+b+c).count("0")
            return (bad, zeros)
        a,b,c = sorted(candidates, key=score)[0]
        if not a.startswith(("0","1")):
            phone = f"({a}) {b}-{c}"

    # Date/time
    dt_regexes = [
        r"(\d{2}[/-]\d{2}[/-]\d{2,4})\s+(\d{1,2}:\d{2}(?::\d{2})?\s?[APMapm]{0,2})",
        r"(\d{2}[/-]\d{2}[/-]\d{2,4})\s+(\d{1,2}:\d{2}(?::\d{2})?)",
        r"(\d{2}[/-]\d{2}[/-]\d{2,4})",
    ]
    txn_date = None
    for ln in lines[::-1]:
        for rx in dt_regexes:
            m = re.search(rx, ln)
            if m:
                txn_date = " ".join([g for g in m.groups() if g]).strip()
                break
        if txn_date:
            break

    # Items sold (# optional; may be on next line)
    items_sold = None
    for i, ln in enumerate(lines):
        if re.search(r"(?:#\s*)?ITEMS\s*SOLD\b", ln, flags=re.I):
            m = re.search(r"(?:#\s*)?ITEMS\s*SOLD\s*[:\-]?\s*(\d+)", ln, flags=re.I)
            if m:
                items_sold = int(m.group(1)); break
            if i + 1 < len(lines):
                m2 = re.search(r"\b(\d{1,3})\b", lines[i+1])
                if m2:
                    items_sold = int(m2.group(1)); break
    if items_sold is None:
        for ln in lines:
            m = re.search(r"\bITEMS\s+(\d{1,3})\b", ln, flags=re.I)
            if m:
                items_sold = int(m.group(1)); break

    return {
        "merchant_name": merchant,
        "tc_code": tc_code,
        "txn_datetime_raw": txn_date,
        "phone": phone,
        "items_sold_meta": items_sold,
    }


def _extract_totals(text: str) -> Dict[str, Optional[float]]:
    """
    Extract subtotal, tax, total, tendered, change due, and discount.
    Tolerant to OCR glitches (e.g., '38 .68', CASH without 'TEND', and 'NON TAXABLE').
    """
    subtotal = tax = total = None
    tend = change_due = None
    tax_pct = None
    discount = None

    # Normalize lines; collapse space before decimal: "38 .68" -> "38.68"
    raw_lines = [ln for ln in text.splitlines() if ln.strip()]
    lines = []
    for ln in raw_lines:
        ln = re.sub(r"\s{2,}", " ", ln.strip())
        ln = re.sub(r"(\d)\s+([.,]\d{2})", r"\1\2", ln)
        lines.append(ln)

    for ln in lines:
        low = ln.lower()

        if discount is None and re.search(r"\bdiscount\s+given\b", low):
            m = re.search(_NUMBER_ANY, ln);  discount = _parse_number(m.group(0)) if m else discount

        if subtotal is None and (re.search(r"\bsub\s*tot[ai1l]\b", low) or "subtotal" in low):
            m = re.search(_NUMBER_ANY, ln);  subtotal = _parse_number(m.group(0)) if m else subtotal

        if total is None and (re.search(r"^\s*tot[ai1l]\b", low) or " total" in f" {low}"):
            m = re.search(_NUMBER_ANY, ln);  total = _parse_number(m.group(0)) if m else total

        if tend is None and re.search(r"\b(cash|debit|credit|card|visa|mastercard)\b", low):
            m = re.search(_NUMBER_ANY, ln);  tend = _parse_number(m.group(0)) if m else tend

        if change_due is None and re.search(r"change\s*due|\bchange\b", low):
            m = re.search(_NUMBER_ANY, ln);  change_due = _parse_number(m.group(0)) if m else change_due

        if tax is None and "tax" in low and not re.search(r"non\s*tax|taxable", low):
            m_val = re.search(_NUMBER_ANY, ln)
            if m_val: tax = _parse_number(m_val.group(0))
            m_pct = re.search(r"(\d{1,2}[.,]?\d{0,2})\s*%+", ln)
            if m_pct:
                val = _parse_number(m_pct.group(1))
                if val is not None:
                    tax_pct = val / 100.0

    if total is None and subtotal is not None and tax is not None:
        total = round(subtotal + tax, 2)
    if tax is None and subtotal is not None and total is not None:
        tax = round(total - subtotal, 2)
    if total is None and tend is not None and change_due is not None:
        total = round(tend - change_due, 2)

    return {
        "subtotal_meta": subtotal,
        "tax_meta_val": tax,
        "tax_meta_pct": tax_pct,
        "total_meta": total,
        "tendered_meta": tend,
        "change_due_meta": change_due,
        "discount_meta_val": discount,
    }


# ---------------------------
# Item extraction
# ---------------------------

_PRICE_DECIMAL = r"(?:\d+[.,]\d{2})"

_ITEM_END_PRICE = re.compile(
    rf"^(?P<name>.+?)\s+(?P<price>{_PRICE_DECIMAL})(?:\s*[A-Za-z])?$",
    re.X
)

_EXCLUDE_ITEM_WORDS = re.compile(
    r"(SUBTOTAL|TOTAL|TAX|DISCOUNT|GIVEN|CASH|DEBIT|CARD|CHANGE|#\s*ITEMS\s*SOLD|"
    r"MANAGER|MGR|SURVEY|THANK|APPROVAL|ACCOUNT|TERMINAL|APPR|TC#|OP#|TE#|TR#|ST#)",
    re.I
)

_DECIMAL_LIST = re.compile(r"\d+[.,]\d{2}")

def _is_valid_name(s: str) -> bool:
    """Require at least one alpha token with length >= 3."""
    tokens = re.findall(r"[A-Za-z]{3,}", s)
    return len(tokens) > 0


def _extract_items(text: str) -> List[Dict[str, Optional[str]]]:
    """
    Extract items with tolerant rules and short look-back for two-line patterns.
    """
    items: List[Dict[str, Optional[str]]] = []
    raw_lines = [ln for ln in text.splitlines()]
    lines = [re.sub(r"\s{2,}", " ", ln.strip()) for ln in raw_lines]

    def is_excluded(s: str) -> bool:
        return bool(_EXCLUDE_ITEM_WORDS.search(s))

    def name_like(s: str) -> Optional[str]:
        alpha = re.sub(r"[^A-Za-z\s\-'/&]", " ", s).strip()
        alpha = re.sub(r"\s{2,}", " ", alpha).strip(" .-:|,")
        return alpha if len(alpha) >= 3 else None

    def find_name_lookback(idx: int) -> Optional[str]:
        for j in range(1, 3):
            k = idx - j
            if k < 0:
                break
            ln = lines[k].strip()
            if not ln or is_excluded(ln) or "@" in ln:
                continue
            nm = name_like(ln)
            if nm:
                return nm
        return None

    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1; continue
        if is_excluded(ln):
            i += 1; continue

        # Case A: current line has '@' (qty @ unit [ total ])
        if "@" in ln:
            decs = _DECIMAL_LIST.findall(ln)
            if len(decs) >= 2:
                qty = _parse_number(decs[0])
                unit = _parse_number(decs[1])
                total = _parse_number(decs[-1]) if len(decs) >= 3 else None
                nm = name_like(ln) or find_name_lookback(i)
                if nm and _is_valid_name(nm) and qty is not None and unit is not None:
                    if total is None:
                        total = qty * unit
                    items.append({
                        "item": nm,
                        "description": "",
                        "quantity": qty,
                        "unit_price": unit,
                        "line_total": round(total, 2)
                    })
                    i += 1
                    continue

        # Case B: two-line pattern (name-ish + next line with '@')
        if i + 1 < len(lines) and "@" in lines[i + 1]:
            decs_next = _DECIMAL_LIST.findall(lines[i + 1])
            nm = name_like(ln)
            if nm and _is_valid_name(nm) and len(decs_next) >= 2:
                qty = _parse_number(decs_next[0])
                unit = _parse_number(decs_next[1])
                total = _parse_number(decs_next[-1]) if len(decs_next) >= 3 else None
                if qty is not None and unit is not None:
                    if total is None:
                        total = qty * unit
                    items.append({
                        "item": nm,
                        "description": "",
                        "quantity": qty,
                        "unit_price": unit,
                        "line_total": round(total, 2)
                    })
                    i += 2
                    continue

        # Case C: ends with decimal price (qty=1)
        m_end = _ITEM_END_PRICE.search(ln)
        if m_end:
            raw_name = m_end.group("name")
            nm = re.sub(r"[^A-Za-z\s\-'/&]", " ", raw_name)
            nm = re.sub(r"\s{2,}", " ", nm).strip(" .-:|,")
            price = _parse_number(m_end.group("price"))
            if nm and _is_valid_name(nm) and price is not None:
                items.append({
                    "item": nm,
                    "description": "",
                    "quantity": 1.0,
                    "unit_price": price,
                    "line_total": price
                })
                i += 1
                continue

        i += 1

    # Final safety
    items = [it for it in items if not re.search(r"SUBTOTAL|TOTAL|TAX", it["item"], re.I)]
    return items


# ---------------------------
# Persistence (Bronze)
# ---------------------------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(items: List[Dict[str, Optional[str]]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "description", "quantity", "unit_price", "line_total"])
        writer.writeheader()
        for it in items:
            writer.writerow(it)


def _write_meta(meta: Dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------
# Public API
# ---------------------------

def extract_receipt_image(
    img_path: str,
    out_root: str = "data/output/bronze/receipts"
) -> Dict[str, str]:
    """
    Extract a single JPG receipt into Bronze artifacts.
    """
    img_path = str(img_path)
    stem = Path(img_path).stem
    out_dir = Path(out_root) / stem

    # --- Preprocess with OpenCV (writes _debug frames) ---
    debug_dir = out_dir / "_debug"
    cfg = PreprocessConfig(min_width=1800, do_autocrop=True, deskew_max_angle=8.0, debug_dir=debug_dir)
    final_img = preprocess_receipt(img_path, cfg=cfg)  # PIL.Image

    # Try to load multiple input variants from _debug if available
    gray_img = None
    bin_img = None
    try:
        gray_path = debug_dir / "01_gray.png"
        if gray_path.exists():
            gray_img = Image.open(gray_path)
    except Exception:
        gray_img = None
    try:
        bin_path = debug_dir / "05_bin_after.png"
        if bin_path.exists():
            bin_img = Image.open(bin_path)
    except Exception:
        bin_img = None

    # --- OCR across variants and pick best ---
    text, cfg_label, input_variant = _run_ocr_multi_inputs(gray_img, bin_img, final_img)

    # Parse
    header = _extract_header(text)
    totals = _extract_totals(text)
    items = _extract_items(text)

    # ---- Derive totals using items + discount when missing ----
    sum_items = round(sum((it.get("line_total") or 0) for it in items), 2)
    discount_val = totals.get("discount_meta_val")

    if totals.get("subtotal_meta") is None and discount_val is not None:
        totals["subtotal_meta"] = round(max(sum_items - discount_val, 0.0), 2)

    if totals.get("total_meta") is None:
        if totals.get("subtotal_meta") is not None and totals.get("tax_meta_val") is not None:
            totals["total_meta"] = round(totals["subtotal_meta"] + totals["tax_meta_val"], 2)
        elif totals.get("tendered_meta") is not None and totals.get("change_due_meta") is not None:
            totals["total_meta"] = round(totals["tendered_meta"] - (totals["change_due_meta"] or 0.0), 2)
        elif discount_val is not None:
            totals["total_meta"] = round(max(sum_items - discount_val, 0.0), 2)

    # Build meta
    meta = {
        "source_path": img_path.replace("\\", "/"),
        "file_hash": _sha256_file(img_path),
        "ingest_ts": datetime.utcnow().isoformat(),
        "extraction_flavor": f"ocr_mp:{cfg_label}",
        "ocr_input_variant": input_variant,  # gray / bin / final
        "merchant_name": header.get("merchant_name"),
        "tc_code": header.get("tc_code"),
        "txn_datetime_raw": header.get("txn_datetime_raw"),
        "phone": header.get("phone"),
        "items_sold_meta": header.get("items_sold_meta"),
        "items_detected": len(items),
        "discount_meta_val": totals.get("discount_meta_val"),
        "subtotal_meta": totals.get("subtotal_meta"),
        "tax_meta_val": totals.get("tax_meta_val"),
        "tax_meta_pct": totals.get("tax_meta_pct"),
        "total_meta": totals.get("total_meta"),
        "tendered_meta": totals.get("tendered_meta"),
        "change_due_meta": totals.get("change_due_meta"),
        "ocr_text_sample": "\n".join(text.splitlines()[:120]),
    }

    # Persist
    csv_path = out_dir / "receipt_line_items.csv"
    json_path = out_dir / "receipt_meta.json"
    _write_csv(items, csv_path)
    _write_meta(meta, json_path)

    return {
        "csv_path": str(csv_path),
        "meta_path": str(json_path),
        "out_dir": str(out_dir),
    }


def extract_receipts_dir(
    in_dir: str = "data/input/receipts",
    out_root: str = "data/output/bronze/receipts"
) -> List[Dict[str, str]]:
    """
    Batch mode: iterate over all .jpg/.jpeg files in in_dir and extract Bronze.
    """
    in_path = Path(in_dir)
    results: List[Dict[str, str]] = []
    candidates = list(in_path.glob("*.jpg")) + list(in_path.glob("*.jpeg"))
    for img_file in sorted(candidates):
        try:
            res = extract_receipt_image(str(img_file), out_root=out_root)
            results.append(res)
        except Exception as e:
            results.append({"error": str(e), "img_path": str(img_file)})
    return results


if __name__ == "__main__":
    out = extract_receipts_dir()
    print(json.dumps(out, indent=2))
