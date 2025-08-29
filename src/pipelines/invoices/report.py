# src/pipelines/invoices/report.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import psycopg


@dataclass
class ReportQueryResult:
    rows: List[Dict[str, Any]]


def _fetch_dicts(cur) -> List[Dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _run_query(conn, sql: str, params: Optional[dict] = None) -> ReportQueryResult:
    with conn.cursor() as cur:
        cur.execute(sql, params or {})
        return ReportQueryResult(rows=_fetch_dicts(cur))


def _mk_table_md(
    rows: List[Dict[str, Any]],
    headers: List[str],
    col_align: Optional[List[str]] = None
) -> str:
    """Render a minimal Markdown table from dict rows."""
    if not rows:
        return "_(no data)_\n"
    col_align = col_align or ["left"] * len(headers)
    sep_map = {"left": ":---", "right": "---:", "center": ":---:"}
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(sep_map.get(a, ":---") for a in col_align) + " |"
    body_lines = []
    for r in rows:
        body_lines.append("| " + " | ".join("" if r.get(h) is None else str(r.get(h)) for h in headers) + " |")
    return "\n".join([header_line, sep_line] + body_lines) + "\n"


def _fmt_money(x: Any) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)


def _fmt_pct(x: Any) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return str(x)


def make_report(
    dsn: str,
    out_path: Path | str = Path("reports/invoices/invoice_report.md"),
    threshold: float = 0.01,
    top_n: int = 5,
    summary_csv_path: Optional[Path | str] = None,
) -> Path:
    """
    Query Postgres and write a Markdown report using etl.documents / etl.line_items.
    Sections:
      - Overview
      - Top N parties (by value)
      - Top N suppliers (by value)
      - Top N items (by value and by frequency)
      - Documents with significant subtotal diffs
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(dsn) as conn:
        # Overview
        q_overview = """
        WITH li AS (
          SELECT d.document_id, d.doc_type, d.sum_line_total
          FROM etl.documents d
        )
        SELECT
            COUNT(*)                                     AS total_documents,
            COUNT(*) FILTER (WHERE doc_type = 'invoice') AS total_invoices,
            COALESCE(SUM(CASE WHEN doc_type='invoice' THEN sum_line_total END),0) AS invoices_total_value
        FROM li;
        """
        overview_rows = _run_query(conn, q_overview).rows
        overview = overview_rows[0] if overview_rows else {}

        # Top parties (bill-to) by value
        q_top_parties = """
        SELECT
            d.party_name,
            COALESCE(SUM(li.line_total),0) AS total_value
        FROM etl.documents d
        JOIN etl.line_items li USING (document_id)
        WHERE d.doc_type = 'invoice'
        GROUP BY d.party_name
        ORDER BY total_value DESC NULLS LAST
        LIMIT %(n)s;
        """
        top_parties = _run_query(conn, q_top_parties, {"n": top_n}).rows
        for r in top_parties:
            r["total_value"] = _fmt_money(r["total_value"])

        # Top suppliers (bill-from) by value
        q_top_suppliers = """
        SELECT
            COALESCE(d.supplier_name, 'Unknown') AS supplier_name,
            COUNT(DISTINCT d.document_id)        AS invoices,
            COALESCE(SUM(li.line_total),0)       AS total_value
        FROM etl.documents d
        JOIN etl.line_items li USING (document_id)
        WHERE d.doc_type = 'invoice'
        GROUP BY COALESCE(d.supplier_name, 'Unknown')
        ORDER BY total_value DESC NULLS LAST
        LIMIT %(n)s;
        """
        top_suppliers = _run_query(conn, q_top_suppliers, {"n": top_n}).rows
        for r in top_suppliers:
            r["total_value"] = _fmt_money(r["total_value"])

        # Top items by value
        q_top_items_value = """
        SELECT
            li.item,
            COUNT(*)                       AS lines,
            COALESCE(SUM(li.line_total),0) AS total_value
        FROM etl.line_items li
        JOIN etl.documents d USING (document_id)
        WHERE d.doc_type = 'invoice'
        GROUP BY li.item
        ORDER BY total_value DESC NULLS LAST
        LIMIT %(n)s;
        """
        top_items_val = _run_query(conn, q_top_items_value, {"n": top_n}).rows
        for r in top_items_val:
            r["total_value"] = _fmt_money(r["total_value"])

        # --- Top items by frequency (sum of quantities) ---
        q_top_items_freq = """
        SELECT
            li.item,
            SUM(COALESCE(li.quantity, 1)) AS quantity
        FROM etl.line_items li
        JOIN etl.documents d USING (document_id)
        WHERE d.doc_type = 'invoice'
        GROUP BY li.item
        ORDER BY quantity DESC NULLS LAST
        LIMIT %(n)s;
        """
        top_items_freq = _run_query(conn, q_top_items_freq, {"n": top_n}).rows

        # Discrepancies: focus on subtotal only
        q_discrepancies = """
        SELECT
            document_id,
            party_name,
            sum_line_total,
            subtotal_meta,
            tax_meta_pct,
            diff_sum_vs_subtotal,
            total_meta
        FROM etl.documents
        WHERE doc_type = 'invoice'
          AND ABS(COALESCE(diff_sum_vs_subtotal, 0)) > %(thr)s
        ORDER BY ABS(COALESCE(diff_sum_vs_subtotal, 0)) DESC;
        """
        discrepancies = _run_query(conn, q_discrepancies, {"thr": threshold}).rows
        for r in discrepancies:
            for k in ["sum_line_total", "subtotal_meta", "diff_sum_vs_subtotal", "total_meta"]:
                r[k] = _fmt_money(r[k])
            r["tax_meta_pct"] = _fmt_pct(r.get("tax_meta_pct"))

    # Optional: small CSV summary
    if summary_csv_path:
        import csv
        summary_csv_path = Path(summary_csv_path)
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["generated_at", "total_documents", "total_invoices", "invoices_total_value"])
            w.writerow([
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                overview.get("total_documents", 0),
                overview.get("total_invoices", 0),
                overview.get("invoices_total_value", 0),
            ])

    # Build Markdown
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    lines: List[str] = []
    lines.append("# Invoice Report")
    lines.append("")
    lines.append(f"_Generated at: {now}_")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    ov_rows = [{
        "total_documents": overview.get("total_documents", 0),
        "total_invoices": overview.get("total_invoices", 0),
        "invoices_total_value": _fmt_money(overview.get("invoices_total_value", 0)),
    }]
    lines.append(_mk_table_md(
        ov_rows,
        headers=["total_documents", "total_invoices", "invoices_total_value"],
        col_align=["right", "right", "right"]
    ))

    # Top parties
    lines.append("")
    lines.append(f"## Top {top_n} Parties by Value")
    lines.append("")
    lines.append(_mk_table_md(
        top_parties,
        headers=["party_name", "total_value"],
        col_align=["left", "right"]
    ))

    # Top suppliers
    lines.append("")
    lines.append(f"## Top {top_n} Suppliers by Value")
    lines.append("")
    lines.append(_mk_table_md(
        top_suppliers,
        headers=["supplier_name", "invoices", "total_value"],
        col_align=["left", "right", "right"]
    ))

    # Top items (value)
    lines.append("")
    lines.append(f"## Top {top_n} Items by Value")
    lines.append("")
    lines.append(_mk_table_md(
        top_items_val,
        headers=["item", "lines", "total_value"],
        col_align=["left", "right", "right"]
    ))

    # Top items (frequency)
    lines.append("")
    lines.append(f"## Top {top_n} Items by Frequency")
    lines.append("")
    lines.append(_mk_table_md(
        top_items_freq,
        headers=["item", "quantity"],
        col_align=["left", "right"]
    ))

    # Discrepancies
    lines.append("")
    lines.append(f"## Documents with Significant Subtotal Diffs (threshold = {threshold})")
    lines.append("")
    if discrepancies:
        lines.append(_mk_table_md(
            discrepancies,
            headers=[
                "document_id", "party_name", "sum_line_total",
                "subtotal_meta", "tax_meta_pct", "diff_sum_vs_subtotal", "total_meta"
            ],
            col_align=["left", "left", "right", "right", "right", "right", "right"]
        ))
    else:
        lines.append("_No discrepancies above threshold._\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
