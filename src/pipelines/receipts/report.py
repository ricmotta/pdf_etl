# src/pipelines/receipts/report.py
# -*- coding: utf-8 -*-
"""
Receipt report (merchant-focused).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List
import psycopg


def _q(conn, sql: str, args: Optional[tuple] = None) -> Tuple[list, list]:
    with conn.cursor() as cur:
        cur.execute(sql, args or ())
        cols = [c.name for c in cur.description]
        rows = cur.fetchall()
        return cols, rows


def make_receipt_report(dsn: str, out_path: Path, top_n: int = 5, threshold: float = 0.03) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(dsn) as conn:
        # Overview
        _, rows = _q(conn, """
            SELECT 
                COUNT(*) AS docs,
                COALESCE(SUM(total_meta),0)::numeric(18,2) AS total_spent,
                AVG(total_meta)::numeric(18,2) AS avg_ticket
            FROM etl.documents
            WHERE doc_type = 'receipt'
        """)
        docs, total_spent, avg_ticket = rows[0] if rows else (0, 0, 0)

        # Top merchants by value
        _, top_merchants = _q(conn, f"""
            SELECT COALESCE(supplier_name,'Unknown') AS merchant,
                   SUM(total_meta)::numeric(18,2)    AS total
            FROM etl.documents
            WHERE doc_type = 'receipt'
            GROUP BY COALESCE(supplier_name,'Unknown')
            ORDER BY total DESC NULLS LAST
            LIMIT {top_n}
        """)

        # Top items by value
        _, top_items_value = _q(conn, f"""
            SELECT li.description, SUM(li.line_total)::numeric(18,2) AS total
            FROM etl.line_items li
            JOIN etl.documents d USING (document_id)
            WHERE d.doc_type = 'receipt'
            GROUP BY li.description
            ORDER BY total DESC NULLS LAST
            LIMIT {top_n}
        """)

        # Top items by frequency (sum of quantities)
        _, top_items_freq = _q(conn, f"""
            SELECT li.description,
                SUM(COALESCE(li.quantity, 1)) AS qty
            FROM etl.line_items li
            JOIN etl.documents d USING (document_id)
            WHERE d.doc_type = 'receipt'
            GROUP BY li.description
            ORDER BY qty DESC NULLS LAST
            LIMIT {top_n}
        """)

        # Discrepancies (focus on subtotal only)
        _, discrepancies = _q(conn, """
            SELECT document_id,
                   COALESCE(supplier_name,'Unknown') AS merchant,
                   sum_line_total,
                   subtotal_meta,
                   total_meta,
                   diff_sum_vs_subtotal
            FROM etl.documents
            WHERE doc_type = 'receipt'
              AND ABS(COALESCE(diff_sum_vs_subtotal,0)) >= %s
            ORDER BY ABS(COALESCE(diff_sum_vs_subtotal,0)) DESC
            LIMIT 50
        """, (threshold,))

    # Markdown helpers
    def _tab(headers: List[str], rows: List[tuple]) -> str:
        out = []
        out.append("| " + " | ".join(headers) + " |")
        out.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            out.append("| " + " | ".join("" if x is None else str(x) for x in r) + " |")
        return "\n".join(out)

    # Write markdown
    md = []
    md.append("# Receipt Report\n")
    md.append("## Overview\n")
    md.append(f"- Documents: **{docs}**\n")
    md.append(f"- Total spent: **{total_spent}**\n")
    md.append(f"- Average ticket: **{avg_ticket}**\n")

    md.append("\n## Top Merchants by Value\n")
    md.append(_tab(["merchant", "total"], top_merchants) or "_no data_")

    md.append("\n\n## Top Items by Value\n")
    md.append(_tab(["item", "total"], top_items_value) or "_no data_")

    md.append("\n\n## Top Items by Frequency (by quantity)\n")
    md.append(_tab(["item", "quantity"], top_items_freq) or "_no data_")

    md.append("\n\n## Documents with Significant Subtotal Diffs\n")
    md.append(_tab(
        ["document_id", "merchant", "sum_line_total", "subtotal_meta", "total_meta", "diff_vs_subtotal"],
        discrepancies
    ) or "_no data_")

    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path
