# src/cli.py
from __future__ import annotations

import json
from pathlib import Path
import typer

# --- Invoices imports ---
from src.ingest import ingest_dir_invoices
from src.pipelines.invoices.extract import extract_invoice
from src.pipelines.invoices.silver import bronze_to_silver as bronze_to_silver_invoices
from src.db.load_pg import load_postgres_and_gold
from src.pipelines.invoices.report import make_report as make_invoice_report

# --- Receipts imports ---
from src.pipelines.receipts.extract import (
    extract_receipt_image,
    extract_receipts_dir,
)
from src.pipelines.receipts.silver import aggregate_receipts_silver
from src.pipelines.receipts.report import make_receipt_report

app = typer.Typer(help="PDF ETL CLI")

# --- Default paths for invoices ---
IN_DIR_DEFAULT = Path("data/input/invoices")
BRONZE_ROOT_DEFAULT = Path("data/output/bronze/invoices")
SILVER_DIR_DEFAULT = Path("data/output/silver/invoices")
REPORT_PATH_DEFAULT = Path("reports/invoices/invoice_report.md")

# --- Default paths for receipts ---
IN_DIR_RECEIPTS_DEFAULT = Path("data/input/receipts")
BRONZE_ROOT_RECEIPTS_DEFAULT = Path("data/output/bronze/receipts")
SILVER_DIR_RECEIPTS_DEFAULT = Path("data/output/silver/receipts")
RECEIPT_REPORT_PATH_DEFAULT = Path("reports/receipts/receipt_report.md")

DDL_DEFAULT = Path("db/ddl.sql")
GOLD_DEFAULT = Path("db/gold.sql")


@app.command("init-data-dirs")
def init_data_dirs():
    """
    Create the recommended data/report directory tree for invoices and receipts.
    """
    dirs = [
        Path("data/input/invoices"),
        Path("data/input/receipts"),
        Path("data/output/bronze/invoices"),
        Path("data/output/bronze/receipts"),
        Path("data/output/silver/invoices"),
        Path("data/output/silver/receipts"),
        Path("reports/invoices"),
        Path("reports/receipts"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    typer.echo("[OK] Created data/report directory tree.")


# =========================
# Invoices commands
# =========================
@app.command("extract-invoice")
def extract_invoice_cmd(
    pdf: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to invoice PDF"),
    out_dir: Path = typer.Option(..., help="Bronze folder for this single PDF"),
):
    """
    Extract one invoice PDF to bronze (line items + meta).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = extract_invoice(pdf_path=pdf, out_dir=out_dir)
    typer.echo(json.dumps(result, indent=2))


@app.command("to-silver")
def to_silver_cmd(
    bronze_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Bronze folder of a single invoice"),
    out_dir: Path = typer.Option(..., help="Where to write the per-document silver (usually bronze/<stem>/_silver)"),
):
    """
    Promote one invoice bronze folder to silver (per-document CSVs).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = bronze_to_silver_invoices(bronze_dir=bronze_dir, out_dir=out_dir)
    typer.echo(json.dumps(result, indent=2))


@app.command("ingest-invoices")
def ingest_invoices_cmd(
    in_dir: Path = typer.Option(IN_DIR_DEFAULT, help="Folder with invoice PDFs"),
    bronze_root: Path = typer.Option(BRONZE_ROOT_DEFAULT, help="Where per-PDF bronze subfolders are created"),
    silver_dir: Path = typer.Option(SILVER_DIR_DEFAULT, help="Where aggregated silver CSVs are written"),
    dsn: str = typer.Option(..., help="Postgres DSN, e.g. postgresql://usr:pass@localhost:5432/etl"),
    ddl: Path = typer.Option(DDL_DEFAULT, help="Path to DDL SQL"),
    gold: Path = typer.Option(GOLD_DEFAULT, help="Path to GOLD SQL (views/functions)"),
    report_out: Path = typer.Option(REPORT_PATH_DEFAULT, help="Markdown report output"),
    threshold: float = typer.Option(0.01, help="Discrepancy threshold for highlighting"),
    top_n: int = typer.Option(5, help="Top-N for rankings"),
    replace: bool = typer.Option(True, help="Replace existing rows for same document_id"),
):
    """
    Batch pipeline for invoices: bronze -> silver -> Postgres -> report.
    """
    out = ingest_dir_invoices(
        in_dir=in_dir,
        bronze_root=bronze_root,
        silver_dir=silver_dir,
        dsn=dsn,
        ddl_path=ddl,
        gold_sql_path=gold,
        report_out=report_out,
        threshold=threshold,
        top_n=top_n,
        replace=replace,
    )
    typer.echo(json.dumps(out, indent=2))


@app.command("make-report")
def make_report_cmd(
    dsn: str = typer.Option(..., help="Postgres DSN"),
    out_path: Path = typer.Option(REPORT_PATH_DEFAULT, help="Markdown report output (invoices)"),
    threshold: float = typer.Option(0.01, help="Discrepancy threshold"),
    top_n: int = typer.Option(5, help="Top-N for rankings"),
):
    """
    Generate the Markdown report for invoices using Postgres data.
    """
    path = make_invoice_report(dsn=dsn, out_path=out_path, threshold=threshold, top_n=top_n)
    typer.echo(f"[OK] Invoice report written to {path}")


# =========================
# Receipts commands
# =========================
@app.command("extract-receipt")
def extract_receipt_cmd(
    img_or_pdf: Path = typer.Option(..., exists=True, help="Path to receipt image/PDF"),
    out_dir: Path = typer.Option(..., help="Bronze folder for this single receipt"),
):
    """
    Extract one receipt image/PDF to bronze (line items + meta).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    res = extract_receipt_image(str(img_or_pdf), out_root=str(out_dir.parent))
    typer.echo(json.dumps(res, indent=2))


@app.command("extract-receipts-dir")
def extract_receipts_dir_cmd(
    in_dir: Path = typer.Option(IN_DIR_RECEIPTS_DEFAULT, help="Folder with receipt images/PDFs"),
    bronze_root: Path = typer.Option(BRONZE_ROOT_RECEIPTS_DEFAULT, help="Where per-receipt bronze subfolders are created"),
):
    """
    Extract all receipts in a folder to bronze.
    """
    out = extract_receipts_dir(in_dir=str(in_dir), out_root=str(bronze_root))
    typer.echo(json.dumps(out, indent=2))


@app.command("to-silver-receipts")
def to_silver_receipts_cmd(
    bronze_root: Path = typer.Option(BRONZE_ROOT_RECEIPTS_DEFAULT, help="Root with bronze/receipts/<stem>"),
    out_dir: Path = typer.Option(SILVER_DIR_RECEIPTS_DEFAULT, help="Aggregated silver for receipts"),
):
    """
    Promote all bronze receipts to aggregated silver CSVs.
    """
    out = aggregate_receipts_silver(bronze_root=bronze_root, silver_dir=out_dir)
    typer.echo(json.dumps(out, indent=2))


@app.command("ingest-receipts")
def ingest_receipts_cmd(
    in_dir: Path = typer.Option(IN_DIR_RECEIPTS_DEFAULT, help="Folder with receipt images/PDFs"),
    bronze_root: Path = typer.Option(BRONZE_ROOT_RECEIPTS_DEFAULT, help="Where per-receipt bronze subfolders are created"),
    silver_dir: Path = typer.Option(SILVER_DIR_RECEIPTS_DEFAULT, help="Where aggregated silver CSVs are written"),
    dsn: str = typer.Option(..., help="Postgres DSN"),
    ddl: Path = typer.Option(DDL_DEFAULT, help="Path to DDL SQL"),
    gold: Path = typer.Option(GOLD_DEFAULT, help="Path to GOLD SQL"),
    report_out: Path = typer.Option(RECEIPT_REPORT_PATH_DEFAULT, help="Receipt Markdown report output"),
    threshold: float = typer.Option(0.03, help="Discrepancy threshold"),
    top_n: int = typer.Option(5, help="Top-N for rankings"),
    replace: bool = typer.Option(True, help="Replace existing rows for same document_id"),
):
    """
    Batch pipeline for receipts: bronze -> silver -> Postgres -> report.
    """
    from src.ingest import ingest_dir_receipts
    out = ingest_dir_receipts(
        in_dir=in_dir,
        bronze_root=bronze_root,
        silver_dir=silver_dir,
        dsn=dsn,
        ddl_path=ddl,
        gold_sql_path=gold,
        report_out=report_out,
        threshold=threshold,
        top_n=top_n,
        replace=replace,
    )
    typer.echo(json.dumps(out, indent=2))


@app.command("make-receipt-report")
def make_receipt_report_cmd(
    dsn: str = typer.Option(..., help="Postgres DSN"),
    out_path: Path = typer.Option(RECEIPT_REPORT_PATH_DEFAULT, help="Markdown report output (receipts)"),
    threshold: float = typer.Option(0.03, help="Discrepancy threshold"),
    top_n: int = typer.Option(5, help="Top-N for rankings"),
):
    """
    Generate the Markdown report for receipts using Postgres data.
    """
    path = make_receipt_report(dsn=dsn, out_path=out_path, threshold=threshold, top_n=top_n)
    typer.echo(f"[OK] Receipt report written to {path}")


# =========================
# Generic Postgres loader
# =========================
@app.command("load-postgres")
def load_postgres_cmd(
    silver_dir: Path = typer.Option(SILVER_DIR_DEFAULT, help="Aggregated silver folder (invoices or receipts)"),
    dsn: str = typer.Option(..., help="Postgres DSN"),
    ddl: Path = typer.Option(DDL_DEFAULT, help="DDL SQL"),
    gold: Path = typer.Option(GOLD_DEFAULT, help="GOLD SQL"),
    replace: bool = typer.Option(True, help="Replace existing rows for same document_id"),
):
    """
    Load silver CSVs into Postgres (also runs GOLD SQL).
    Works for either invoices or receipts: pass the corresponding --silver-dir.
    """
    out = load_postgres_and_gold(
        silver_dir=silver_dir, dsn=dsn, ddl_path=ddl, gold_sql_path=gold, replace=replace
    )
    typer.echo(json.dumps(out, indent=2))
    typer.echo("[OK] Gold views/functions are ready.")


if __name__ == "__main__":
    app()
