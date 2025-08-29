# PDF ETL Pipeline (Invoices & Receipts)

## Project Overview
This project implements an **end-to-end ETL pipeline** for extracting structured information from unstructured PDF and JPG documents such as **invoices** and **receipts**.  

The pipeline:
- Extracts raw structured data from PDFs (**Bronze layer**).
- Cleans, normalizes, and reconciles values (**Silver layer**).
- Loads into PostgreSQL tables and analytical views (**Gold layer**).
- Generates **automated Markdown reports** with insights.  

---

## Contents

* [Project Overview](#project-overview)
* [Project Structure](#project-structure)
* [Medallion Layers](#medallion-layers)
* [Database Schema](#database-schema)
* [Key Decisions & Trade-offs](#key-decisions--trade-offs)
* [Running the Pipeline](#running-the-pipeline)
  * [Initialize directories](#initialize-directories)
  * [Ingest invoices](#ingest-invoices)
  * [Ingest receipts](#ingest-receipts)
  * [Generate reports](#generate-reports)
* [Requirements](#requirements)
* [System Dependencies](#system-dependencies)

---

## Project Structure
```

project-root/
├── src/
│   ├── cli.py               # Unified CLI (Typer)
│   ├── ingest.py            # Batch orchestration
│   ├── db/
│   │   └── load\_pg.py       # PostgreSQL loader (DDL + COPY + views)
│   └── pipelines/
│       ├── invoices/        # Invoice pipeline (extract, silver, report)
│       └── receipts/        # Receipt pipeline (extract, preprocess, report)
│
├── data/
│   ├── input/               # Raw PDFs/images
│   │   ├── invoices/
│   │   └── receipts/
│   └── output/
│       ├── bronze/          # Per-document raw extractions
│       └── silver/          # Consolidated CSV/Parquet outputs
│
├── db/
│   ├── ddl.sql              # Schema definition
│   └── gold.sql             # Analytical views & functions
│
├── reports/                 # Generated reports
│   ├── invoices/
│   └── receipts/
│
└── requirements.txt

````

---

## Medallion Layers

![ETL Architecture](documents/etl_architecture.png)

- **Bronze:**  
  First structured extraction from raw documents (PDF invoices, scanned JPG receipts).  
  Stores per-document outputs such as:  
  - `invoice_line_items.csv`, `invoice_meta.json`  
  - `receipt_line_items.csv`, `receipt_meta.json` (+ `_debug/` OCR snapshots)

- **Silver:**  
  Curated, normalized datasets with consistent schema.  
  - `invoice_items.csv`, `invoice_documents.csv`  
  - `receipts_items.csv`, `receipts_documents.csv`  

  These outputs are then **loaded into PostgreSQL tables**:  
  - `etl.documents` (one row per document, normalized metadata)  
  - `etl.line_items` (line-level details with FK to documents)

- **Gold:**  
  Analytical views and functions in PostgreSQL (`db/gold.sql`) for business insights:  
  - `v_summary_overview`, `v_top_parties`, `v_top_suppliers`, `v_top_items_by_value`, `v_consistency_diffs`, etc.  
  - `f_consistency_outliers(threshold)` for anomaly detection.  

- **Reports:**
  Final deliverables generated automatically from Silver tables:  
  - `reports/invoices/invoice_report.md`  
  - `reports/receipts/receipt_report.md` (optional)  
  - Additional CSV/JSON exports for dashboards or BI tools.  

---

## Database Schema

![ETL Diagram](documents/etl_diagram.png)

### `etl.documents`
| Column                 | Type          | Notes                                                 |
|-------------------------|--------------|-------------------------------------------------------|
| document_id (PK)        | TEXT         | `SUPPLIERABBR-number-yyyymmdd` or fallback hash       |
| doc_type                | TEXT         | `invoice` or `receipt`                                |
| document_number         | TEXT         | Extracted ID; fallback to filename or "Not Found"     |
| party_name              | TEXT         | Client (invoice) / End Customer (receipt)             |
| supplier_name           | TEXT         | Vendor/merchant issuing document                      |
| currency                | TEXT         | Normalized (USD, EUR, BRL, GBP)                       |
| items_count             | INT          | Number of items                                       |
| sum_line_total          | NUMERIC      | Computed sum of line totals                          |
| subtotal_meta           | NUMERIC      | Extracted subtotal                                    |
| tax_meta_pct            | NUMERIC      | Always stored as full percentage (e.g., 6 = 6%)       |
| total_meta              | NUMERIC      | Extracted or reconciled total                         |
| diff_sum_vs_subtotal    | NUMERIC      | Absolute diff (sum_line_total – subtotal_meta)        |
| diff_sum_vs_amount_due  | NUMERIC      | Absolute diff (sum_line_total – total_meta)           |
| source_path             | TEXT         | File path                                             |
| extraction_flavor       | TEXT         | Extraction method (OCR, table-based, etc.)            |
| updated_at              | TIMESTAMPTZ  | Actual DB load timestamp (governance)                 |

### `etl.line_items`
| Column       | Type            | Notes                             |
|--------------|-----------------|-----------------------------------|
| id (PK)      | BIGSERIAL       | Internal surrogate key            |
| document_id  | TEXT (FK)       | References `etl.documents`        |
| item         | TEXT            | Item code/name                    |
| description  | TEXT            | Item description                  |
| quantity     | NUMERIC(14,3)   | Normalized quantity               |
| unit_price   | NUMERIC(14,2)   | Normalized price                  |
| line_total   | NUMERIC(14,2)   | Line total (computed if missing)  |

---

## Key Decisions & Trade-offs
- **Unified schema:** One `documents` table covers both invoices and receipts → avoids sparsity.  
- **Fallbacks:**  
  - Missing totals are recomputed from line items.  
  - Document IDs are deterministic; fallbacks use filename or hash.  
- **Normalization:**  
  - `tax_meta_pct` always full percent.  
  - Diff columns always absolute numeric values.  
- **Bronze/Silver separation:**  
  - Debuggable per-document outputs before global aggregation.  
  - More intermediate files, but easier traceability.  
- **Postgres loading:**  
  - COPY FROM STDIN for speed.  
  - `--replace` option ensures idempotency.  
- **Reports:**  
  - Markdown always generated.  
  - Invoices: overview, top parties/suppliers, top items (value/frequency), discrepancies.  
  - Receipts: overview, top merchants, top items.  

---

## Running the Pipeline

### Initialize directories
```bash
python -m src.cli init-data-dirs
````

### Ingest invoices

```bash
python -m src.cli ingest-invoices --dsn "postgresql://usr:pass@localhost:5432/etl"
```

### Ingest receipts

```bash
python -m src.cli ingest-receipts --dsn "postgresql://usr:pass@localhost:5432/etl"
```

### Generate reports

```bash
python -m src.cli make-report --dsn "..."
python -m src.cli make-receipt-report --dsn "..."
```

**Results:**

* Database populated (`etl.documents`, `etl.line_items`).
* Reports in `reports/invoices/` and `reports/receipts/`.

---

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## System Dependencies

Besides the Python requirements, a few system-level tools are required for full reproducibility:

- **PostgreSQL**  
  - A local PostgreSQL instance is required to load the Silver data into the analytical layer.  
  - Can be installed locally or run via Docker.  
  - Example with Docker:  
    ```bash
    docker run --name etl-postgres -e POSTGRES_PASSWORD=etl -e POSTGRES_DB=etl -p 5432:5432 -d postgres:15
    ```
  - Database inspection and debugging can be done with **DBeaver** (or any SQL client).

- **Ghostscript**  
  - Required by `camelot-py` for invoice table extraction.  
  - Installation:  
    - Ubuntu/Debian: `sudo apt-get install ghostscript`  
    - macOS (brew): `brew install ghostscript`  
    - Windows: download from [https://www.ghostscript.com/](https://www.ghostscript.com/)

- **Tesseract OCR**  
  - Required for receipts OCR pipeline.  
  - Installation:  
    - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`  
    - macOS (brew): `brew install tesseract`  
    - Windows: binaries available at [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

---

## Next Steps

While the current implementation covers the full ETL, I wrote down possible directions worth exploring with more time:

### 1. Data Quality & Testing
- Add **Great Expectations** validations in the Silver layer (totals, numeric ranges, mandatory fields).
- Implement **unit tests** for parsers and ID builders, plus **end-to-end tests** with synthetic PDFs.
- Improve logging and error handling (structured logs per PDF, store failed artifacts for debugging).

### 2. Cloud Integration
- Fetch PDFs directly from **AWS S3** (or other storage) instead of local `data/input/`.
- Store Silver/Gold outputs in cloud warehouses for scalability.

### 3. Interfaces
- Extend the CLI with more granular commands (`reprocess-one`, `validate-only`).
- Build a lightweight **FastAPI service** with an endpoint `/ingest` to trigger ingestion via API.

### 4. Advanced Parsing
- Integrate **AI-assisted OCR/parsing** for receipts and contracts with irregular layouts.
- Add support for multi-page invoices and contracts with mixed text+tables.

### 5. Containerization & CI/CD
- Package the pipeline with **Docker** for reproducible environments.
- Add **GitHub Actions** workflows for linting, testing, and running ingestion automatically.

### 6. Analytics & Visualization
- Expand PostgreSQL Gold views (trends by month, tax distributions, outlier detection).
- Add dashboards to visualize top suppliers, spend by category, and reconciliation mismatches.
