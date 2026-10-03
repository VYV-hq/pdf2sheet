# pdf2sheet

Scan a Korean bank-statement PDF (KB or IBK) and merge every page into one Excel workbook. Large files are processed in batches of *n* pages, then combined.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Install Tesseract, then put Korean/English models in `tessdata/` (see `tessdata/README.md`).

## Run

Web UI (KB + IBK, auto-detect):

```bash
.venv/bin/uvicorn app.server:app --host 127.0.0.1 --port 8199
```

- http://127.0.0.1:8199/ — all banks
- http://127.0.0.1:8199/ibk — IBK-oriented page

CLI:

```bash
.venv/bin/python -m app.cli input.pdf -o output.xlsx --batch-size 20 --workers 6
```
