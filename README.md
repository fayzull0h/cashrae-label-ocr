# Appliance Energy Information Retrieval Tool

A university capstone project that automates part of the ASHRAE building energy audit
workflow. The goal is to photograph an appliance rating plate, extract its brand and
model number, and retrieve authoritative technical data — primarily **Annual Energy
Consumption (AEC) in kWh/year** — from Natural Resources Canada (NRCan) datasets.

Rating plates rarely list annual energy consumption directly, so the value is looked up
in a structured database using the extracted manufacturer and model number.

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| **1A** | NRCan dataset acquisition, import pipeline, SQLite database | ✅ Complete |
| **1B** | Retrieval engine (normalization, exact + fuzzy matching) | ✅ Complete, tested |
| **2** | Label recognition / OCR / VLM extraction | 🟨 In progress — OCR foundation |
| **3** | Integration of image extraction with retrieval | ⬜ Not started |
| **4** | Expanded data sources (manuals, other datasets, APIs) | ⬜ Not started |
| **5** | User interface, reporting, final integration | ⬜ Not started |

**Current position:** Phase 2, Stage 1 of 6 complete. The Phase 1 backend is functional and
tested, and raw text can now be extracted from real label photos. Field parsing
(brand/model/serial) is the next stage.

---

## What Works Today

- **Repeatable import pipeline** — ingests all 10 NRCan CSV datasets into SQLite, with
  encoding fallback (UTF-8 → CP1252 → Latin-1), nested-directory support, category
  detection, and preserved original fields.
- **10 appliance categories** — clothes dryers, clothes washer-dryers, clothes washers,
  cooktops, dishwashers, freezers, ovens, ranges, refrigerators/wine chillers, and room
  air conditioners.
- **Normalization** — case, spaces, hyphens, wildcards, and punctuation are stripped so
  `PCKS-433-ET`, `pcks 433 et`, and `PCKS433ET#**` all match the same record.
- **Two-stage matching** — exact normalized match first, then fuzzy matching
  (`difflib.SequenceMatcher`, threshold `0.70`).
- **Ambiguity handling** — returns `ambiguous_match` with candidate lists instead of
  guessing when scores are too close (`0.04` margin).
- **Heterogeneous energy columns** — resolves the correct AEC column per category.
- **Honest energy reporting** — never fabricates values; room air conditioners report
  `EE_RATIO` (EER) instead of kWh/year and return `None` with an explanatory note.
- **Source provenance** — every result includes the database, table, source file, and
  `REC_REF_KEY`.
- **CLI demo + interactive mode** and a **12-test suite** covering the Phase 1 requirements.
- **OCR foundation (Phase 2)** — Tesseract + Pillow pipeline with size-aware preprocessing,
  a pluggable backend interface, single-image and batch CLI modes, and OCR confidence
  reporting. Verified on a real room-AC rating plate.

---

## Repository Structure

```
appliance-energy-tool/
├── .venv/                         # Main Python environment (Python 3.13)
├── .venv-paddle/                  # Isolated experimental environment (gitignored)
├── .gitignore                     # Venvs, bytecode, generated outputs
├── data/
│   ├── raw/                       # NRCan source data (001_ to 010_)
│   │   ├── 001_.../               # .zip archives + nested extracted folders
│   │   └── ...                    # Each holds *_Data_*.csv, *_Dictionary_*.csv, *_Information.csv
│   ├── samples/                   # Real appliance label photos + ground_truth.json
│   └── processed/                 # Debug images and generated comparison outputs
├── database/
│   └── appliances.db              # SQLite: 10 appliance tables + 2 metadata tables
├── docs/
│   └── paddleocr-experiment.md    # PaddleOCR comparison write-up (experimental branch)
├── scripts/
│   ├── import_datasets.py         # CSV ingestion pipeline into SQLite
│   ├── verify_import.py           # Database inspection / sanity check
│   ├── appliance_lookup.py        # Retrieval engine (normalization, exact/fuzzy search)
│   ├── label_ocr.py               # Phase 2: preprocessing + Tesseract OCR backend
│   ├── label_ocr_paddle.py        # Experimental PaddleOCR backend (same interface)
│   ├── compare_ocr.py             # Tesseract vs PaddleOCR comparison harness
│   └── test_lookup.py             # CLI demo, interactive mode, ad-hoc lookups
├── tests/
│   ├── test_lookup.py             # 12 integration tests for the retrieval engine
│   ├── test_label_ocr.py          # 23 tests for the OCR foundation
│   ├── test_label_ocr_paddle.py   # PaddleOCR backend tests (integration skipped without Paddle)
│   └── test_compare_ocr.py        # Comparison metrics and reporting tests
├── requirements.txt               # pandas, Pillow, pytesseract
├── requirements-paddle.txt        # Experimental PaddleOCR dependencies
├── AGENTS.md                      # Agent context and conventions
└── handoff-prompt.md              # Project background and requirements
```

---

## Database

`database/appliances.db` contains **6,697 appliance records** across 12 tables.

### Appliance tables (10)

| Table | Records |
|---|---:|
| `clothes_dryers` | 298 |
| `clothes_washer_dryers` | 43 |
| `clothes_washers` | 414 |
| `cooktops` | 144 |
| `dishwashers` | 600 |
| `freezers` | 425 |
| `ovens` | 276 |
| `ranges` | 366 |
| `refrigerators` | 3,021 |
| `room_air_conditioners` | 1,110 |

### Metadata tables (2)

- `datasets` — 10 rows: source filenames, ingestion dates, English/French product names.
- `field_dictionary` — 163 rows: original NRCan field names, data types, and labels.

### Energy column per category

NRCan uses different column names across datasets:

| Category | Column |
|---|---|
| `clothes_dryers`, `clothes_washer_dryers`, `freezers`, `refrigerators` | `AEC` |
| `clothes_washers` | `MOD_J1_AEC` |
| `cooktops`, `ovens`, `ranges` | `AEC_03` |
| `dishwashers` | `TAEC_C373_04` |
| `room_air_conditioners` | none — reports `EE_RATIO` (EER) |

---

## Requirements

- Python 3.13+
- `pandas` — database import and inspection
- `Pillow` + `pytesseract` — image preprocessing and OCR
- **Tesseract OCR 5.x** binary (external, not a Python package)
- SQLite 3 (bundled with Python)

The project virtual environment is at `.venv/` and targets Windows.

### Install dependencies

```bash
# Python packages
.venv/Scripts/python.exe -m pip install -r requirements.txt

# Tesseract binary (Windows)
winget install --id UB-Mannheim.TesseractOCR -e
```

If Tesseract is not on `PATH` after installation, set `TESSERACT_CMD` to the full path of
`tesseract.exe` (for example `C:\Program Files\Tesseract-OCR\tesseract.exe`). The OCR module
also auto-detects the default install locations.

---

## Quick Start

Run all commands from the project root.

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
python scripts/verify_import.py
```

**Direct execution (Windows, or WSL/Linux via interop):**

```bash
.venv/Scripts/python.exe scripts/verify_import.py
```

### Rebuild the database from raw CSVs

```bash
.venv/Scripts/python.exe scripts/import_datasets.py
```

### Inspect the database

```bash
.venv/Scripts/python.exe scripts/verify_import.py
```

### Run the test suite

```bash
# Everything (Phase 1 + OCR foundation + experiment tests)
.venv/Scripts/python.exe -m unittest discover -s tests

# Just the retrieval engine
.venv/Scripts/python.exe -m unittest tests/test_lookup.py

# Just the OCR foundation
.venv/Scripts/python.exe -m unittest tests/test_label_ocr.py
```

Expected result: `Ran 56 tests ... OK` for the full suite. The PaddleOCR integration
test is skipped unless you run the suite with `.venv-paddle/Scripts/python.exe`.

### Extract text from a label photo (Phase 2)

```bash
# Single image
.venv/Scripts/python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png

# JSON output
.venv/Scripts/python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png --json

# Batch over a directory
.venv/Scripts/python.exe scripts/label_ocr.py --dir data/samples

# Save preprocessed images for inspection
.venv/Scripts/python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png --debug
```

### Experimental: compare Tesseract with PaddleOCR

This is an isolated experiment. It lives on the `experiment/paddleocr` branch, uses a
separate virtual environment, and **does not change the default OCR engine**.

```bash
# One-time setup (isolated from .venv)
.venv/Scripts/python.exe -m venv .venv-paddle
.venv-paddle/Scripts/python.exe -m pip install -r requirements-paddle.txt

# Run the PaddleOCR backend directly (tuned PP-OCRv5 mobile profile)
.venv-paddle/Scripts/python.exe scripts/label_ocr_paddle.py --image data/samples/airconditioner_label.png

# Accurate but slow profile (PP-OCRv6 medium, ~25x slower on CPU)
.venv-paddle/Scripts/python.exe scripts/label_ocr_paddle.py --image data/samples/airconditioner_label.png --accurate

# Compare both engines and write a Markdown report
.venv-paddle/Scripts/python.exe scripts/compare_ocr.py \
  --dir data/samples \
  --ground-truth data/samples/ground_truth.json
```

`paddlepaddle` is pinned to `3.2.2` in `requirements-paddle.txt`: version `3.3.1`
crashes on Windows CPU in the oneDNN/PIR executor.

Results are written to `data/processed/ocr_comparison.json` and
`data/processed/ocr_comparison.md`. See `docs/paddleocr-experiment.md` for methodology,
measured results, and the adoption decision.

### Try the CLI

```bash
# Automated demonstration (6 scenarios)
.venv/Scripts/python.exe scripts/test_lookup.py --demo

# Ad-hoc lookup
.venv/Scripts/python.exe scripts/test_lookup.py --manufacturer "GE" --model "PCKS-433-ET"

# Restrict to one category
.venv/Scripts/python.exe scripts/test_lookup.py --manufacturer "Smeg" --model "PLA8743XU" --category dishwashers

# Interactive mode (no arguments)
.venv/Scripts/python.exe scripts/test_lookup.py
```

---

## Python API

```python
from scripts.appliance_lookup import get_annual_energy_consumption

result = get_annual_energy_consumption(
    manufacturer="GE",
    model_number="PCKS433ET#**",
    category="clothes_dryers",  # optional; omit to search all categories
)

print(result["status"])                     # exact_match | fuzzy_match | ambiguous_match | not_found | invalid_input
print(result["annual_energy_consumption"])  # 430.0 (or None if unavailable)
print(result["unit"])                       # "kWh/year" (or None)
print(result["match_confidence"])           # 0.0 – 1.0
print(result["source"])                     # database, table, source_file, rec_ref_key
```

### Result statuses

| Status | Meaning |
|---|---|
| `exact_match` | Unique normalized model (and brand) match |
| `fuzzy_match` | Best candidate above threshold and clear of the ambiguity margin |
| `ambiguous_match` | Multiple exact matches, or fuzzy scores within `0.04`; candidates returned, AEC is `None` |
| `not_found` | No candidate above `MIN_FUZZY_CONFIDENCE` (0.70) |
| `invalid_input` | Empty model number or unrecognized category |

When a manufacturer is omitted, an exact model-only match still succeeds but includes a
`warning` noting that the match was made on model number alone.

---

## Known Limitations

- **Room air conditioners** have no kWh/year AEC in the NRCan data; only `EE_RATIO` is
  available, so `annual_energy_consumption` is `None` for that category.
- **Fuzzy matching is brute force** — a Python scan over every row in the target tables
  (up to 3,021 for refrigerators). Fine at current scale, not indexed.
- **OCR is raw text only** — `scripts/label_ocr.py` does not yet parse brand/model fields;
  that is the next Phase 2 stage. OCR output on dense rating plates is noisy (confidence
  ~0.4–0.8), which is why parsing and error-tolerant matching are separate stages.
- **Tesseract accuracy varies** — stylized or low-contrast plates may need better
  preprocessing, a different `--psm`, or a VLM backend. The `OCRBackend` interface keeps
  that swappable.
- **Records without an AEC value** return `None` with `energy_consumption_available: False`
  rather than an estimate.
- **No web interface** — interaction is CLI / Python only.

---

## Roadmap

1. **Phase 2 (in progress)** — Stage 1 OCR foundation is complete. Remaining stages:
   field parsing (brand/model/serial/electrical), OCR-error candidate generation,
   end-to-end extraction → retrieval bridge, CLI polish, and evaluation on real photos.
2. **Phase 3** — connect extraction to the retrieval engine; end-to-end photo → energy
   value workflow with confidence scoring and manual correction.
3. **Phase 4** — add manufacturer manuals, other government datasets, and external APIs
   with explicit source attribution and reliability indicators.
4. **Phase 5** — user interface, audit-ready exportable reports, final integration.

---

## Data Source

Appliance records come from Natural Resources Canada (NRCan) energy-efficiency datasets
for the 10 appliance categories above. Original values are preserved unmodified; the
import pipeline is the single source of truth for rebuilding the database.
CASHRAE label OCR tool
