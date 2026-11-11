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
| **2** | Label recognition / OCR / VLM extraction | 🟨 In progress — OCR + field parsing |
| **3** | Integration of image extraction with retrieval | ⬜ Not started |
| **4** | Expanded data sources (manuals, other datasets, APIs) | ⬜ Not started |
| **5** | User interface, reporting, final integration | ⬜ Not started |

**Current position:** Phase 2, Stage 2 of 6 complete. The Phase 1 backend is functional and
tested. Raw text extraction uses PaddleOCR by default (with automatic Tesseract fallback),
and a rule-based parser extracts brand, model number, and serial number from the OCR text.
OCR-error candidate generation and the extraction → retrieval bridge are next.

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
- **OCR foundation (Phase 2)** — PaddleOCR (tuned PP-OCRv5 mobile profile) is the default
  engine, with automatic fallback to Tesseract when Paddle is unavailable. Pillow
  preprocessing, a pluggable backend interface, single-image and batch CLI modes, and
  OCR confidence reporting. Evaluated on six verified rating plates: PaddleOCR reached
  100% field recall versus 33% for Tesseract at lower CPU latency.
- **Structured field parsing (Phase 2)** — `scripts/label_parse.py` turns raw OCR text into
  `brand`, `model_number`, and `serial_number` using label anchors, adjacent-line lookups,
  and heuristic token scoring. Per-field rule-based confidence, bilingual (EN/FR) anchors,
  and `normalize_alphanum()` output ready for the retrieval engine. Live evaluation on the
  six verified plates: **6/6 brand, 6/6 model, 6/6 serial** (serial prefixes not scored).

---

## Repository Structure

```
appliance-energy-tool/
├── .venv/                         # Main Python environment (Python 3.13, includes OCR extras)
├── .venv-paddle/                  # Optional isolated environment for OCR experiments
├── .gitignore                     # Venvs, bytecode, external photos, generated outputs
├── data/
│   ├── raw/                       # NRCan source data (001_ to 010_)
│   │   ├── 001_.../               # .zip archives + nested extracted folders
│   │   └── ...                    # Each holds *_Data_*.csv, *_Dictionary_*.csv, *_Information.csv
│   ├── samples/                   # Bundled photo + ground truth; external/ is gitignored
│   │   └── README.md              # Where to download the external photo bundle
│   └── processed/                 # Debug images, generated comparisons, tracked OCR fixture
├── database/
│   └── appliances.db              # SQLite: 10 appliance tables + 2 metadata tables
├── docs/
│   └── paddleocr-experiment.md    # OCR engine evaluation and adoption record
├── scripts/
│   ├── import_datasets.py         # CSV ingestion pipeline into SQLite
│   ├── verify_import.py           # Database inspection / sanity check
│   ├── appliance_lookup.py        # Retrieval engine (normalization, exact/fuzzy search)
│   ├── label_ocr.py               # Phase 2: preprocessing + OCR entry point (Paddle default)
│   ├── ocr_backends.py            # Engine selection and per-engine preprocessing defaults
│   ├── label_ocr_paddle.py        # PaddleOCR backend (fast profile, optional accurate mode)
│   ├── label_parse.py             # Phase 2: brand/model/serial parser (anchors + heuristics)
│   ├── evaluate_parse.py          # Per-field parsing accuracy over data/samples
│   ├── compare_ocr.py             # Tesseract vs PaddleOCR comparison harness
│   └── test_lookup.py             # CLI demo, interactive mode, ad-hoc lookups
├── tests/
│   ├── test_lookup.py             # 12 integration tests for the retrieval engine
│   ├── test_label_ocr.py          # OCR foundation tests
│   ├── test_label_ocr_paddle.py   # PaddleOCR backend tests (integration skipped without Paddle)
│   ├── test_label_parse.py        # Field parsing tests (synthetic + six verified plates)
│   ├── test_evaluate_parse.py     # Evaluation helper tests
│   ├── test_compare_ocr.py        # Comparison metrics and reporting tests
│   └── test_ocr_backends.py       # Engine selection and fallback tests
├── requirements.txt               # pandas, Pillow, pytesseract (Tesseract fallback)
├── requirements-paddle.txt        # PaddleOCR dependencies (default engine)
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
- `Pillow` + `pytesseract` — image preprocessing and Tesseract fallback
- `paddlepaddle` + `paddleocr` — default OCR engine (see the pin note below)
- **Tesseract OCR 5.x** binary (external, not a Python package; fallback engine)
- SQLite 3 (bundled with Python)

The project virtual environment is at `.venv/` and targets Windows.

### Install dependencies

```bash
# Python packages (base + default OCR engine)
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -r requirements-paddle.txt

# Tesseract binary (Windows) — fallback engine
winget install --id UB-Mannheim.TesseractOCR -e
```

`paddlepaddle` is pinned to `3.2.2`: version `3.3.1` crashes on Windows CPU in the
oneDNN/PIR executor. The Paddle install plus model cache is roughly 1.1 GB. If Paddle
is not installed, `label_ocr.py` automatically falls back to Tesseract and reports
`"backend": "tesseract"` in its output.

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
# Everything (Phase 1 + OCR + field parsing + experiment tests)
.venv/Scripts/python.exe -m unittest discover -s tests

# Just the retrieval engine
.venv/Scripts/python.exe -m unittest tests/test_lookup.py

# Just the OCR foundation
.venv/Scripts/python.exe -m unittest tests/test_label_ocr.py

# Just field parsing
.venv/Scripts/python.exe -m unittest tests/test_label_parse.py
```

Expected result: `Ran 105 tests ... OK` for the full suite. The PaddleOCR integration
test runs when PaddleOCR is installed (the default setup) and is skipped otherwise.

### Extract text from a label photo (Phase 2)

PaddleOCR is the default engine. `--engine tesseract` forces the fallback; `--engine auto`
(the default) uses Paddle when installed.

```bash
# Single image (PaddleOCR fast profile by default)
.venv/Scripts/python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png

# Force the Tesseract fallback
.venv/Scripts/python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png --engine tesseract

# JSON output
.venv/Scripts/python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png --json

# Batch over a directory (recurses into data/samples/external/)
.venv/Scripts/python.exe scripts/label_ocr.py --dir data/samples

# Save preprocessed images for inspection
.venv/Scripts/python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png --debug
```

Preprocessing defaults differ by engine: PaddleOCR gets grayscale + autocontrast with no
upscaling (very large photos are still clamped to 5000 px; Paddle caps detection at 960 px),
while Tesseract keeps the full size-aware resize pipeline.

### Parse structured fields from a label (Phase 2, Stage 2)

`scripts/label_parse.py` turns raw OCR text into `brand`, `model_number`, and
`serial_number`. It runs OCR first when given `--image`, or parses text directly.

```bash
# OCR + parse in one step
.venv/Scripts/python.exe scripts/label_parse.py --image data/samples/airconditioner_label.png

# Parse raw text
.venv/Scripts/python.exe scripts/label_parse.py --text "MODEL NO: ABC-123"

# JSON output
.venv/Scripts/python.exe scripts/label_parse.py --image data/samples/airconditioner_label.png --json

# Ask the OCR CLI to include parsed fields
.venv/Scripts/python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png --parse
```

Each field includes `value`, `normalized`, a rule-based `confidence` (0–1), the `method`
that produced it (`label_anchor`, `label_anchor_before`, `adjacent_line`, `known_brand`,
`heuristic_model`, …), and the `source_line`. The parser handles English and French anchors
(`MODEL`/`MODÈLE`, `SERIAL`/`SÉRIE`, `BRAND`/`MARQUE`), common abbreviations (`MOD.`,
`S/N`, `SER`), values before or after the anchor, and a known-brand fallback list.

### Measure parsing accuracy on the verified plates

```bash
# Default: PaddleOCR + parser over data/samples (including external/)
.venv/Scripts/python.exe scripts/evaluate_parse.py

# Force Tesseract and emit JSON
.venv/Scripts/python.exe scripts/evaluate_parse.py --engine tesseract --json
```

The harness prints per-image pass/fail, per-field accuracy, OCR confidence, and latency.
Serial numbers are compared prefix-tolerantly because the verified plates sometimes omit a
leading letter (for example `E140302785` vs `140302785`).

### OCR engine evaluation and comparison

The six-photo evaluation that selected PaddleOCR is reproducible with:

```bash
# Default: Tesseract vs the tuned PaddleOCR fast profile
.venv/Scripts/python.exe scripts/compare_ocr.py \
  --dir data/samples \
  --ground-truth data/samples/ground_truth.json

# Include the slow PP-OCRv6 medium model for reference (~25x slower on CPU)
.venv/Scripts/python.exe scripts/compare_ocr.py \
  --dir data/samples \
  --ground-truth data/samples/ground_truth.json \
  --backends tesseract,paddleocr,paddleocr-fast
```

Results are written to `data/processed/ocr_comparison.json` and
`data/processed/ocr_comparison.md` (both gitignored). See
`docs/paddleocr-experiment.md` for methodology, measured results, and the adoption
decision, and `data/samples/README.md` for the external photo bundle.

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
- **OCR output is noisy** — on dense rating plates confidence is ~0.4–0.9, and confidence
  does not reliably separate correct from incorrect text.
- **Field parsing is rule-based** — it handles the six verified plates and common bilingual
  labels, but it does not yet generate or rank OCR-error candidates (`0`/`O`, `1`/`I`,
  `8`/`B`), and unusual layouts may need new anchors. Parsed values are not yet
  cross-checked against the NRCan database.
- **Serial prefixes are not scored** — a leading letter (for example `E140302785`) may be
  present or absent in ground truth; the evaluator treats a suffix match as correct.
- **PaddleOCR is CPU-tuned** — the default profile uses up to 16 CPU threads and is
  benchmarked on a 22-thread desktop CPU. A low-core laptop will be slower (about 5.7 s
  per large plate at 4 threads); set `PADDLE_PDX_CPU_NUM_THREADS` to match the machine.
  Tesseract remains available via `--engine tesseract` for comparison or as an automatic
  fallback.
- **Records without an AEC value** return `None` with `energy_consumption_available: False`
  rather than an estimate.
- **No web interface** — interaction is CLI / Python only.

---

## Roadmap

1. **Phase 2 (in progress)** — Stage 1 OCR foundation and Stage 2 field parsing are
   complete. Remaining stages: OCR-error candidate generation, end-to-end extraction →
   retrieval bridge, CLI polish, and evaluation on more real photos.
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
