# AGENTS.md - Appliance Energy Information Retrieval Tool

This document provides context, architecture details, conventions, and operational instructions for LLM agents working on the `appliance-energy-tool` codebase.

---

## 1. Project Overview & Objectives

- **Context:** University Capstone Project focused on automating aspects of the ASHRAE building energy audit process using computer vision, data processing, and retrieval.
- **Goal:** Enable an auditor or user to take a photograph of an appliance rating plate/label, extract identifying information (brand, model number), and retrieve authoritative technical data—primarily **Annual Energy Consumption (AEC in kWh/year)**.
- **Key Insight:** Rating plates rarely list annual energy consumption directly; this data must be retrieved from authoritative external datasets (starting with Natural Resources Canada - NRCan) using the extracted model/brand.
- **Target Appliance Categories (10):**
  - Clothes dryers
  - Clothes washer-dryers
  - Clothes washers
  - Cooktops
  - Dishwashers
  - Freezers
  - Ovens
  - Ranges
  - Refrigerators and wine chillers
  - Room air conditioners

---

## 2. Planned Development Phases

1. **Phase 1: Appliance Database and Retrieval Engine**
   - **Phase 1A (Complete):** Acquire NRCan CSV datasets, extract schemas, preserve field dictionaries, and construct a structured SQLite database.
   - **Phase 1B (Implemented & Verified):** Build normalization, exact matching, and fuzzy matching retrieval engine for model numbers and manufacturers across heterogeneous energy columns.
2. **Phase 2: Appliance Label Recognition and Information Extraction (In Progress)**
   - **Stage 1 (Complete):** OCR foundation in `scripts/label_ocr.py` — Pillow preprocessing, Tesseract backend, batch CLI, confidence reporting.
   - **Remaining stages:** field parsing, OCR-error candidate generation, extraction → retrieval bridge, CLI polish, evaluation on real photos.
   - Image preprocessing, OCR / Vision-Language Model (VLM) extraction of brand, model number, serial number, and electrical specs.
3. **Phase 3: Integration of Image Recognition and Retrieval**
   - Connect OCR extraction pipeline to the Phase 1 retrieval engine with confidence scoring and manual fallback/correction mechanisms.
4. **Phase 4: Expanded Information Sources**
   - Manufacturer manuals, secondary datasets, external specification APIs with clear source attribution.
5. **Phase 5: Application Interface & Reporting**
   - User-facing UI, audit-ready exportable reports.

---

## 3. Tech Stack & Environment

- **Language:** Python 3.13+
- **Database:** SQLite3 (`database/appliances.db`)
- **Core Libraries:** `pandas`, `Pillow`, `pytesseract`, `sqlite3` (standard library), `difflib` (standard library), `unittest` (standard library)
- **External Tool:** Tesseract OCR 5.x (required for Phase 2; install via `winget install --id UB-Mannheim.TesseractOCR -e`)
- **Platform:** Windows
- **Virtual Environment:** Located in `.venv/`
  - Activate in PowerShell: `.venv\Scripts\Activate.ps1`
  - Direct execution: `.venv\Scripts\python.exe <script_path>`
- **Working Directory Convention:** Always execute commands and scripts from the project root (`appliance-energy-tool/`).

---

## 4. Repository Structure

```
appliance-energy-tool/
├── .venv/                         # Python virtual environment (Python 3.13)
├── data/
│   ├── raw/                       # NRCan source data (001_ to 010_)
│   │   ├── 001_.../               # .zip archives + nested extracted folders per category
│   │   └── ...                    # Each folder holds *_Data_*.csv, *_Dictionary_*.csv, *_Information.csv
│   ├── samples/                   # Real appliance label photos for OCR testing
│   └── processed/                 # Debug images and derived outputs
├── database/
│   └── appliances.db              # SQLite database (10 appliance tables + 2 metadata tables)
├── scripts/
│   ├── import_datasets.py         # CSV ingestion pipeline into SQLite
│   ├── verify_import.py           # Database verification & sanity check script
│   ├── appliance_lookup.py        # Core retrieval engine (normalization, exact/fuzzy search)
│   ├── label_ocr.py               # Phase 2: preprocessing + Tesseract OCR backend
│   └── test_lookup.py             # CLI demonstration tool with interactive & demo modes
├── tests/
│   ├── test_lookup.py             # Integration test suite (12 unit tests covering Phase 1)
│   └── test_label_ocr.py          # OCR foundation tests (23 tests)
├── requirements.txt               # pandas, Pillow, pytesseract
├── AGENTS.md                      # Agent context and reference document (this file)
├── README.md                      # Project overview, status, setup, usage, and limitations
└── handoff-prompt.md              # Detailed project requirements and background specification
```

---

## 5. Database Schema & Architecture

The database (`database/appliances.db`) contains **6,697 appliance records** across 12 tables:

### Appliance Tables (10)
- `clothes_dryers` (298 rows)
- `clothes_washer_dryers` (43 rows)
- `clothes_washers` (414 rows)
- `cooktops` (144 rows)
- `dishwashers` (600 rows)
- `freezers` (425 rows)
- `ovens` (276 rows)
- `ranges` (366 rows)
- `refrigerators` (3,021 rows)
- `room_air_conditioners` (1,110 rows)

### Metadata Tables (2)
- `datasets`: Ingestion dates, filenames, and English/French product names.
- `field_dictionary`: 163 rows mapping original NRCan column names, descriptions, data types, and French/English labels.

### Heterogeneous Energy Columns
NRCan uses different column names for annual energy consumption across appliance types:
- `clothes_dryers`, `clothes_washer_dryers`, `freezers`, `refrigerators`: `AEC`
- `clothes_washers`: `MOD_J1_AEC`
- `cooktops`, `ovens`, `ranges`: `AEC_03`
- `dishwashers`: `TAEC_C373_04`
- `room_air_conditioners`: Does **not** report kWh/year AEC in NRCan data; reports `EE_RATIO` (Energy Efficiency Ratio) instead.

---

## 6. Core Retrieval Engine Conventions (`scripts/appliance_lookup.py`)

- **Primary Entry Points:**
  ```python
  from scripts.appliance_lookup import get_annual_energy_consumption, lookup_appliance

  # High-level lookup
  result = get_annual_energy_consumption(
      manufacturer="GE",
      model_number="PCKS433ET#**",
      category="clothes_dryers"  # Optional
  )
  ```
- **Normalization:**
  - `normalize_alphanum()` strips spaces, hyphens, wildcards, and punctuation, converting text to uppercase alphanumeric strings.
  - Custom SQLite function `NORMALIZE` is registered during DB connection for efficient SQL-level matching.
  - `normalize_category()` maps user input to a table name: case-insensitive, accepts hyphens/spaces, strips numeric prefixes (`001_`), and resolves aliases such as `dryer`, `air_conditioners`, and `refrigerators_wine_chillers`. Unrecognized categories raise `ValueError` (surfaced as `status: "invalid_input"`).
- **Two-Stage Matching:**
  - **Stage A (Exact):** Queries normalized model number (and brand if supplied). Returns immediately if a unique match is found.
  - **Stage B (Fuzzy):** Uses Python's `difflib.SequenceMatcher` with a minimum confidence threshold (`MIN_FUZZY_CONFIDENCE = 0.70`). If a manufacturer is supplied, only rows of that exact normalized brand are compared first; if none exist, the search broadens to the whole table. When the brand differs, the score is blended as `model_score * 0.85 + brand_score * 0.15` (a penalty, not a boost). `HIGH_CONFIDENCE_THRESHOLD = 0.85` marks a reliable fuzzy match (defined but not currently enforced separately).
- **Ambiguity Detection:**
  - If multiple exact matches exist across categories or if top fuzzy candidate scores are within `AMBIGUITY_MARGIN = 0.04`, the engine returns `status: "ambiguous_match"` with candidate lists rather than guessing. Ambiguous results deliberately return `annual_energy_consumption: None` so no single value is implied. Fuzzy ambiguity returns up to 10 candidates; a clear `fuzzy_match` also includes the top 5 in `candidates`.
- **Return Statuses:** `exact_match`, `fuzzy_match`, `ambiguous_match`, `not_found`, `invalid_input`. Exact matches include `is_exact: True`; a model-only exact match adds a `warning` that no manufacturer was specified.
- **Honest Energy Handling:**
  - Never fabricate or infer missing energy numbers. If a record lacks AEC (or the category reports EER like room ACs), return `annual_energy_consumption: None`, `energy_consumption_available: False`, and explanatory notes.

### Phase 2 OCR Conventions (`scripts/label_ocr.py`)

- **Backend:** Tesseract via `pytesseract`. The binary is resolved from `TESSERACT_CMD` → `PATH` → common Windows install locations; failures raise `OCRUnavailableError` with install instructions.
- **Preprocessing is size-aware:** images below `upscale_threshold` (1600 px) are upscaled; very large photos are clamped to `max_dimension` (5000 px). Defaults: grayscale + autocontrast, no sharpening, `psm=3`. Sharpening and aggressive downscaling measurably hurt dense rating plates.
- **Pluggable engines:** `OCRBackend` protocol with `extract(image) -> (text, confidence)`. Keep new engines (e.g. a VLM) behind this interface so parsing/matching code does not change.
- **Raw text only:** `extract_text()` returns `{image_path, raw_text, confidence (0–1), backend, preprocessing, word_count}`. Field parsing belongs in a later module, not here.
- **Batch safety:** `extract_directory()` records per-file errors instead of aborting the batch.

---

## 7. Common Commands & Verification

Run all commands using the project `.venv` from the repository root:

- **Run database verification:**
  ```bash
  .venv\Scripts\python.exe scripts/verify_import.py
  ```
- **Run the full test suite:**
  ```bash
  .venv\Scripts\python.exe -m unittest discover -s tests
  ```
- **Run the Phase 1 test suite:**
  ```bash
  .venv\Scripts\python.exe -m unittest tests/test_lookup.py
  ```
- **Run the OCR test suite:**
  ```bash
  .venv\Scripts\python.exe -m unittest tests/test_label_ocr.py
  ```
- **Run OCR on a label photo:**
  ```bash
  .venv\Scripts\python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png
  ```
- **Run OCR over a directory:**
  ```bash
  .venv\Scripts\python.exe scripts/label_ocr.py --dir data/samples --json
  ```
- **Run CLI demo:**
  ```bash
  .venv\Scripts\python.exe scripts/test_lookup.py --demo
  ```
- **Run ad-hoc CLI lookup:**
  ```bash
  .venv\Scripts\python.exe scripts/test_lookup.py --manufacturer "GE" --model "PCKS-433-ET"
  ```
- **Re-import datasets from CSVs:**
  ```bash
  .venv\Scripts\python.exe scripts/import_datasets.py
  ```

---

## 8. Development Guidelines for Future Agents

1. **Preserve Database Authenticity:** Do not alter original NRCan values. Keep changes repeatable via `scripts/import_datasets.py`.
2. **Prioritize Simplicity & Robustness:** Prefer Python standard library and lightweight, explainable algorithms over heavy machine-learning architectures or external paid APIs unless explicitly required.
3. **Handle Noisy Inputs Gracefully:** When building Phase 2 (OCR), anticipate character substitution errors (e.g., `0` vs `O`, `1` vs `I`/`l`, `8` vs `B`) and format inconsistencies.
4. **Preserve Output Schema Integrity:** Any additions to the retrieval engine or OCR pipeline should adhere to structured dictionaries with confidence metrics and complete source provenance.
5. **Keep `README.md` in Sync:** Whenever you change functionality — adding, removing, or altering scripts, APIs, CLI flags, database schema, dependencies, test counts, phase status, or known limitations — update `README.md` in the same change. The README is the user-facing source of truth and must accurately reflect the current state of the project. If a change does not affect behavior, the README may stay as-is; when in doubt, update it.
