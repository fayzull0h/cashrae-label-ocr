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
   - **Stage 1 (Complete):** OCR foundation in `scripts/label_ocr.py` — Pillow preprocessing, PaddleOCR default engine with Tesseract fallback, batch CLI, confidence reporting. Evaluated on six verified rating plates.
   - **Stage 2 (Complete):** Rule-based field parsing in `scripts/label_parse.py` — brand, model number, and serial number from raw OCR text using anchors, adjacent-line lookups, and token heuristics. Live evaluation: 6/6 brand, 6/6 model, 6/6 serial (prefix-tolerant) on the six verified plates.
   - **Stage 3 (Complete):** OCR-error candidate generation in `scripts/label_candidates.py` — curated confusion map, one-substitution candidates by default, explainable edit trail and plausibility score, exact-only NRCan validation. Synthetic evaluation: recall@10 100% on the verified models; 265/265 true-model-first on sampled database models.
   - **Remaining stages:** extraction → retrieval bridge, CLI polish, evaluation on more real photos.
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
- **Core Libraries:** `pandas`, `Pillow`, `paddlepaddle` + `paddleocr` (default OCR engine), `pytesseract` (fallback OCR engine), `sqlite3` (standard library), `difflib` (standard library), `unittest` (standard library)
- **External Tool:** Tesseract OCR 5.x (fallback engine; install via `winget install --id UB-Mannheim.TesseractOCR -e`). PaddleOCR is a Python package and needs no external binary.
- **Platform:** Windows
- **Virtual Environment:** Located in `.venv/`
  - Activate in PowerShell: `.venv\Scripts\Activate.ps1`
  - Direct execution: `.venv\Scripts\python.exe <script_path>`
- **Version Control:** Git + GitHub
  - Remote `origin` → `git@github.com:fayzull0h/cashrae-label-ocr.git`
  - `main` is the default branch and must always be green (full test suite passing).
  - Experimental work goes on `experiment/<topic>` branches — never directly on `main`.
- **Working Directory Convention:** Always execute commands and scripts from the project root (`appliance-energy-tool/`).

---

## 4. Repository Structure

```
appliance-energy-tool/
├── .venv/                         # Main virtual environment (Python 3.13, includes OCR extras)
├── .venv-paddle/                  # Optional isolated venv for OCR experiments (gitignored)
├── .gitignore                     # Root ignore rules (venvs, caches, external photos, generated outputs)
├── data/
│   ├── raw/                       # NRCan source data (001_ to 010_)
│   │   ├── 001_.../               # .zip archives + nested extracted folders per category
│   │   └── ...                    # Each folder holds *_Data_*.csv, *_Dictionary_*.csv, *_Information.csv
│   ├── samples/                   # Bundled photo + ground truth; external/ is gitignored
│   │   └── README.md              # Where to download the external photo bundle
│   └── processed/                 # Debug images and derived outputs
├── database/
│   └── appliances.db              # SQLite database (10 appliance tables + 2 metadata tables)
├── scripts/
│   ├── import_datasets.py         # CSV ingestion pipeline into SQLite
│   ├── verify_import.py           # Database verification & sanity check script
│   ├── appliance_lookup.py        # Core retrieval engine (normalization, exact/fuzzy search)
│   ├── label_ocr.py               # Phase 2: preprocessing + OCR entry point (Paddle default)
│   ├── ocr_backends.py            # Engine selection and per-engine preprocessing defaults
│   ├── label_ocr_paddle.py        # PaddleOCR backend (fast profile, optional accurate mode)
│   ├── label_parse.py             # Phase 2: brand/model/serial parser (anchors + heuristics)
│   ├── label_candidates.py        # Phase 2: OCR-error candidates for model numbers
│   ├── evaluate_parse.py          # Per-field parsing accuracy over data/samples
│   ├── evaluate_candidates.py     # Candidate recall on synthetic OCR corruptions
│   ├── compare_ocr.py             # Tesseract vs PaddleOCR comparison harness
│   └── test_lookup.py             # CLI demonstration tool with interactive & demo modes
├── tests/
│   ├── test_lookup.py             # Integration test suite (12 unit tests covering Phase 1)
│   ├── test_label_ocr.py          # OCR foundation tests
│   ├── test_label_ocr_paddle.py   # PaddleOCR backend tests (integration skipped without Paddle)
│   ├── test_label_parse.py        # Field parsing tests (synthetic + six verified plates)
│   ├── test_label_candidates.py   # Candidate generation, ranking, and CLI tests
│   ├── test_evaluate_parse.py     # Evaluation helper tests
│   ├── test_compare_ocr.py        # Comparison metrics and reporting tests
│   └── test_ocr_backends.py       # Engine selection and fallback tests
├── docs/
│   └── paddleocr-experiment.md    # OCR engine evaluation and adoption record
├── requirements.txt               # pandas, Pillow, pytesseract (Tesseract fallback)
├── requirements-paddle.txt        # PaddleOCR dependencies (default engine)
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

### Phase 2 OCR Conventions (`scripts/label_ocr.py`, `scripts/ocr_backends.py`)

- **Default engine:** PaddleOCR with the tuned PP-OCRv5 mobile profile (`fast_profile_kwargs()`). Tesseract is an automatic fallback used when Paddle is not installed or cannot be initialised. `get_backend("auto"|"paddleocr"|"tesseract")` in `scripts/ocr_backends.py` is the single selection point; `auto` warns once and falls back.
- **Preprocessing is engine-specific:** PaddleOCR gets grayscale + autocontrast with **no upscaling** (photos longer than 5000 px are still clamped to `max_dimension`; Paddle caps detection at 960 px). Tesseract keeps the size-aware pipeline: images below `upscale_threshold` (1600 px) are upscaled, very large photos are clamped to `max_dimension` (5000 px), defaults grayscale + autocontrast, no sharpening, `psm=3`. Sharpening and aggressive downscaling measurably hurt dense rating plates.
- **Tesseract resolution:** The binary is resolved from `TESSERACT_CMD` → `PATH` → common Windows install locations; failures raise `OCRUnavailableError` with install instructions.
- **Pluggable engines:** `OCRBackend` protocol with `extract(image) -> (text, confidence)`. Keep new engines (e.g. a VLM) behind this interface so parsing/matching code does not change.
- **Raw text by default:** `extract_text()` returns `{image_path, raw_text, confidence (0–1), backend, preprocessing, word_count}`. The `--parse` CLI flag attaches a `parsed` block via `scripts.label_parse.py`; the OCR layer itself still does no field parsing.
- **Batch safety:** `extract_directory()` records per-file errors instead of aborting the batch.
- **PaddleOCR packaging:** `scripts/label_ocr_paddle.py` implements the `OCRBackend` protocol. Paddle dependencies live in `requirements-paddle.txt` (installed into `.venv` per the README) and are kept out of `requirements.txt` so the base install stays light. `.venv-paddle/` is only for isolated experiments. `paddlepaddle` must stay pinned to `3.2.2` on Windows (3.3.1 crashes with oneDNN). `PADDLE_PDX_CPU_NUM_THREADS` overrides the thread count; `--accurate` switches to PP-OCRv6 medium.
- **Confidence is not an accuracy gate:** measured cases show correct text at lower confidence than incorrect text, so do not threshold on confidence alone.

### Phase 2 Field Parsing Conventions (`scripts/label_parse.py`)

- **Entry points:** `parse_fields(raw_text)` returns `{brand, model_number, serial_number, unparsed_lines}`. Each field is `{value, normalized, confidence, source_line, method}`. `parse_ocr_result(result)` adds `image_path` and an `ocr` provenance block (`backend`, `confidence`, `preprocessing`).
- **Normalization:** every value is normalized with `normalize_alphanum()` from `scripts/appliance_lookup.py`, so parsed fields can be passed directly to the retrieval engine.
- **Extraction order:** explicit anchors (`MODEL`/`MODÈLE`/`MOD`, `SERIAL`/`SÉRIE`/`S/N`/`SER`/`SN`, `BRAND`/`MANUFACTURER`/`MARQUE`/`MFG`) → same-line value after the anchor → same-line value before the anchor → adjacent lines (forward, then backward for serials; backward only as a last resort for models) → heuristic token scoring. Brands also use a known-brand list (longest match wins) and a company-suffix fallback.
- **Scoring:** `model_score()` and `serial_score()` reject noise substrings (voltage, pressure, refrigerant, etc.), dates, and tokens lacking letters or digits where appropriate. `MODEL_MIN_SCORE = 0.95`, `SERIAL_MIN_SCORE = 0.9`. Model and serial values must never be identical; the serial is dropped when it equals the model.
- **Confidence is rule-based per field**, keyed by extraction method (`label_anchor` 0.9, `label_anchor_before` 0.85, `adjacent_line` 0.8, `known_brand` 0.8, `company_suffix`/`glued_anchor` 0.7, heuristics 0.5–0.6). Do not use aggregate OCR confidence as a gate.
- **Bilingual by default:** anchors and noise handling cover English and French (Canadian) labels.
- **Scope:** parsing stops at brand/model/serial and never "corrects" tokens. OCR-error candidate generation lives in `scripts/label_candidates.py` (Stage 3); the retrieval bridge is a later stage.
- **Evaluation:** `scripts/evaluate_parse.py` runs OCR + parsing over `data/samples` and reports per-field accuracy. Serial comparison is prefix-tolerant (a leading letter may be absent from ground truth).
- **Fixtures:** `data/processed/ocr_dump.json` is a tracked fixture containing the PaddleOCR raw text for all six verified plates; `tests/test_label_parse.py` uses it so parser tests run without the external photos.

### Phase 2 Candidate Generation Conventions (`scripts/label_candidates.py`)

- **Entry points:** `generate_candidates(model_number, max_edits=1, max_candidates=64)` returns the original first, then substitutions sorted by plausibility. `rank_candidates(candidates, manufacturer=..., category=..., validate=True)` adds `db_match`/`db_categories` and final ranks. `candidates_from_parse(parsed_fields, ...)` bridges parser output and returns `{original, manufacturer, category, validated, candidates}`.
- **Never rewrite the parsed value:** the original normalized model is always rank 1 with `is_original: True`. Candidates are suggestions for the retrieval stage, not in-place corrections.
- **Confusion map:** `CONFUSION_GROUPS` / `CONFUSION_MAP` are curated and symmetric (`0`/`O`, `1`/`I`/`L`, `2`/`Z`, `4`/`A`, `5`/`S`, `6`/`G`, `7`/`T`, `8`/`B`, `C`/`G`, `M`/`N`, `U`/`V`). Keep it narrow; every group multiplies the candidate list.
- **Bounds:** substitutions only; `max_edits` is 0–2 (2 is opt-in and noisier); `max_candidates` includes the original; candidates are deduplicated and deterministically ordered.
- **Ranking:** original → exact database-confirmed → fewer edits → higher `plausibility` → alphabetical. `plausibility` is the mean `class_fit`: whether the replacement matches the digit/letter class of its neighbours.
- **Validation:** use `find_exact_matches()` / `find_exact_matches_batch()` from `scripts/appliance_lookup.py`, which run the exact stage only (no fuzzy fallback). `db_match` is `None` when validation is skipped, so "not checked" stays distinguishable from "checked and absent". `rank_candidates(validator=...)` accepts an in-memory index for batch evaluation.
- **Evaluation:** `scripts/evaluate_candidates.py` corrupts the six verified models and a deterministic sample of NRCan models with every single confusion substitution. Recall is measured among corrections because the original always ranks first. Defaults: 40 database models, up to 25 corruptions each, seed 0.
- **Non-goals:** insertions/deletions, per-character OCR confidence (the OCR backend exposes only aggregate confidence), and automatic correction.

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
- **Run the field-parsing test suite:**
  ```bash
  .venv\Scripts\python.exe -m unittest tests/test_label_parse.py
  ```
- **Run OCR on a label photo (PaddleOCR default):**
  ```bash
  .venv\Scripts\python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png
  ```
- **Force the Tesseract fallback:**
  ```bash
  .venv\Scripts\python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png --engine tesseract
  ```
- **Run OCR over a directory (recurses into `data/samples/external/`):**
  ```bash
  .venv\Scripts\python.exe scripts/label_ocr.py --dir data/samples --json
  ```
- **Parse fields from a label photo (OCR + parser):**
  ```bash
  .venv\Scripts\python.exe scripts/label_parse.py --image data/samples/airconditioner_label.png
  ```
- **Attach parsed fields to the OCR CLI output:**
  ```bash
  .venv\Scripts\python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png --parse
  ```
- **Measure per-field parsing accuracy on the verified plates:**
  ```bash
  .venv\Scripts\python.exe scripts/evaluate_parse.py
  ```
- **Generate OCR-error candidates for a model number:**
  ```bash
  .venv\Scripts\python.exe scripts/label_candidates.py --model "PCKS433ET" --brand GE
  ```
- **Skip database validation (pure string ranking):**
  ```bash
  .venv\Scripts\python.exe scripts/label_candidates.py --model "PCKS433ET" --no-validate --json
  ```
- **Measure candidate-generation recall (synthetic corruptions):**
  ```bash
  .venv\Scripts\python.exe scripts/evaluate_candidates.py
  ```
- **Run the candidate-generation test suite:**
  ```bash
  .venv\Scripts\python.exe -m unittest tests/test_label_candidates.py
  ```
- **Run the PaddleOCR backend directly (isolated venv):**
  ```bash
  .venv-paddle\Scripts\python.exe scripts/label_ocr_paddle.py --image data/samples/airconditioner_label.png
  ```
- **Compare Tesseract vs PaddleOCR (default: Tesseract + tuned fast profile):**
  ```bash
  .venv\Scripts\python.exe scripts/compare_ocr.py --dir data/samples --ground-truth data/samples/ground_truth.json
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

---

## 9. Git Workflow & Repository Hygiene

- **Branch naming:** `feature/<topic>`, `fix/<topic>`, `experiment/<topic>`, `chore/<topic>`. Anything uncertain, heavy, or dependency-adding is an `experiment/` branch.
- **Never commit to `main` while its tests fail.** Run `.venv/Scripts/python.exe -m unittest discover -s tests` before every commit.
- **Commit messages:** short imperative subject with a prefix — `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `experiment:`, `refactor:`. Example: `feat: add PaddleOCR backend behind OCRBackend protocol`.
- **Small, single-purpose commits.** Do not mix a dependency change with a refactor.
- **Never force-push `main`** and never rewrite shared history. Force-push is acceptable only on your own unmerged `experiment/` branch.
- **No secrets or machine-specific paths** in commits. Use environment variables such as `TESSERACT_CMD`.

### What is tracked on purpose

- `database/appliances.db` — reproducible build artifact, kept for zero-setup runs.
- `data/raw/**` — authoritative NRCan source CSVs; do not edit or regenerate.
- `data/samples/*` — the bundled air-conditioner photo, `ground_truth.json`, and `README.md`.
- `data/processed/ocr_dump.json` — tracked fixture: PaddleOCR raw text for the six verified plates, used by parser tests.

### What must never be committed

- `.venv/`, `.venv-paddle/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`
- `data/samples/external/**` — large label photos (5–12 MB each) pulled from the shared drive; see `data/samples/README.md`.
- New files in `data/processed/` (debug images, comparison dumps) unless deliberately promoted to fixtures.
- PaddleOCR model caches (`~/.paddlex` or any `PADDLE_PDX_CACHE_HOME` inside the repo).
- Files larger than 10 MB without discussing them first.

### Known cleanup

`scripts/__pycache__/*.pyc` and `tests/__pycache__/*.pyc` were tracked in the initial commit. They are removed from the index by `.gitignore` plus `git rm -r --cached`; do not delete the working files.

