# PaddleOCR evaluation and adoption

**Status:** complete — adopted as the default OCR engine
**Decision:** PaddleOCR with the tuned PP-OCRv5 mobile profile is the default; Tesseract
remains an automatic fallback.

This started as an isolated experiment on `experiment/paddleocr` and was merged after the
six-photo evaluation met every decision criterion. `scripts/ocr_backends.py` is the single
engine-selection point.

---

## 1. Decision

| Question | Answer |
|---|---|
| Default engine | `paddleocr` (tuned PP-OCRv5 mobile det/rec) |
| Fallback | `tesseract`, used automatically when Paddle is missing or fails to initialise |
| Preprocessing | Grayscale + autocontrast, **no resize** for Paddle; existing size-aware pipeline for Tesseract |
| How to override | `--engine tesseract` / `--engine paddleocr`; `--accurate` for PP-OCRv6 medium |
| Why | +67 pp model-number recall over Tesseract at lower CPU latency |

The model number is the field that drives the Phase 1 retrieval engine, so its recall was
the deciding metric. On six verified plates, Tesseract read 1 of 6 model numbers correctly;
PaddleOCR read 6 of 6.

## 2. Method

### Backends

| Engine name | Configuration |
|---|---|
| `tesseract` | `scripts/label_ocr.py`, size-aware Pillow pipeline, `psm=3` |
| `paddleocr-fast` | PP-OCRv5 mobile det/rec, ≤16 CPU threads, recognition batch 32, detection side limit 960 |
| `paddleocr` | PP-OCRv6 medium (PaddleOCR default), oneDNN enabled — reference only, ~25× slower |

All implement the same `OCRBackend.extract(image) -> (text, confidence)` protocol.

### Preprocessing variants

| Variant | Pipeline |
|---|---|
| `raw` | Original image, no preprocessing |
| `gray` | Grayscale + autocontrast, no resize, no sharpening |
| `pipeline` | Tesseract defaults (grayscale, autocontrast, size-aware resize to 5000 px) |

### Metrics

Field parsing is a later stage, so the comparison is at raw-text level. For each
ground-truth field (`brand`, `model_number`, `serial_number`):

- **Exact recall** — the normalized field value appears as a substring of the normalized
  OCR text (reuses `normalize_alphanum()` from Phase 1).
- **Mean similarity** — best `SequenceMatcher` ratio against a sliding window of
  `len(expected) ± 2` characters, tolerating OCR splits and single-character confusions.
- **Confidence** — each engine's own 0–1 score.
- **Latency** — wall-clock seconds per image after model load.

### Fixtures

Six hand-verified rating plates spanning five appliance categories:

| File | Category | Brand | Model |
|---|---|---|---|
| `airconditioner_label.png` | room air conditioners | International Comfort Products | `NXA630GKA100` |
| `dishwasher_label.png` | dishwashers | KitchenAid | `KUDC10FXSS7` |
| `dryer_label.png` | clothes dryers | LG | `DLE8400WE` |
| `garage_fridge_label.png` | refrigerators | General Electric | `GTH17GBTZRWW` |
| `garage_minifridge_label.png` | refrigerators | Danby | `DAR044A4WDD-6` |
| `washer_label.png` | clothes washers | LG | `WT8400CW` |

The air-conditioner photo is committed. The other five are large (5–12 MB each) and live
outside git; see `data/samples/README.md` for the external bundle.

## 3. Environment

| Component | Version |
|---|---|
| Python | 3.13.0 |
| Pillow | 13.0.0 |
| pytesseract | 0.3.13 |
| paddlepaddle | 3.2.2 |
| paddleocr | 3.7.0 |
| paddlex | 3.7.2 |
| Tesseract binary | 5.4.0 |
| CPU | 22 logical processors, no GPU |

| Footprint | Size |
|---|---|
| Paddle packages in `.venv` | ~700 MB |
| Paddle model cache (`~/.paddlex`) | 246 MB |
| First-run download | ~246 MB |
| Model load (warm cache) | ~1–4 s |

The five external photos are not part of the install footprint.

## 4. Results

Full output: `data/processed/ocr_comparison.md` (gitignored). Aggregate across all six
plates:

| Engine | Variant | Mean exact recall | Mean similarity | Mean latency (s) |
|---|---|---|---|---|
| **paddleocr-fast (v5 mobile)** | **gray** | **1.000** | **1.000** | **2.75** |
| **paddleocr-fast (v5 mobile)** | **pipeline** | **1.000** | **1.000** | **2.52** |
| paddleocr-fast (v5 mobile) | raw | 0.945 | 0.997 | 3.12 |
| tesseract | gray | 0.333 | 0.667 | 4.19 |
| tesseract | pipeline | 0.333 | 0.667 | 3.98 |
| tesseract | raw | 0.333 | 0.629 | 7.88 |
| paddleocr (v6 medium) | raw | 1.000 | 1.000 | ~56 |
| paddleocr (v6 medium) | gray | 0.945 | 0.994 | ~57 |
| paddleocr (v6 medium) | pipeline | 0.945 | 0.994 | ~55 |

Latency is machine- and load-dependent: repeated runs of the fast profile ranged from
~1.6 s to ~2.8 s per plate on the same CPU. The accuracy columns were identical across
runs. PP-OCRv6 medium is shown for reference only; it is ~20–35× slower and did not beat
the mobile profile once preprocessing was applied, so it is not in the default comparison.

### Per-field breakdown (`gray` variant, six plates)

| Field | Tesseract | PaddleOCR fast |
|---|---|---|
| brand | 3 / 6 | **6 / 6** |
| model_number | 1 / 6 | **6 / 6** |
| serial_number | 2 / 6 | **6 / 6** |

Per-image model-number results:

| Image | Tesseract | PaddleOCR fast |
|---|---|---|
| air conditioner | ❌ `0.64` similarity | ✅ exact |
| dishwasher | ❌ `0.30` | ✅ exact |
| dryer | ✅ exact | ✅ exact |
| garage fridge | ❌ `0.31` | ✅ exact |
| minifridge | ❌ `0.70` | ✅ exact |
| washer | ❌ `0.44` | ✅ exact |

### Observations that shaped the defaults

- **Grayscale matters for Paddle on the dishwasher plate.** Raw input missed the serial
  number (recall 0.667); grayscale recovered it (1.000). Hence grayscale + autocontrast is
  the Paddle default. Resizing added cost without helping, so it is disabled for Paddle.
- **Confidence is not an accuracy gate.** The dishwasher plate scored 1.000 recall at
  0.704 confidence with grayscale, while the raw variant scored 0.667 at 0.790 confidence.
  Do not threshold on confidence alone.
- **The pipeline variant ties grayscale for Paddle** because Paddle re-caps detection at
  960 px. Grayscale is preferred because it avoids a redundant full-resolution resize.
- **Tesseract remains fast and useful** on clean plates (the dryer plate is perfect), which
  is why it stays as a fallback rather than being removed.

### Rejected configurations

| Configuration | Result |
|---|---|
| paddlepaddle 3.3.1 + oneDNN | **Crash**: `ConvertPirAttribute2RuntimeAttribute` in `onednn_instruction.cc` |
| paddlepaddle 3.3.1, oneDNN disabled | Works, but v6 medium 71 s, v5 mobile 25 s |
| paddlepaddle 3.2.2, oneDNN disabled | v6 medium 114 s |
| PP-OCRv5 server models | Slower than mobile without a measured accuracy gain; not pursued |
| Recognition batch 32 alone | Marginal change (76 s vs 71 s on v6 medium) |

## 5. Decision criteria

| Criterion | Required | Measured | Verdict |
|---|---|---|---|
| Model-number recall gain over Tesseract | ≥ 10 pp | +67 pp (1/6 → 6/6) | ✅ |
| Overall field recall | ≥ Tesseract | 1.000 vs 0.333 | ✅ |
| Warm latency on CPU | ≤ 5 s/image | 1.6–2.8 s | ✅ |
| Install + model footprint | < ~2 GB | ~0.95 GB | ✅ |
| Offline after first download | Yes | Models cached and reused | ✅ |
| Verified sample size | ≥ 5 photos | 6 photos, 5 categories | ✅ |
| Low-core behaviour | Documented / tunable | 4 threads: 5.68 s, still 1.000 recall | ✅ documented |

## 6. Adoption details

- `scripts/ocr_backends.py` — `get_backend("auto"|"paddleocr"|"tesseract")`,
  per-engine preprocessing defaults, and a single one-time fallback warning.
- `scripts/label_ocr.py` — defaults to `auto` (Paddle if available), adds `--engine`,
  `--lang`, `--device`, `--accurate`, and keeps `--psm` for Tesseract.
- `scripts/label_ocr_paddle.py` — default engine, tuned fast profile, thread override.
- `scripts/compare_ocr.py` — default backends are `tesseract,paddleocr-fast`; PP-OCRv6
  medium is opt-in via `--backends`.
- `requirements-paddle.txt` — installed into `.venv`; kept separate from `requirements.txt`
  so the base install stays light. `paddlepaddle` stays pinned to `3.2.2` on Windows.
- `data/samples/external/` — large photos are gitignored; `data/samples/README.md`
  documents the download.

## 7. Reproduction

```bash
# From the repository root
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -r requirements-paddle.txt

# Full test suite (Paddle integration test runs instead of skipping)
.venv/Scripts/python.exe -m unittest discover -s tests

# Single-image smoke test (fast profile)
.venv/Scripts/python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png

# Accurate but slow profile (PP-OCRv6 medium)
.venv/Scripts/python.exe scripts/label_ocr.py --image data/samples/airconditioner_label.png --accurate

# Six-photo comparison (Tesseract vs tuned fast profile)
.venv/Scripts/python.exe scripts/compare_ocr.py \
  --dir data/samples \
  --ground-truth data/samples/ground_truth.json

# Include PP-OCRv6 medium for reference
.venv/Scripts/python.exe scripts/compare_ocr.py \
  --dir data/samples \
  --ground-truth data/samples/ground_truth.json \
  --backends tesseract,paddleocr,paddleocr-fast
```

An isolated `.venv-paddle/` is still supported for future engine experiments, but it is no
longer required for normal use.

## 8. Stage 2 field parsing (downstream check)

The parser in `scripts/label_parse.py` was evaluated end-to-end on the same six plates with
the default PaddleOCR fast profile (`scripts/evaluate_parse.py`):

| Field | Correct | Notes |
|---|---|---|
| brand | 6 / 6 | canonical names from the known-brand list |
| model_number | 6 / 6 | exact after `normalize_alphanum()` |
| serial_number | 6 / 6 | prefix-tolerant; the air-conditioner serial keeps its `E` prefix |

All six plates also pass the "all three fields" check. Per-image latency was 1.5–4.5 s,
including OCR. This confirms that the raw-text advantage measured above carries through to
structured extraction; it does not yet prove the extraction → retrieval bridge (Phase 3).

```bash
.venv/Scripts/python.exe scripts/evaluate_parse.py
```

## 9. Stage 3 OCR-error candidates (downstream check)

`scripts/label_candidates.py` turns a parsed model number into ranked corrections for the
confusions this engine actually produces. Because none of the six verified models exist in
the NRCan data, the candidates cannot be database-confirmed there, so recall is measured on
string ranking over every single-character corruption of each verified model:

| Metric (corrections only) | Result |
|---|---|
| recall@1 | 9 / 49 (18.4%) |
| recall@5 | 43 / 49 (87.8%) |
| recall@10 | 49 / 49 (100.0%) |
| MRR | 0.4454 |

The low recall@1 is expected without a dictionary: `L` and `I` are genuinely ambiguous in a
letter run, and only a database hit can disambiguate them. That is what validation is for.
Sampling 40 real NRCan models and corrupting them the same way gives a very different
picture, because the true model is present in the database:

| Metric (265 cases, exact validation) | Result |
|---|---|
| true model confirmed | 265 / 265 (100.0%) |
| true model is top DB-confirmed candidate | 265 / 265 (100.0%) |
| only one DB-confirmed candidate | 265 / 265 (100.0%) |
| MRR | 1.0000 |

So the pipeline for Stage 4 is: try the parsed model, then the ranked corrections, and let
the exact NRCan lookup decide. Candidate generation never rewrites the parsed value.

```bash
.venv/Scripts/python.exe scripts/evaluate_candidates.py
```

## 10. Risks and limitations

- **Small sample.** Six plates across five categories is enough to justify the switch, but
  not to claim universal accuracy. Plate designs vary widely; keep collecting fixtures.
- **Windows-specific bug.** `paddlepaddle 3.3.1` crashes with oneDNN on this platform; the
  pin to `3.2.2` is load-bearing.
- **Thread dependence.** The fast profile uses up to 16 CPU threads and was measured on a
  22-thread desktop. On a 4-core laptop latency will be higher: forcing
  `PADDLE_PDX_CPU_NUM_THREADS=4` raised the largest plate from ~2.8 s to 5.68 s, though
  accuracy stayed at 1.000. Set the variable to the core count.
- **Model drift.** PaddleOCR downloads versioned weights on first run; pinning the package
  does not pin the weights.
- **Raw-text metrics → parsing.** Stage 2 parsing now reaches 6/6 on the verified plates,
  but end-to-end retrieval accuracy is still unmeasured; that is the Phase 3 bridge.
- **Candidates are substitution-only.** Stage 3 generates single-character confusion swaps
  (two edits are opt-in) and does not model insertions, deletions, or merged characters.
  Validation confirms exact normalized matches only, so a correction that is not in NRCan
  remains unconfirmed rather than being guessed.
- **Large external fixtures.** The six-photo benchmark is only reproducible after pulling
  the external bundle; the committed air-conditioner photo still works on its own.
