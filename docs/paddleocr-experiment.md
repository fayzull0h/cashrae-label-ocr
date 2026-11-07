# PaddleOCR experiment

**Status:** complete (single-photo trial) — branch `experiment/paddleocr`
**Decision:** provisional — **keep Tesseract as the default for now**; PaddleOCR's tuned
fast profile is the leading candidate once more photos are verified.

This is an experiment, not a migration. Tesseract remains the default on `main` and in
`scripts/label_ocr.py` throughout.

---

## 1. Why isolate it

| Concern | Mitigation |
|---|---|
| Paddle pulls in `paddlex`, OpenCV, and a large transitive tree | Separate `.venv-paddle/` environment; `requirements.txt` untouched |
| Python 3.13 support in the Paddle ecosystem is new | `paddlepaddle 3.2.2` ships a `cp313-win_amd64` wheel and is stable on this machine |
| Model downloads (~250 MB) on first run | Measured; caches stay in `~/.paddlex`, outside the repo |
| Experiment contaminating `main` | All work on `experiment/paddleocr`; default backend never changed on the branch |

## 2. Method

### Backends

| Engine name | Configuration |
|---|---|
| `tesseract` | `scripts/label_ocr.py`, current production default |
| `paddleocr` | PP-OCRv6 medium (PaddleOCR default), oneDNN enabled |
| `paddleocr-fast` | PP-OCRv5 mobile det/rec, 16 CPU threads, recognition batch 32, detection side limit 960 |

All three implement the same `OCRBackend.extract(image) -> (text, confidence)` protocol.

### Preprocessing variants

| Variant | Pipeline |
|---|---|
| `raw` | Original image, no preprocessing |
| `gray` | Grayscale + autocontrast, no resize, no sharpening |
| `pipeline` | Current Tesseract defaults (grayscale, autocontrast, size-aware resize to 5000 px, no sharpening) |

### Metrics

Field parsing is a later stage, so the comparison is at raw-text level. For each
ground-truth field (`brand`, `model_number`, `serial_number`):

- **Exact recall** — normalized field value appears as a substring of the normalized
  OCR text (reuses `normalize_alphanum()` from Phase 1).
- **Mean similarity** — best `SequenceMatcher` ratio against a sliding window of
  `len(expected) ± 2` characters, tolerating OCR splits and single-character confusions.
- **Confidence** — each engine's own 0–1 score.
- **Latency** — wall-clock seconds per image after model load.

### Ground truth

`data/samples/ground_truth.json`:

| Field | Value | How it was verified |
|---|---|---|
| brand | International Comfort Products | Printed on the plate |
| model_number | NXA630GKA100 | Confirmed against ICP/Arcoaire product documentation |
| serial_number | 140302785 | Read from the plate; E/F prefix not scored |

Only one photo is currently available, so the sample is too small for a final engine
swap. The decision criteria require at least five verified photos.

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
| `.venv-paddle` | 879 MB |
| Paddle model cache (`~/.paddlex`) | 246 MB |
| **Total** | **~1.1 GB** |
| First-run download | ~246 MB |
| Model load (warm cache) | ~1–4 s |

## 4. Results

Full output: `data/processed/ocr_comparison.md` (gitignored). Aggregate for
`airconditioner_label.png`:

| Engine | Variant | Exact recall | Mean similarity | Latency (s) |
|---|---|---|---|---|
| paddleocr (v6 medium) | raw | 1.000 | 1.000 | 74.28 |
| paddleocr (v6 medium) | gray | 1.000 | 1.000 | 74.23 |
| paddleocr (v6 medium) | pipeline | 1.000 | 1.000 | 76.30 |
| **paddleocr-fast (v5 mobile)** | raw | **1.000** | **1.000** | **2.95** |
| **paddleocr-fast (v5 mobile)** | gray | **1.000** | **1.000** | **2.87** |
| **paddleocr-fast (v5 mobile)** | pipeline | **1.000** | **1.000** | **2.93** |
| tesseract | raw | 0.667 | 0.870 | 2.37 |
| tesseract | gray | 0.667 | 0.879 | 1.48 |
| tesseract | pipeline | 0.667 | 0.879 | 1.60 |

### Per-field breakdown

| Field | Tesseract | PaddleOCR (both tiers) |
|---|---|---|
| brand | ✅ exact | ✅ exact |
| model_number | ❌ read `NXAG3EGKA100` (similarity 0.61–0.64) | ✅ `NXA630GKA100` |
| serial_number | ✅ exact | ✅ exact |

The single miss is the field that matters most: the model number drives the Phase 1
lookup. Tesseract confused `6→G` and `0→E`; Phase 2 Stage 3 candidate generation may
recover it, but PaddleOCR got it right with no correction.

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
| Field recall gain over Tesseract | ≥ 10 pp | +33 pp | ✅ (n = 1) |
| Warm latency on CPU | ≤ 5 s/image | 2.87–2.95 s (fast tier) | ✅ |
| Install + model footprint | < ~2 GB | ~1.1 GB | ✅ |
| Offline after first download | Yes | Models cached and reused | ✅ (cache reuse verified; full network-off test pending) |
| Verified sample size | ≥ 5 photos | 1 photo | ❌ |

## 6. Outcome

**Keep Tesseract as the default for now.** The tuned `paddleocr-fast` profile is a
strong candidate — it matches or beats Tesseract on every measured field at roughly the
same latency — but the evidence comes from one image. A 33 pp gain on a single plate is
not enough to justify a default-engine swap.

Next steps before switching:

1. Add at least five verified photos spanning refrigerators, ranges, dishwashers, and
   washers/dryers.
2. Re-run `scripts/compare_ocr.py` and confirm the recall advantage holds.
3. Test with the network disabled to confirm fully offline operation.
4. Measure on a lower-core laptop, since the fast profile depends on CPU threads.
5. If the advantage holds, merge the backend and make `paddleocr-fast` the default in
   Phase 2 Stage 3, keeping Tesseract as a fallback.

## 7. Reproduction

```bash
# From the repository root, on branch experiment/paddleocr
.venv/Scripts/python.exe -m venv .venv-paddle
.venv-paddle/Scripts/python.exe -m pip install -r requirements-paddle.txt

# Full test suite (Paddle integration test runs instead of skipping)
.venv-paddle/Scripts/python.exe -m unittest discover -s tests

# Single-image smoke test (fast profile)
.venv-paddle/Scripts/python.exe scripts/label_ocr_paddle.py --image data/samples/airconditioner_label.png

# Accurate but slow profile (PP-OCRv6 medium)
.venv-paddle/Scripts/python.exe scripts/label_ocr_paddle.py --image data/samples/airconditioner_label.png --accurate

# Side-by-side comparison
.venv-paddle/Scripts/python.exe scripts/compare_ocr.py \
  --dir data/samples \
  --ground-truth data/samples/ground_truth.json
```

## 8. Risks and limitations

- **One image.** The accuracy result does not generalise to other appliance categories
  or plate designs.
- **Windows-specific bug.** `paddlepaddle 3.3.1` crashes with oneDNN on this platform;
  the pin to `3.2.2` is load-bearing.
- **Thread dependence.** The fast profile uses up to 16 CPU threads. On a 4-core laptop
  latency will be higher and may exceed the 5 s target.
- **Model drift.** PaddleOCR downloads versioned model weights on first run; pinning the
  package does not pin the weights.
- **Raw-text metrics.** Better raw text does not guarantee better end-to-end retrieval;
  that is measured after Stage 2 field parsing exists.
