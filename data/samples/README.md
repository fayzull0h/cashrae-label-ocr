# Sample label photos

This directory holds the fixtures used to evaluate the OCR pipeline.

## What is tracked

| File | Source | Notes |
|---|---|---|
| `airconditioner_label.png` | Committed | Room air-conditioner rating plate (ICP `NXA630GKA100`) |
| `ground_truth.json` | Committed | Hand-verified field values for every fixture |
| `external/` | **Not tracked** | Large photos downloaded separately (see below) |

## External photo bundle

The remaining fixtures are large (5–12 MB each) and are **not stored in git**.
Download them from the shared Google Drive folder and place them in
`data/samples/external/`:

> **TODO:** add the Google Drive link here.

Expected filenames (the comparison harness matches them to `ground_truth.json`
by basename, and `find_images()` searches subdirectories automatically):

| File | Category | Brand | Model |
|---|---|---|---|
| `dishwasher_label.png` | dishwashers | KitchenAid | `KUDC10FXSS7` |
| `dryer_label.png` | clothes dryers | LG | `DLE8400WE` |
| `garage_fridge_label.png` | refrigerators | General Electric | `GTH17GBTZRWW` |
| `garage_minifridge_label.png` | refrigerators | Danby | `DAR044A4WDD-6` |
| `washer_label.png` | clothes washers | LG | `WT8400CW` |

If a file is missing, the harness simply skips it; the bundled air-conditioner
photo and its ground truth still work on their own.

## Running the comparison

```bash
.venv-paddle/Scripts/python.exe scripts/compare_ocr.py \
  --dir data/samples \
  --ground-truth data/samples/ground_truth.json
```

Results are written to `data/processed/ocr_comparison.json` and
`data/processed/ocr_comparison.md` (both gitignored).
