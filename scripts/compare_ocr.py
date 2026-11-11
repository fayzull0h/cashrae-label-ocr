"""Side-by-side OCR comparison harness (Tesseract vs PaddleOCR).

Runs both engines over a directory of label photos under several preprocessing
variants and reports raw-text metrics against a hand-written ground-truth file.

This intentionally measures *raw text* only. Field parsing is a later Phase 2
stage, so the metrics are token recall and best-window similarity for the
brand / model / serial strings that are expected to appear on the plate.

Usage::

    .venv/Scripts/python.exe scripts/compare_ocr.py \
        --dir data/samples \
        --ground-truth data/samples/ground_truth.json \
        --output data/processed/ocr_comparison.json \
        --markdown data/processed/ocr_comparison.md

By default it runs Tesseract and the tuned ``paddleocr-fast`` profile. Add
``paddleocr`` to ``--backends`` to include the slow PP-OCRv6 medium model.
Photos under ``data/samples/external/`` are picked up automatically.

The output files live under ``data/processed/`` and are gitignored.
"""

from __future__ import annotations

from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
import argparse
import json
import sys
import time

from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.appliance_lookup import normalize_alphanum  # noqa: E402
from scripts.label_ocr import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    OCRUnavailableError,
    PreprocessOptions,
    TesseractBackend,
    _load_image,
    find_images,
    preprocess_image_with_steps,
)

FIELD_KEYS = ("brand", "model_number", "serial_number")

VARIANTS: dict[str, PreprocessOptions | None] = {
    "raw": None,
    "gray": PreprocessOptions(grayscale=True, autocontrast=True, upscale=1.0, sharpen=False),
    "pipeline": PreprocessOptions(),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def best_window_similarity(expected: str, text: str, pad: int = 2) -> float:
    """Best ``SequenceMatcher`` ratio between ``expected`` and a window of ``text``.

    OCR splits and merges characters, so a strict substring test is too brittle.
    Sliding windows of ``len(expected) ± pad`` capture near-misses such as
    ``NXAGB3OGKA1OO`` vs ``NXAGB3OGKA100``.
    """
    if not expected or not text:
        return 0.0
    target_length = len(expected)
    best = 0.0
    for size in range(max(1, target_length - pad), target_length + pad + 1):
        if size > len(text):
            continue
        for start in range(0, len(text) - size + 1):
            ratio = SequenceMatcher(None, expected, text[start : start + size]).ratio()
            if ratio > best:
                best = ratio
                if best == 1.0:
                    return 1.0
    return round(best, 3)


def token_metrics(raw_text: str, ground_truth: dict[str, Any]) -> dict[str, Any]:
    """Score the expected field tokens against the raw OCR text."""
    normalized = normalize_alphanum(raw_text)
    per_field: dict[str, Any] = {}
    for field in FIELD_KEYS:
        value = ground_truth.get(field)
        if not value:
            continue
        target = normalize_alphanum(str(value))
        if not target:
            continue
        per_field[field] = {
            "expected": str(value),
            "normalized": target,
            "exact": target in normalized,
            "similarity": best_window_similarity(target, normalized),
        }

    if not per_field:
        return {"fields": {}, "exact_recall": None, "mean_similarity": None}

    exact_recall = sum(1 for item in per_field.values() if item["exact"]) / len(per_field)
    mean_similarity = sum(item["similarity"] for item in per_field.values()) / len(per_field)
    return {
        "fields": per_field,
        "exact_recall": round(exact_recall, 3),
        "mean_similarity": round(mean_similarity, 3),
    }


# ---------------------------------------------------------------------------
# Engine runners
# ---------------------------------------------------------------------------


def _prepare(image: Image.Image, options: PreprocessOptions | None) -> tuple[Image.Image, list[str]]:
    if options is None:
        return image.convert("RGB"), []
    return preprocess_image_with_steps(image, options)


def _run_backend(backend: Any, image: Image.Image) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        text, confidence = backend.extract(image)
    except Exception as exc:  # noqa: BLE001 - one engine must not abort the run
        return {"error": f"{type(exc).__name__}: {exc}", "latency_s": round(time.perf_counter() - started, 3)}
    return {
        "raw_text": text,
        "confidence": confidence,
        "latency_s": round(time.perf_counter() - started, 3),
        "word_count": len(text.split()),
    }


def _build_backends(names: list[str]) -> tuple[dict[str, Any], dict[str, str]]:
    backends: dict[str, Any] = {}
    errors: dict[str, str] = {}

    if "tesseract" in names:
        try:
            started = time.perf_counter()
            backends["tesseract"] = TesseractBackend()
            print(f"tesseract ready in {time.perf_counter() - started:.2f}s")
        except OCRUnavailableError as exc:
            errors["tesseract"] = str(exc)

    paddle_names = [name for name in names if name.startswith("paddleocr")]
    if paddle_names:
        try:
            from scripts.label_ocr_paddle import PaddleOCRBackend, fast_profile_kwargs
        except Exception as exc:  # noqa: BLE001
            for name in paddle_names:
                errors[name] = f"{type(exc).__name__}: {exc}"
            return backends, errors

        # Default tier: PP-OCRv6 medium (most accurate, slow on CPU).
        # Fast tier: tuned PP-OCRv5 mobile (roughly 25x faster).
        factories = {
            "paddleocr": lambda: PaddleOCRBackend(),
            "paddleocr-fast": lambda: PaddleOCRBackend(**fast_profile_kwargs()),
        }
        for name in paddle_names:
            factory = factories.get(name)
            if factory is None:
                errors[name] = f"unknown engine: {name}"
                continue
            try:
                started = time.perf_counter()
                backends[name] = factory()
                print(f"{name} ready in {time.perf_counter() - started:.2f}s")
            except OCRUnavailableError as exc:
                errors[name] = str(exc)
            except Exception as exc:  # noqa: BLE001 - model download / runtime issues
                errors[name] = f"{type(exc).__name__}: {exc}"

    return backends, errors


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _summarize(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_entry in payload["images"]:
        for variant, engines in image_entry["variants"].items():
            for engine, result in engines.items():
                metrics = result.get("metrics") or {}
                rows.append(
                    {
                        "image": image_entry["image"],
                        "variant": variant,
                        "engine": engine,
                        "exact_recall": metrics.get("exact_recall"),
                        "mean_similarity": metrics.get("mean_similarity"),
                        "confidence": result.get("confidence"),
                        "latency_s": result.get("latency_s"),
                        "error": result.get("error"),
                    }
                )
    return rows


def _to_markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = ["# OCR comparison: Tesseract vs PaddleOCR", ""]
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append("")
    if payload.get("backend_errors"):
        lines.append("## Backend errors")
        for engine, message in payload["backend_errors"].items():
            lines.append(f"- **{engine}**: {message}")
        lines.append("")

    lines.append("## Per-image results")
    lines.append("")
    lines.append("| Image | Variant | Engine | Exact recall | Mean similarity | Confidence | Latency (s) |")
    lines.append("|---|---|---|---|---|---|---|")

    def fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    for row in rows:
        lines.append(
            "| {image} | {variant} | {engine} | {recall} | {similarity} | {confidence} | {latency} |".format(
                image=row["image"],
                variant=row["variant"],
                engine=row["engine"],
                recall=fmt(row["exact_recall"]),
                similarity=fmt(row["mean_similarity"]),
                confidence=fmt(row["confidence"]),
                latency=fmt(row["latency_s"], 2),
            )
        )

    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Engine | Variant | Mean exact recall | Mean similarity | Mean latency (s) |")
    lines.append("|---|---|---|---|---|")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["error"] or row["exact_recall"] is None:
            continue
        grouped.setdefault((row["engine"], row["variant"]), []).append(row)

    for (engine, variant), items in sorted(grouped.items()):
        recall = sum(item["exact_recall"] for item in items) / len(items)
        similarity = sum(item["mean_similarity"] for item in items) / len(items)
        latency = sum(item["latency_s"] or 0.0 for item in items) / len(items)
        lines.append(f"| {engine} | {variant} | {recall:.3f} | {similarity:.3f} | {latency:.2f} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Tesseract and PaddleOCR on label photos")
    parser.add_argument("--dir", default="data/samples", help="Directory of label images")
    parser.add_argument("--ground-truth", default="data/samples/ground_truth.json", help="Ground-truth JSON")
    parser.add_argument("--output", default=str(DEFAULT_PROCESSED_DIR / "ocr_comparison.json"))
    parser.add_argument("--markdown", default=str(DEFAULT_PROCESSED_DIR / "ocr_comparison.md"))
    parser.add_argument("--backends", default="tesseract,paddleocr-fast", help="Comma-separated engine list")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="Comma-separated preprocessing variants")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images")
    args = parser.parse_args(argv)

    images = find_images(args.dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"No supported images found in {args.dir}", file=sys.stderr)
        return 1

    ground_truth: dict[str, Any] = {}
    truth_path = Path(args.ground_truth)
    if truth_path.is_file():
        ground_truth = json.loads(truth_path.read_text(encoding="utf-8"))
    else:
        print(f"Warning: ground truth not found at {truth_path}; metrics will be empty", file=sys.stderr)

    backend_names = [name.strip() for name in args.backends.split(",") if name.strip()]
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in variant_names if name not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants: {', '.join(unknown)}")

    backends, backend_errors = _build_backends(backend_names)
    for engine, message in backend_errors.items():
        print(f"Warning: {engine} unavailable: {message}", file=sys.stderr)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "directory": str(Path(args.dir)),
        "ground_truth_file": str(truth_path) if truth_path.is_file() else None,
        "backend_errors": backend_errors,
        "images": [],
    }

    for index, path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {path.name}")
        image = _load_image(path)
        truth = ground_truth.get(path.name, {})
        entry: dict[str, Any] = {"image": path.name, "ground_truth": truth, "variants": {}}

        for variant in variant_names:
            prepared, steps = _prepare(image, VARIANTS[variant])
            engine_results: dict[str, Any] = {}
            for engine, backend in backends.items():
                result = _run_backend(backend, prepared)
                result["preprocessing"] = steps
                if "raw_text" in result:
                    result["metrics"] = token_metrics(result["raw_text"], truth)
                engine_results[engine] = result
            entry["variants"][variant] = engine_results
            entry.setdefault("preprocessed_size", {})[variant] = list(prepared.size)

        payload["images"].append(entry)

    rows = _summarize(payload)
    payload["summary"] = rows

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")

    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_to_markdown(payload, rows), encoding="utf-8")
    print(f"Wrote {markdown_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
