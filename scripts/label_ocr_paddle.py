"""PaddleOCR backend for appliance labels (default OCR engine).

This module implements the project's default :class:`~scripts.label_ocr.OCRBackend`.
Tesseract remains available as a fallback and for comparison; see
``scripts/ocr_backends.py`` for engine selection.

``paddleocr`` is imported lazily: this module imports cleanly in environments
where Paddle is not installed and raises
:class:`~scripts.label_ocr.OCRUnavailableError` only when the backend is actually
constructed.

Typical usage::

    from scripts.label_ocr import extract_text
    from scripts.label_ocr_paddle import PaddleOCRBackend

    result = extract_text("data/samples/label.jpg", backend=PaddleOCRBackend())

CLI::

    python scripts/label_ocr_paddle.py --image data/samples/label.jpg
    python scripts/label_ocr_paddle.py --dir data/samples --json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence
import argparse
import json
import os
import sys

from PIL import Image

# Allow direct execution as ``python scripts/label_ocr_paddle.py``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.label_ocr import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    OCRUnavailableError,
    PreprocessOptions,
    extract_directory,
    extract_text,
)

PADDLE_INSTALL_HINT = (
    "PaddleOCR is not installed in this environment. Install the OCR extras:\n"
    "  .venv/Scripts/python.exe -m pip install -r requirements-paddle.txt\n"
    "PaddleOCR is the default engine; without it, label_ocr.py falls back to Tesseract."
)


def _import_paddleocr():
    """Import ``PaddleOCR`` lazily and translate failures into a useful error."""
    try:
        from paddleocr import PaddleOCR  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise OCRUnavailableError(PADDLE_INSTALL_HINT) from exc
    return PaddleOCR


def paddle_available() -> bool:
    """True when both ``paddleocr`` and ``paddle`` can be imported."""
    try:
        _import_paddleocr()
        import paddle  # type: ignore[import-not-found]  # noqa: F401
    except Exception:  # noqa: BLE001 - any import-time failure means unavailable
        return False
    return True


def fast_profile_kwargs() -> dict[str, Any]:
    """Tuned PP-OCRv5 mobile settings for CPU inference.

    Measured on a 22-thread Windows CPU: ~2.9 s per 4284x5712 label photo with
    100% field recall, versus ~74 s for the default PP-OCRv6 medium models.
    ``cpu_threads`` defaults to the core count, capped at 16, and can be
    overridden with ``PADDLE_PDX_CPU_NUM_THREADS`` for low-core machines.
    """
    configured = os.environ.get("PADDLE_PDX_CPU_NUM_THREADS")
    try:
        threads = int(configured) if configured else (os.cpu_count() or 8)
    except ValueError:
        threads = os.cpu_count() or 8
    return {
        "ocr_version": "PP-OCRv5",
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_recognition_model_name": "PP-OCRv5_mobile_rec",
        "cpu_threads": max(1, min(threads, 16)),
        "text_recognition_batch_size": 32,
        "text_det_limit_side_len": 960,
        "text_det_limit_type": "max",
    }


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _result_get(result: Any, key: str, default: Any = None) -> Any:
    """Read a key from a PaddleX result object or plain dict."""
    try:
        value = result[key]
    except (KeyError, TypeError, IndexError):
        value = getattr(result, key, default)
    return default if value is None else value


def _box_bounds(box: Sequence[Any]) -> tuple[float, float, float, float]:
    """Return ``(x1, y1, x2, y2)`` for either an AABB or a polygon."""
    # AABB: [x1, y1, x2, y2] — the first element is a scalar, not a point.
    if len(box) == 4 and not hasattr(box[0], "__len__"):
        x1, y1, x2, y2 = (float(value) for value in box)
        return x1, y1, x2, y2

    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _ordered_entries(result: Any) -> list[tuple[float, float, str, float]]:
    """Return ``(line_key, x, text, score)`` tuples in reading order."""
    texts = [str(text) for text in (_result_get(result, "rec_texts") or [])]
    scores = [float(score) for score in (_result_get(result, "rec_scores") or [])]
    if len(scores) < len(texts):
        scores.extend([0.0] * (len(texts) - len(scores)))

    boxes = _result_get(result, "rec_boxes")
    if boxes is None:
        boxes = _result_get(result, "rec_polys")

    if boxes is None or len(boxes) != len(texts):
        # No usable geometry: keep the engine's own order.
        return [(float(index), 0.0, text, scores[index]) for index, text in enumerate(texts)]

    bounds = [_box_bounds(box) for box in boxes]
    heights = [max(y2 - y1, 1.0) for _, y1, _, y2 in bounds]
    line_height = max(_median(heights), 1.0)

    entries: list[tuple[float, float, str, float]] = []
    for (x1, y1, _, _), text, score in zip(bounds, texts, scores):
        # Bucket by vertical position so words on the same visual line group together.
        line_key = round(y1 / (line_height * 0.6))
        entries.append((float(line_key), x1, text, score))
    return sorted(entries, key=lambda entry: (entry[0], entry[1]))


def _merge_results(results: Iterable[Any]) -> tuple[str, float | None]:
    """Flatten PaddleX results into ``(line-separated text, mean confidence)``."""
    lines: list[str] = []
    scores: list[float] = []
    for result in results or []:
        for _, _, text, score in _ordered_entries(result):
            cleaned = text.strip()
            if cleaned:
                lines.append(cleaned)
                scores.append(score)
    confidence = round(sum(scores) / len(scores), 4) if scores else None
    return "\n".join(lines), confidence


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class PaddleOCRBackend:
    """PaddleOCR engine behind the project's ``OCRBackend`` protocol."""

    name = "paddleocr"

    def __init__(
        self,
        lang: str = "en",
        device: str | None = None,
        enable_mkldnn: bool = True,
        ocr_version: str | None = None,
        text_recognition_batch_size: int | None = None,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        **extra_kwargs: Any,
    ) -> None:
        PaddleOCR = _import_paddleocr()
        kwargs: dict[str, Any] = {
            "lang": lang,
            # Requires paddlepaddle==3.2.2 on Windows: 3.3.1 crashes in the
            # oneDNN/PIR executor (ConvertPirAttribute2RuntimeAttribute).
            "enable_mkldnn": enable_mkldnn,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_textline_orientation": use_textline_orientation,
        }
        if device:
            kwargs["device"] = device
        if ocr_version:
            kwargs["ocr_version"] = ocr_version
        if text_recognition_batch_size:
            kwargs["text_recognition_batch_size"] = text_recognition_batch_size
        kwargs.update(extra_kwargs)
        self.lang = lang
        self.device = device
        self.enable_mkldnn = enable_mkldnn
        self.ocr_version = ocr_version
        self._ocr = PaddleOCR(**kwargs)

    def extract(self, image: Image.Image) -> tuple[str, float | None]:
        """Run PaddleOCR on a Pillow image and return text plus 0–1 confidence."""
        import numpy as np  # bundled with paddleocr

        rgb = np.asarray(image.convert("RGB"))
        # PaddleOCR's array input follows OpenCV's BGR convention.
        bgr = rgb[:, :, ::-1].copy()
        results = self._ocr.predict(bgr)
        return _merge_results(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_result(result: dict[str, Any]) -> None:
    if "error" in result:
        print(f"Image:  {result['image_path']}")
        print(f"Error:  {result['error']}\n")
        return

    confidence = result.get("confidence")
    confidence_text = "n/a" if confidence is None else f"{confidence:.2f}"
    print("=" * 65)
    print("Appliance Label OCR (PaddleOCR)")
    print("=" * 65)
    print(f"Image:         {result['image_path']}")
    print(f"Backend:       {result['backend']}")
    print(f"Confidence:    {confidence_text}")
    print(f"Preprocessing: {', '.join(result.get('preprocessing') or []) or '(none)'}")
    print(f"Words:         {result['word_count']}")
    print("\n--- Raw text ---")
    print(result.get("raw_text") or "(no text detected)")
    print()


def main(argv: list[str] | None = None) -> int:
    # See scripts/label_ocr.py: keep non-ASCII OCR text from crashing on cp1252.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Extract raw text with PaddleOCR")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--dir", help="Directory of images to process (batch mode)")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report")
    parser.add_argument("--preprocess", action="store_true", help="Apply grayscale/autocontrast preprocessing (off by default here)")
    parser.add_argument("--accurate", action="store_true", help="Use PP-OCRv6 medium (experimental; ~25x slower on CPU)")
    parser.add_argument("--no-mkldnn", action="store_true", help="Disable oneDNN acceleration (slower; use with paddlepaddle 3.3.x)")
    parser.add_argument("--debug", action="store_true", help="Save preprocessed images to data/processed/")
    parser.add_argument("--lang", default="en", help="PaddleOCR language code (default: en)")
    parser.add_argument("--device", help="Paddle device, e.g. cpu or gpu:0 (default: auto)")
    parser.add_argument("--output", help="Write JSON results to this file")
    args = parser.parse_args(argv)

    if not args.image and not args.dir:
        parser.error("provide --image or --dir")

    debug_dir = DEFAULT_PROCESSED_DIR if args.debug else None

    try:
        backend = PaddleOCRBackend(
            lang=args.lang,
            device=args.device,
            enable_mkldnn=not args.no_mkldnn,
            **({} if args.accurate else fast_profile_kwargs()),
        )
        if args.dir:
            results = extract_directory(
                args.dir,
                backend=backend,
                preprocess=args.preprocess,
                debug_dir=debug_dir,
            )
        else:
            results = [
                extract_text(
                    args.image,
                    backend=backend,
                    preprocess=args.preprocess,
                    debug_dir=debug_dir,
                )
            ]
    except OCRUnavailableError as exc:
        print(f"OCR unavailable: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload: Any = results if args.dir else results[0]

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {output_path}")

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if not results:
            print(f"No supported images found in {args.dir}")
        for result in results:
            _print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
