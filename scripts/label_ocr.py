"""Appliance label OCR foundation (Phase 2).

Extracts raw text from appliance rating-plate photographs using PaddleOCR by
default, with automatic fallback to Tesseract, plus Pillow-based preprocessing
and a small pluggable backend interface.

The output is intentionally *raw*: this module performs no field parsing.
Parsing lives in ``scripts/label_parse.py`` (a later stage) so the OCR layer
stays swappable (e.g. for a vision-language model backend).

Typical usage::

    from scripts.label_ocr import extract_text

    result = extract_text("data/samples/fridge.jpg")
    print(result["raw_text"])

CLI::

    python scripts/label_ocr.py --image data/samples/fridge.jpg
    python scripts/label_ocr.py --dir data/samples --json --engine tesseract
    python scripts/label_ocr.py --image data/samples/fridge.jpg --parse
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
import argparse
import json
import os
import shutil
import sys

from PIL import Image, ImageFilter, ImageOps

try:  # pragma: no cover - import guard exercised only when dependency missing
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Allow direct execution as ``python scripts/label_ocr.py``.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

#: Engine names accepted by the CLI and :func:`scripts.ocr_backends.get_backend`.
SUPPORTED_OCR_ENGINES = ("auto", "paddleocr", "tesseract")

# Common install locations on Windows (the project's primary platform).
WINDOWS_DEFAULT_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")),
]

TESSERACT_INSTALL_HINT = (
    "Install Tesseract and retry:\n"
    "  winget install --id UB-Mannheim.TesseractOCR -e\n"
    "Then either restart your shell so PATH updates, or set the TESSERACT_CMD "
    "environment variable to the full path of tesseract.exe."
)


class OCRUnavailableError(RuntimeError):
    """Raised when the OCR engine (binary or Python wrapper) cannot be used."""


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


@dataclass
class PreprocessOptions:
    """Controls the Pillow preprocessing pipeline."""

    grayscale: bool = True
    autocontrast: bool = True
    upscale: float = 2.0
    upscale_threshold: int = 1600
    max_dimension: int = 5000
    sharpen: bool = False
    median_filter: int = 0
    threshold: int | None = None

    def describe(self) -> list[str]:
        """Return a human-readable summary of the intended pipeline."""
        steps: list[str] = []
        if self.grayscale or self.threshold is not None:
            steps.append("grayscale")
        if self.autocontrast:
            steps.append("autocontrast")
        if self.median_filter:
            steps.append(f"median filter {self.median_filter}")
        if self.upscale and self.upscale != 1:
            steps.append(
                f"resize (upscale x{self.upscale:g} below {self.upscale_threshold}px, "
                f"max {self.max_dimension}px)"
            )
        if self.sharpen:
            steps.append("sharpen")
        if self.threshold is not None:
            steps.append(f"threshold {self.threshold}")
        return steps


def _load_image(source: str | Path | Image.Image) -> Image.Image:
    """Load an image from a path or pass through an existing Pillow image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as img:
        return img.convert("RGB")


def preprocess_image(
    source: str | Path | Image.Image,
    options: PreprocessOptions | None = None,
) -> Image.Image:
    """Apply the configured preprocessing pipeline and return a Pillow image."""
    image, _ = preprocess_image_with_steps(source, options)
    return image


def preprocess_image_with_steps(
    source: str | Path | Image.Image,
    options: PreprocessOptions | None = None,
) -> tuple[Image.Image, list[str]]:
    """Apply preprocessing and return the image plus the steps actually performed.

    Resizing is size-aware: small images are upscaled for readability, while very
    large photos are downscaled to ``max_dimension`` to keep OCR fast.
    """
    opts = options or PreprocessOptions()
    image = _load_image(source)
    steps: list[str] = []

    if opts.grayscale or opts.threshold is not None:
        image = image.convert("L")
        steps.append("grayscale")
    if opts.autocontrast:
        image = ImageOps.autocontrast(image)
        steps.append("autocontrast")
    if opts.median_filter and opts.median_filter > 1:
        image = image.filter(ImageFilter.MedianFilter(size=opts.median_filter))
        steps.append(f"median filter {opts.median_filter}")

    width, height = image.size
    longest = max(width, height)
    scale = 1.0
    if opts.upscale and opts.upscale != 1 and longest < opts.upscale_threshold:
        scale = opts.upscale
    if longest * scale > opts.max_dimension:
        scale = opts.max_dimension / longest
    if abs(scale - 1.0) > 1e-3:
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        steps.append(f"resize x{scale:.2f} ({new_size[0]}x{new_size[1]})")

    if opts.sharpen:
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        steps.append("sharpen")
    if opts.threshold is not None:
        image = image.point(lambda px: 255 if px > opts.threshold else 0)
        steps.append(f"threshold {opts.threshold}")

    return image, steps


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------


@dataclass
class OCRResult:
    """Raw OCR output plus provenance about how it was produced."""

    image_path: str
    raw_text: str
    confidence: float | None
    backend: str
    preprocessing: list[str] = field(default_factory=list)
    word_count: int = 0
    preprocessed_image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "image_path": self.image_path,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "backend": self.backend,
            "preprocessing": list(self.preprocessing),
            "word_count": self.word_count,
        }
        if self.preprocessed_image:
            data["preprocessed_image"] = self.preprocessed_image
        return data


class OCRBackend(Protocol):
    """Minimal interface so OCR engines can be swapped without touching callers."""

    name: str

    def extract(self, image: Image.Image) -> tuple[str, float | None]:
        """Return ``(raw_text, confidence_0_to_1)`` for a Pillow image."""
        ...


def resolve_tesseract() -> str:
    """Locate the Tesseract executable.

    Checks ``TESSERACT_CMD`` first, then ``PATH``, then common Windows install
    locations. Raises :class:`OCRUnavailableError` with install instructions
    when nothing is found.
    """
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).is_file():
        return env_cmd

    found = shutil.which("tesseract")
    if found:
        return found

    for candidate in WINDOWS_DEFAULT_PATHS:
        if candidate.is_file():
            return str(candidate)

    raise OCRUnavailableError(f"Tesseract executable not found.\n{TESSERACT_INSTALL_HINT}")


def _text_from_data(data: dict[str, list[Any]]) -> str:
    """Reconstruct line-separated text from ``pytesseract.image_to_data`` output."""
    lines: dict[tuple[int, int, int], list[str]] = {}
    texts = data.get("text", [])
    for index, word in enumerate(texts):
        cleaned = (word or "").strip()
        if not cleaned:
            continue
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        lines.setdefault(key, []).append(cleaned)
    return "\n".join(" ".join(words) for _, words in sorted(lines.items()))


def _confidence_from_data(data: dict[str, list[Any]]) -> float | None:
    """Average per-word Tesseract confidence, normalised to 0.0–1.0."""
    values: list[float] = []
    for raw in data.get("conf", []):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            values.append(value)
    if not values:
        return None
    return round(sum(values) / len(values) / 100.0, 4)


class TesseractBackend:
    """Tesseract OCR backend using the ``pytesseract`` wrapper."""

    name = "tesseract"

    def __init__(
        self,
        tesseract_cmd: str | None = None,
        lang: str = "eng",
        psm: int = 3,
        oem: int = 3,
    ) -> None:
        if pytesseract is None:
            raise OCRUnavailableError(
                "pytesseract is not installed. Run:\n  pip install pillow pytesseract"
            )
        self.tesseract_cmd = tesseract_cmd or resolve_tesseract()
        self.lang = lang
        self.psm = psm
        self.oem = oem
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def extract(self, image: Image.Image) -> tuple[str, float | None]:
        config = f"--oem {self.oem} --psm {self.psm}"
        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        return _text_from_data(data).strip(), _confidence_from_data(data)


# ---------------------------------------------------------------------------
# High-level extraction helpers
# ---------------------------------------------------------------------------


def extract_text(
    image_path: str | Path,
    backend: OCRBackend | None = None,
    options: PreprocessOptions | None = None,
    preprocess: bool = True,
    debug_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Extract raw text from a single image and return a structured dict."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if backend is None or options is None:
        # Imported lazily: ocr_backends imports this module at the top level.
        from scripts.ocr_backends import default_options_for, get_backend

    engine = backend or get_backend()
    opts = options or default_options_for(engine)

    image = _load_image(path)
    steps: list[str] = []
    if preprocess:
        image, steps = preprocess_image_with_steps(image, opts)

    text, confidence = engine.extract(image)
    result = OCRResult(
        image_path=str(path),
        raw_text=text,
        confidence=confidence,
        backend=getattr(engine, "name", type(engine).__name__),
        preprocessing=steps,
        word_count=len(text.split()),
    )

    if debug_dir is not None:
        output_dir = Path(debug_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{path.stem}_preprocessed.png"
        image.save(destination)
        result.preprocessed_image = str(destination)

    return result.to_dict()


def find_images(directory: str | Path) -> list[Path]:
    """Return supported image files under ``directory``, sorted by path."""
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def extract_directory(
    directory: str | Path,
    backend: OCRBackend | None = None,
    options: PreprocessOptions | None = None,
    preprocess: bool = True,
    debug_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run OCR over every supported image in a directory.

    Individual failures are captured as ``{"image_path": ..., "error": ...}``
    entries so one bad file does not abort a batch.
    """
    if backend is None:
        from scripts.ocr_backends import get_backend

    engine = backend or get_backend()
    results: list[dict[str, Any]] = []
    for path in find_images(directory):
        try:
            results.append(
                extract_text(
                    path,
                    backend=engine,
                    options=options,
                    preprocess=preprocess,
                    debug_dir=debug_dir,
                )
            )
        except Exception as exc:  # noqa: BLE001 - batch should not abort
            results.append({"image_path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return results


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
    steps = ", ".join(result.get("preprocessing") or []) or "(none)"

    print("=" * 65)
    print("Appliance Label OCR")
    print("=" * 65)
    print(f"Image:         {result['image_path']}")
    print(f"Backend:       {result['backend']}")
    print(f"Confidence:    {confidence_text}")
    print(f"Preprocessing: {steps}")
    print(f"Words:         {result['word_count']}")
    if result.get("preprocessed_image"):
        print(f"Debug image:   {result['preprocessed_image']}")

    parsed = result.get("parsed")
    if parsed:
        print("\n--- Parsed fields ---")
        for key in ("brand", "model_number", "serial_number"):
            field = parsed[key]
            value = field["value"] if field["value"] is not None else "(not found)"
            print(f"{key:14} {value}")
            if field["value"] is not None:
                print(
                    f"{'':14} confidence {field['confidence']:.2f} · "
                    f"{field['method']} · {field['source_line']!r}"
                )

    print("\n--- Raw text ---")
    print(result.get("raw_text") or "(no text detected)")
    print()


def main(argv: list[str] | None = None) -> int:
    # Redirected Windows stdout uses the locale codec (often cp1252), which cannot
    # encode non-ASCII OCR output. Force UTF-8 with replacement instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # StringIO in tests, or a wrapped stream
            pass

    parser = argparse.ArgumentParser(description="Extract raw text from appliance label images")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--dir", help="Directory of images to process (batch mode)")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report")
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Also parse brand/model/serial fields (Phase 2 Stage 2)",
    )
    parser.add_argument("--debug", action="store_true", help="Save preprocessed images to data/processed/")
    parser.add_argument("--no-preprocess", action="store_true", help="Send the original image to OCR")
    parser.add_argument(
        "--engine",
        choices=SUPPORTED_OCR_ENGINES,
        default="auto",
        help="OCR engine: auto (Paddle if available, else Tesseract), paddleocr, or tesseract",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Language code (default: en for PaddleOCR, eng for Tesseract)",
    )
    parser.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode (Tesseract only)")
    parser.add_argument("--device", help="Paddle device, e.g. cpu or gpu:0 (Paddle only; default: auto)")
    parser.add_argument(
        "--accurate",
        action="store_true",
        help="Use PP-OCRv6 medium instead of the fast PP-OCRv5 mobile profile (Paddle only; much slower)",
    )
    parser.add_argument("--output", help="Write JSON results to this file")
    args = parser.parse_args(argv)

    if not args.image and not args.dir:
        parser.error("provide --image or --dir")

    debug_dir = DEFAULT_PROCESSED_DIR if args.debug else None

    try:
        from scripts.ocr_backends import get_backend

        backend = get_backend(
            args.engine,
            lang=args.lang,
            psm=args.psm,
            device=args.device,
            accurate=args.accurate,
        )
        if args.dir:
            results = extract_directory(
                args.dir,
                backend=backend,
                preprocess=not args.no_preprocess,
                debug_dir=debug_dir,
            )
        else:
            results = [
                extract_text(
                    args.image,
                    backend=backend,
                    preprocess=not args.no_preprocess,
                    debug_dir=debug_dir,
                )
            ]
    except OCRUnavailableError as exc:
        print(f"OCR unavailable: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.parse:
        from scripts.label_parse import parse_ocr_result

        results = [
            {**result, "parsed": parse_ocr_result(result)} if "raw_text" in result else result
            for result in results
        ]

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
