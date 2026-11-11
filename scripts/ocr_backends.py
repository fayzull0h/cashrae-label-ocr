"""OCR backend selection and per-engine preprocessing defaults.

PaddleOCR's tuned fast profile is the default engine. Tesseract remains a
first-class fallback and is used automatically when Paddle is not installed or
cannot be initialised.

Keeping the factory in its own module avoids a circular import:
``label_ocr_paddle`` imports from ``label_ocr``, so ``label_ocr`` only imports
this module lazily inside its high-level helpers.

Typical usage::

    from scripts.ocr_backends import get_backend

    backend = get_backend("auto")          # Paddle if available, else Tesseract
    backend = get_backend("tesseract")     # force Tesseract
    backend = get_backend("paddleocr")     # force Paddle (raises if unavailable)
"""

from __future__ import annotations

from typing import Any
import sys

from scripts.label_ocr import (
    OCRBackend,
    PreprocessOptions,
    SUPPORTED_OCR_ENGINES,
    TesseractBackend,
)

PADDLE_LANG = "en"
TESSERACT_LANG = "eng"

_fallback_warned = False


def _warn_fallback(exc: Exception) -> None:
    """Print a single, process-wide fallback warning."""
    global _fallback_warned
    if _fallback_warned:
        return
    _fallback_warned = True
    print(
        f"PaddleOCR unavailable ({type(exc).__name__}: {exc}); "
        "falling back to Tesseract.",
        file=sys.stderr,
    )


def paddle_options() -> PreprocessOptions:
    """Preprocessing that helped PaddleOCR on real rating plates.

    Grayscale + autocontrast only. Paddle performs its own detection and caps
    the detection side at 960 px, so resizing here adds cost without benefit.
    """
    return PreprocessOptions(
        grayscale=True,
        autocontrast=True,
        upscale=1.0,
        sharpen=False,
    )


def tesseract_options() -> PreprocessOptions:
    """Current Tesseract defaults (size-aware resize, no sharpening)."""
    return PreprocessOptions()


def default_options_for(backend: OCRBackend) -> PreprocessOptions:
    """Return the preprocessing defaults for ``backend``."""
    if getattr(backend, "name", "") == "paddleocr":
        return paddle_options()
    return tesseract_options()


def _build_paddle(
    *,
    lang: str | None,
    device: str | None,
    enable_mkldnn: bool,
    accurate: bool,
) -> OCRBackend:
    from scripts.label_ocr_paddle import PaddleOCRBackend, fast_profile_kwargs

    kwargs: dict[str, Any] = {}
    if not accurate:
        kwargs.update(fast_profile_kwargs())
    return PaddleOCRBackend(
        lang=lang or PADDLE_LANG,
        device=device,
        enable_mkldnn=enable_mkldnn,
        **kwargs,
    )


def _build_tesseract(*, lang: str | None, psm: int) -> OCRBackend:
    return TesseractBackend(lang=lang or TESSERACT_LANG, psm=psm)


def get_backend(
    name: str = "auto",
    *,
    lang: str | None = None,
    psm: int = 3,
    device: str | None = None,
    enable_mkldnn: bool = True,
    accurate: bool = False,
) -> OCRBackend:
    """Build an OCR backend by name.

    ``auto`` prefers PaddleOCR and falls back to Tesseract with a warning when
    Paddle cannot be imported or initialised. ``paddleocr`` and ``tesseract``
    force a specific engine and raise instead of falling back.
    """
    normalized = (name or "auto").strip().lower()

    if normalized in {"paddle", "paddleocr"}:
        return _build_paddle(
            lang=lang,
            device=device,
            enable_mkldnn=enable_mkldnn,
            accurate=accurate,
        )
    if normalized == "tesseract":
        return _build_tesseract(lang=lang, psm=psm)
    if normalized != "auto":
        raise ValueError(
            f"Unknown OCR engine: {name!r}. "
            f"Choose one of: {', '.join(SUPPORTED_OCR_ENGINES)}."
        )

    try:
        return _build_paddle(
            lang=lang,
            device=device,
            enable_mkldnn=enable_mkldnn,
            accurate=accurate,
        )
    except Exception as exc:  # noqa: BLE001 - a broken Paddle install must not block OCR
        _warn_fallback(exc)
        return _build_tesseract(lang=lang, psm=psm)
