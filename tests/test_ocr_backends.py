"""Tests for OCR engine selection (``scripts/ocr_backends.py``)."""

from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from scripts import ocr_backends
from scripts.label_ocr import OCRUnavailableError, PreprocessOptions


class _FakeBackend:
    def __init__(self, name: str) -> None:
        self.name = name

    def extract(self, image):  # pragma: no cover - protocol stub
        return "", None


class TestDefaultOptions(unittest.TestCase):
    def test_paddle_uses_grayscale_without_resize(self):
        options = ocr_backends.default_options_for(_FakeBackend("paddleocr"))

        self.assertTrue(options.grayscale)
        self.assertTrue(options.autocontrast)
        self.assertEqual(options.upscale, 1.0)
        self.assertFalse(options.sharpen)

    def test_tesseract_keeps_existing_defaults(self):
        options = ocr_backends.default_options_for(_FakeBackend("tesseract"))

        self.assertEqual(options, PreprocessOptions())

    def test_unknown_backend_uses_tesseract_defaults(self):
        options = ocr_backends.default_options_for(_FakeBackend("other"))

        self.assertEqual(options, PreprocessOptions())


class TestGetBackend(unittest.TestCase):
    def setUp(self):
        ocr_backends._fallback_warned = False

    def test_auto_prefers_paddle(self):
        fake = _FakeBackend("paddleocr")
        with mock.patch.object(ocr_backends, "_build_paddle", return_value=fake) as build:
            result = ocr_backends.get_backend("auto")

        self.assertIs(result, fake)
        build.assert_called_once()

    def test_auto_falls_back_to_tesseract(self):
        fake = _FakeBackend("tesseract")
        stderr = StringIO()
        with mock.patch.object(
            ocr_backends, "_build_paddle", side_effect=OCRUnavailableError("no paddle")
        ), mock.patch.object(ocr_backends, "_build_tesseract", return_value=fake), redirect_stderr(stderr):
            result = ocr_backends.get_backend("auto")

        self.assertIs(result, fake)
        self.assertIn("falling back to Tesseract", stderr.getvalue())

    def test_explicit_paddle_does_not_fall_back(self):
        with mock.patch.object(
            ocr_backends, "_build_paddle", side_effect=OCRUnavailableError("no paddle")
        ), mock.patch.object(ocr_backends, "_build_tesseract") as build_tesseract:
            with self.assertRaises(OCRUnavailableError):
                ocr_backends.get_backend("paddleocr")

        build_tesseract.assert_not_called()

    def test_explicit_tesseract_forwards_language_and_psm(self):
        fake = _FakeBackend("tesseract")
        with mock.patch.object(ocr_backends, "_build_tesseract", return_value=fake) as build:
            result = ocr_backends.get_backend("tesseract", lang="eng", psm=6)

        self.assertIs(result, fake)
        build.assert_called_once_with(lang="eng", psm=6)

    def test_paddle_alias_is_accepted(self):
        fake = _FakeBackend("paddleocr")
        with mock.patch.object(ocr_backends, "_build_paddle", return_value=fake):
            self.assertIs(ocr_backends.get_backend("paddle"), fake)

    def test_unknown_engine_raises(self):
        with self.assertRaises(ValueError):
            ocr_backends.get_backend("magic")


class TestExtractTextWiring(unittest.TestCase):
    def test_extract_text_uses_selected_default_backend(self):
        from scripts.label_ocr import extract_text

        fake = _FakeBackend("paddleocr")
        fake.extract = mock.Mock(return_value=("ACME MODEL ABC-123", 0.9))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label.png"
            Image.new("RGB", (80, 40), "white").save(path)
            with mock.patch.object(ocr_backends, "get_backend", return_value=fake) as get:
                result = extract_text(path)

        self.assertEqual(result["backend"], "paddleocr")
        self.assertIn("ACME", result["raw_text"])
        self.assertEqual(result["confidence"], 0.9)
        get.assert_called_once_with()


class TestBuildPaddle(unittest.TestCase):
    def test_fast_profile_is_applied_by_default(self):
        fake_class = mock.Mock(return_value=_FakeBackend("paddleocr"))
        with mock.patch("scripts.label_ocr_paddle.PaddleOCRBackend", fake_class), mock.patch(
            "scripts.label_ocr_paddle.fast_profile_kwargs",
            return_value={"ocr_version": "PP-OCRv5", "cpu_threads": 4},
        ) as fast_profile:
            ocr_backends._build_paddle(
                lang=None, device=None, enable_mkldnn=True, accurate=False
            )

        fast_profile.assert_called_once()
        kwargs = fake_class.call_args.kwargs
        self.assertEqual(kwargs["ocr_version"], "PP-OCRv5")
        self.assertEqual(kwargs["lang"], "en")

    def test_accurate_skips_fast_profile(self):
        fake_class = mock.Mock(return_value=_FakeBackend("paddleocr"))
        with mock.patch("scripts.label_ocr_paddle.PaddleOCRBackend", fake_class), mock.patch(
            "scripts.label_ocr_paddle.fast_profile_kwargs"
        ) as fast_profile:
            ocr_backends._build_paddle(
                lang="en", device=None, enable_mkldnn=True, accurate=True
            )

        fast_profile.assert_not_called()
        self.assertNotIn("ocr_version", fake_class.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
