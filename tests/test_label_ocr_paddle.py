"""Tests for the experimental PaddleOCR backend (``scripts/label_ocr_paddle.py``).

The parsing tests use plain dicts that mimic PaddleX result objects, so they run
without Paddle installed. The integration test is skipped unless the
``.venv-paddle`` environment is active.
"""

from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image, ImageDraw, ImageFont

from scripts import label_ocr_paddle
from scripts.label_ocr import OCRUnavailableError
from scripts.label_ocr_paddle import (
    PaddleOCRBackend,
    _box_bounds,
    _merge_results,
    _ordered_entries,
    _result_get,
    fast_profile_kwargs,
    paddle_available,
)


def _fake_result(texts, scores=None, boxes=None):
    result = {"rec_texts": list(texts)}
    if scores is not None:
        result["rec_scores"] = list(scores)
    if boxes is not None:
        result["rec_boxes"] = boxes
    return result


class TestResultHelpers(unittest.TestCase):
    def test_result_get_handles_dict_and_object(self):
        class Result:
            rec_texts = ["A"]

        self.assertEqual(_result_get({"rec_texts": ["B"]}, "rec_texts"), ["B"])
        self.assertEqual(_result_get(Result(), "rec_texts"), ["A"])
        self.assertIsNone(_result_get({}, "missing"))

    def test_box_bounds_supports_aabb_and_polygon(self):
        self.assertEqual(_box_bounds([10, 20, 30, 40]), (10.0, 20.0, 30.0, 40.0))
        polygon = [[10, 20], [30, 20], [30, 40], [10, 40]]
        self.assertEqual(_box_bounds(polygon), (10.0, 20.0, 30.0, 40.0))

    def test_merge_joins_lines_without_boxes(self):
        text, confidence = _merge_results([_fake_result(["ACME", "MODEL: ABC-123"], [0.9, 0.8])])

        self.assertEqual(text, "ACME\nMODEL: ABC-123")
        self.assertAlmostEqual(confidence, 0.85)

    def test_merge_orders_by_visual_position(self):
        result = _fake_result(
            ["SERIAL", "ACME", "12345"],
            [0.9, 0.95, 0.7],
            boxes=[
                [10, 100, 200, 130],
                [10, 10, 150, 40],
                [10, 140, 200, 170],
            ],
        )

        text, _ = _merge_results([result])

        self.assertEqual(text.splitlines(), ["ACME", "SERIAL", "12345"])

    def test_merge_skips_blank_text(self):
        text, confidence = _merge_results([_fake_result(["ACME", "  ", "MODEL"], [0.9, 0.0, 0.8])])

        self.assertEqual(text, "ACME\nMODEL")
        self.assertAlmostEqual(confidence, 0.85)

    def test_merge_pads_missing_scores(self):
        text, confidence = _merge_results([_fake_result(["A", "B"], [0.6])])

        self.assertEqual(text, "A\nB")
        self.assertAlmostEqual(confidence, 0.3)

    def test_empty_results(self):
        self.assertEqual(_merge_results([]), ("", None))

    def test_ordered_entries_keeps_engine_order_without_boxes(self):
        entries = _ordered_entries(_fake_result(["B", "A"], [0.5, 0.5]))

        self.assertEqual([entry[2] for entry in entries], ["B", "A"])


class TestBackendAvailability(unittest.TestCase):
    def test_fast_profile_uses_mobile_models(self):
        kwargs = fast_profile_kwargs()

        self.assertEqual(kwargs["ocr_version"], "PP-OCRv5")
        self.assertIn("mobile", kwargs["text_detection_model_name"])
        self.assertIn("mobile", kwargs["text_recognition_model_name"])
        self.assertGreaterEqual(kwargs["cpu_threads"], 1)
        self.assertLessEqual(kwargs["cpu_threads"], 16)

    def test_fast_profile_respects_thread_override(self):
        with mock.patch.dict(os.environ, {"PADDLE_PDX_CPU_NUM_THREADS": "4"}):
            self.assertEqual(fast_profile_kwargs()["cpu_threads"], 4)

    def test_fast_profile_ignores_invalid_thread_override(self):
        with mock.patch.dict(os.environ, {"PADDLE_PDX_CPU_NUM_THREADS": "many"}):
            kwargs = fast_profile_kwargs()
        self.assertGreaterEqual(kwargs["cpu_threads"], 1)
        self.assertLessEqual(kwargs["cpu_threads"], 16)

    def test_backend_raises_when_paddle_missing(self):
        with mock.patch.object(
            label_ocr_paddle,
            "_import_paddleocr",
            side_effect=OCRUnavailableError("missing"),
        ):
            with self.assertRaises(OCRUnavailableError):
                PaddleOCRBackend()

    def test_paddle_available_false_when_import_fails(self):
        with mock.patch.object(label_ocr_paddle, "_import_paddleocr", side_effect=ImportError("nope")):
            self.assertFalse(paddle_available())


def _font(size: int = 52):
    for candidate in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - older Pillow
        return ImageFont.load_default()


@unittest.skipUnless(paddle_available(), "PaddleOCR not installed in this environment")
class TestPaddleIntegration(unittest.TestCase):
    def test_extract_text_from_synthetic_label(self):
        from scripts.label_ocr import extract_text

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label.png"
            image = Image.new("RGB", (900, 320), "white")
            draw = ImageDraw.Draw(image)
            draw.text((40, 40), "ACME APPLIANCES", fill="black", font=_font())
            draw.text((40, 140), "MODEL: ABC-123", fill="black", font=_font())
            image.save(path)

            result = extract_text(path, backend=PaddleOCRBackend(), preprocess=False)

            self.assertEqual(result["backend"], "paddleocr")
            self.assertIn("ACME", result["raw_text"].upper())
            self.assertGreater(result["word_count"], 0)
            self.assertIsNotNone(result["confidence"])


if __name__ == "__main__":
    unittest.main()
