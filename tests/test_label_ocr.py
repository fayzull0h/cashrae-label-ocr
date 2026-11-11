"""Tests for the Phase 2 OCR foundation (``scripts/label_ocr.py``)."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image, ImageDraw, ImageFont

from scripts import label_ocr
from scripts.label_ocr import (
    OCRResult,
    OCRUnavailableError,
    PreprocessOptions,
    TesseractBackend,
    _confidence_from_data,
    _text_from_data,
    extract_directory,
    extract_text,
    find_images,
    preprocess_image,
    preprocess_image_with_steps,
    resolve_tesseract,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


def _ocr_ready() -> bool:
    """True when both pytesseract and the Tesseract binary are usable."""
    if label_ocr.pytesseract is None:
        return False
    try:
        resolve_tesseract()
    except OCRUnavailableError:
        return False
    return True


OCR_READY = _ocr_ready()


def _font(size: int = 52) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - older Pillow
        return ImageFont.load_default()


def make_label_image(path: Path, lines: tuple[str, ...] | None = None) -> Path:
    """Render a clean synthetic rating plate for deterministic OCR tests."""
    lines = lines or (
        "ACME APPLIANCES",
        "MODEL: ABC-123",
        "SERIAL: SN98765",
        "120V 60Hz 5A",
    )
    image = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(image)
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black", font=_font(52))
        y += 110
    image.save(path)
    return path


class TestPreprocessing(unittest.TestCase):
    def test_converts_to_grayscale_and_upscales(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.png"
            Image.new("RGB", (100, 50), "white").save(path)

            result = preprocess_image(path, PreprocessOptions(upscale=2.0))

            self.assertEqual(result.mode, "L")
            self.assertEqual(result.size, (200, 100))

    def test_threshold_produces_binary_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gray.png"
            Image.new("RGB", (60, 40), (120, 120, 120)).save(path)

            result = preprocess_image(path, PreprocessOptions(threshold=100))

            self.assertEqual(result.mode, "L")
            pixels = result.get_flattened_data() if hasattr(result, "get_flattened_data") else result.getdata()
            self.assertTrue(set(pixels).issubset({0, 255}))

    def test_accepts_pillow_image_objects(self):
        image = Image.new("RGB", (80, 40), "white")
        result = preprocess_image(image, PreprocessOptions(upscale=1.0))
        self.assertEqual(result.size, (80, 40))

    def test_describe_lists_applied_steps(self):
        steps = PreprocessOptions(threshold=140).describe()
        self.assertIn("grayscale", steps)
        self.assertIn("threshold 140", steps)
        self.assertTrue(any("resize" in step for step in steps))

    def test_large_image_is_downscaled_to_max_dimension(self):
        image = Image.new("RGB", (6000, 4000), "white")

        result, steps = preprocess_image_with_steps(image, PreprocessOptions())

        self.assertEqual(max(result.size), 5000)
        self.assertTrue(any("resize" in step for step in steps))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            preprocess_image("does/not/exist.png")


class TestDataHelpers(unittest.TestCase):
    def test_text_from_data_reconstructs_lines(self):
        data = {
            "text": ["ACME", "", "MODEL:", "ABC-123", "SERIAL:", "SN98765"],
            "block_num": [1, 1, 1, 1, 1, 1],
            "par_num": [1, 1, 1, 1, 1, 1],
            "line_num": [1, 1, 2, 2, 3, 3],
        }
        self.assertEqual(_text_from_data(data), "ACME\nMODEL: ABC-123\nSERIAL: SN98765")

    def test_confidence_averages_valid_words(self):
        data = {"conf": ["90", "80", "-1", "bad", "100"]}
        self.assertEqual(_confidence_from_data(data), 0.9)

    def test_confidence_none_when_no_valid_words(self):
        self.assertIsNone(_confidence_from_data({"conf": ["-1", "-1"]}))

    def test_ocr_result_to_dict_omits_empty_debug_image(self):
        result = OCRResult(
            image_path="x.png",
            raw_text="hello",
            confidence=0.5,
            backend="tesseract",
            preprocessing=["grayscale"],
            word_count=1,
        )
        data = result.to_dict()
        self.assertEqual(data["raw_text"], "hello")
        self.assertNotIn("preprocessed_image", data)


class TestBackendResolution(unittest.TestCase):
    def test_env_var_takes_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "tesseract.exe"
            fake.write_text("stub")
            with mock.patch.dict(os.environ, {"TESSERACT_CMD": str(fake)}):
                self.assertEqual(resolve_tesseract(), str(fake))

    def test_raises_with_install_hint_when_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(label_ocr.shutil, "which", return_value=None), \
                mock.patch.object(label_ocr, "WINDOWS_DEFAULT_PATHS", []):
            with self.assertRaises(OCRUnavailableError) as ctx:
                resolve_tesseract()
            self.assertIn("winget install", str(ctx.exception))

    def test_backend_raises_when_pytesseract_missing(self):
        with mock.patch.object(label_ocr, "pytesseract", None):
            with self.assertRaises(OCRUnavailableError):
                TesseractBackend()


class TestImageDiscovery(unittest.TestCase):
    def test_find_images_filters_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.JPG").write_bytes(b"x")
            (root / "a.png").write_bytes(b"x")
            (root / "notes.txt").write_text("ignore")
            nested = root / "nested"
            nested.mkdir()
            (nested / "c.webp").write_bytes(b"x")

            found = [path.name for path in find_images(root)]

            self.assertEqual(found, ["a.png", "b.JPG", "c.webp"])

    def test_missing_directory_returns_empty(self):
        self.assertEqual(find_images("does/not/exist"), [])


@unittest.skipUnless(OCR_READY, "Tesseract or pytesseract not available")
class TestOcrIntegration(unittest.TestCase):
    def test_extract_text_from_synthetic_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_label_image(Path(tmp) / "label.png")

            result = extract_text(path, backend=TesseractBackend())

            self.assertEqual(result["backend"], "tesseract")
            self.assertGreater(result["word_count"], 0)
            self.assertIn("ACME", result["raw_text"])
            self.assertIn("ABC", result["raw_text"])
            self.assertIsNotNone(result["confidence"])
            self.assertGreaterEqual(result["confidence"], 0.0)

    def test_extract_text_without_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_label_image(Path(tmp) / "label.png")

            result = extract_text(path, backend=TesseractBackend(), preprocess=False)

            self.assertEqual(result["preprocessing"], [])
            self.assertIn("MODEL", result["raw_text"])

    def test_debug_mode_saves_preprocessed_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_label_image(root / "label.png")
            debug_dir = root / "processed"

            result = extract_text(path, backend=TesseractBackend(), debug_dir=debug_dir)

            saved = Path(result["preprocessed_image"])
            self.assertTrue(saved.is_file())
            self.assertEqual(saved.parent, debug_dir)

    def test_batch_directory_reports_each_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_label_image(root / "one.png")
            make_label_image(root / "two.png")

            results = extract_directory(root, backend=TesseractBackend())

            self.assertEqual(len(results), 2)
            self.assertTrue(all("raw_text" in result for result in results))

    def test_batch_isolates_bad_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_label_image(root / "good.png")
            (root / "broken.png").write_bytes(b"not really a png")

            results = extract_directory(root, backend=TesseractBackend())

            self.assertEqual(len(results), 2)
            self.assertTrue(any("error" in result for result in results))
            self.assertTrue(any("raw_text" in result for result in results))

    def test_cli_json_output_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_label_image(Path(tmp) / "label.png")
            buffer = StringIO()

            with redirect_stdout(buffer):
                exit_code = label_ocr.main(["--image", str(path), "--json", "--engine", "tesseract"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(buffer.getvalue())
            self.assertIn("raw_text", payload)
            self.assertIn("ACME", payload["raw_text"])

    def test_cli_missing_image_returns_error_code(self):
        with redirect_stdout(StringIO()), mock.patch("sys.stderr", new_callable=StringIO):
            exit_code = label_ocr.main(["--image", "does/not/exist.png"])
        self.assertEqual(exit_code, 1)

    def test_real_samples_if_present(self):
        images = find_images(SAMPLES_DIR)
        if not images:
            self.skipTest("no real photos in data/samples/")
        result = extract_text(images[0], backend=TesseractBackend())
        self.assertTrue(result["raw_text"].strip())


class TestCliEncoding(unittest.TestCase):
    def test_cli_json_survives_non_ascii_on_cp1252_stdout(self):
        """Redirected Windows stdout is cp1252; non-ASCII OCR text must not crash."""
        from scripts import ocr_backends

        backend = mock.Mock()
        backend.name = "paddleocr"
        backend.extract.return_value = ("MODEL 25A 米", 0.9)

        with tempfile.TemporaryDirectory() as tmp:
            path = make_label_image(Path(tmp) / "label.png")
            raw = BytesIO()
            stream = TextIOWrapper(raw, encoding="cp1252")
            with mock.patch.object(ocr_backends, "get_backend", return_value=backend), redirect_stdout(stream):
                exit_code = label_ocr.main(["--image", str(path), "--json"])
            stream.flush()
            payload = json.loads(raw.getvalue().decode("utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn("米", payload["raw_text"])


if __name__ == "__main__":
    unittest.main()
