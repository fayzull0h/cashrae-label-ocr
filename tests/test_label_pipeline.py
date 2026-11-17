"""Tests for Phase 2 Stage 4 label → energy pipeline (``scripts/label_pipeline.py``)."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from scripts import label_pipeline
from scripts.label_pipeline import analyze_label

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUMP_PATH = PROJECT_ROOT / "data" / "processed" / "ocr_dump.json"

#: Stable fixture row: ``PCKS433ET#**`` / GE in the clothes dryers table.
KNOWN_MODEL = "PCKS433ET"
KNOWN_BRAND = "GE"
KNOWN_CATEGORY = "clothes_dryers"

#: One OCR substitution away from ``PCKS443EB#**`` (also a GE clothes dryer).
CORRUPTED_MODEL = "PCKS443E8"
TRUE_CORRECTED_MODEL = "PCKS443EB"

#: A verified-plate model that is deliberately absent from the NRCan data.
ABSENT_MODEL = "DLE8400WE"


def _stub_ocr(raw_text: str) -> dict:
    return {
        "image_path": "data/samples/airconditioner_label.png",
        "raw_text": raw_text,
        "confidence": 0.93,
        "backend": "stub",
        "preprocessing": {"steps": ["grayscale"]},
        "word_count": len(raw_text.split()),
    }


class TestManualOverride(unittest.TestCase):
    def test_exact_match_is_high_confidence_and_from_original(self):
        result = analyze_label(
            model_number=KNOWN_MODEL,
            manufacturer=KNOWN_BRAND,
            category=KNOWN_CATEGORY,
        )

        self.assertEqual(result["status"], "exact_match")
        self.assertEqual(result["retrieval"]["model_used"], KNOWN_MODEL)
        self.assertTrue(result["retrieval"]["from_original"])
        self.assertEqual(result["retrieval"]["annual_energy_consumption"], 430.0)
        self.assertEqual(result["retrieval"]["unit"], "kWh/year")
        self.assertTrue(result["retrieval"]["energy_consumption_available"])
        self.assertEqual(result["confidence"], "high")
        self.assertIsNone(result["ocr"])
        self.assertIsNone(result["parsed"])

    def test_manual_override_skips_ocr_and_parsing(self):
        result = analyze_label(
            image_path="does/not/exist.png",
            model_number=KNOWN_MODEL,
            manufacturer=KNOWN_BRAND,
            category=KNOWN_CATEGORY,
        )

        self.assertIsNone(result["ocr"])
        self.assertEqual(result["status"], "exact_match")

    def test_model_number_is_normalized(self):
        result = analyze_label(
            model_number="pcks-433-et#**",
            manufacturer=KNOWN_BRAND,
            category=KNOWN_CATEGORY,
        )

        self.assertEqual(result["retrieval"]["model_used"], KNOWN_MODEL)

    def test_fuzzy_match_is_not_high_confidence(self):
        result = analyze_label(
            model_number="PCKS433EX",
            manufacturer=KNOWN_BRAND,
            category=KNOWN_CATEGORY,
        )

        self.assertEqual(result["status"], "fuzzy_match")
        self.assertIn(result["confidence"], ("medium", "low"))
        self.assertGreater(result["retrieval"]["match_confidence"], 0.0)

    def test_absent_model_with_fuzzy_disabled_is_not_found(self):
        result = analyze_label(
            model_number=ABSENT_MODEL,
            manufacturer="LG",
            category=KNOWN_CATEGORY,
            allow_fuzzy=False,
        )

        self.assertEqual(result["status"], "not_found")
        self.assertIsNone(result["confidence"])
        self.assertFalse(result["retrieval"]["energy_consumption_available"])
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(result["attempts"][0]["result"], "not_found")

    def test_room_air_conditioner_reports_no_aec_honestly(self):
        result = analyze_label(
            model_number="SCA09LS",
            manufacturer="Wallmate",
            category="room_air_conditioners",
        )

        self.assertEqual(result["status"], "exact_match")
        self.assertIsNone(result["retrieval"]["annual_energy_consumption"])
        self.assertFalse(result["retrieval"]["energy_consumption_available"])
        self.assertIn("EE_RATIO", result["retrieval"]["notes"])


class TestCorrectionPath(unittest.TestCase):
    def test_single_character_error_is_corrected_from_parse_output(self):
        result = analyze_label(
            raw_text=f"{KNOWN_BRAND}\nMODEL: {CORRUPTED_MODEL}",
            category=KNOWN_CATEGORY,
        )

        self.assertEqual(result["status"], "exact_match")
        self.assertEqual(result["retrieval"]["model_used"], TRUE_CORRECTED_MODEL)
        self.assertFalse(result["retrieval"]["from_original"])
        self.assertEqual(result["confidence"], "medium")
        self.assertEqual(result["retrieval"]["annual_energy_consumption"], 430.0)
        self.assertEqual(result["confidence_components"]["candidate_rank"], 2)

    def test_original_attempt_is_first_and_unchanged(self):
        result = analyze_label(
            raw_text=f"MODEL: {CORRUPTED_MODEL}",
            category=KNOWN_CATEGORY,
        )

        first = result["attempts"][0]
        self.assertTrue(first["is_original"])
        self.assertEqual(first["model"], CORRUPTED_MODEL)
        self.assertEqual(result["parsed"]["model_number"]["value"], CORRUPTED_MODEL)

    def test_attempts_stop_at_the_first_exact_hit(self):
        result = analyze_label(
            raw_text=f"MODEL: {CORRUPTED_MODEL}",
            category=KNOWN_CATEGORY,
        )

        self.assertEqual(result["attempts"][-1]["model"], TRUE_CORRECTED_MODEL)
        self.assertEqual(result["attempts"][-1]["result"], "exact_match")

    def test_max_attempts_limits_the_search(self):
        result = analyze_label(
            raw_text=f"MODEL: {CORRUPTED_MODEL}",
            category=KNOWN_CATEGORY,
            max_attempts=1,
            allow_fuzzy=False,
        )

        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(result["status"], "not_found")

    def test_absent_model_falls_back_to_fuzzy_by_default(self):
        result = analyze_label(
            raw_text=f"LG\nMODEL: {ABSENT_MODEL}",
            category=KNOWN_CATEGORY,
        )

        self.assertIn(result["status"], ("fuzzy_match", "ambiguous_match", "not_found"))
        self.assertIsNone(result["confidence_components"]["candidate_rank"])

    def test_absent_model_with_fuzzy_disabled_returns_not_found(self):
        result = analyze_label(
            raw_text=f"LG\nMODEL: {ABSENT_MODEL}",
            category=KNOWN_CATEGORY,
            allow_fuzzy=False,
        )

        self.assertEqual(result["status"], "not_found")
        self.assertIsNone(result["retrieval"]["annual_energy_consumption"])


class TestNoModel(unittest.TestCase):
    def test_text_without_a_model_returns_no_model_found(self):
        result = analyze_label(raw_text="ACME APPLIANCES INC.\n120V 60Hz 5A")

        self.assertEqual(result["status"], "no_model_found")
        self.assertIsNone(result["retrieval"])
        self.assertEqual(result["attempts"], [])
        self.assertIsNone(result["confidence"])

    def test_missing_input_raises(self):
        with self.assertRaises(ValueError):
            analyze_label()

    def test_invalid_category_raises(self):
        with self.assertRaises(ValueError):
            analyze_label(
                model_number=KNOWN_MODEL,
                manufacturer=KNOWN_BRAND,
                category="not_a_category",
            )

    def test_invalid_max_attempts_raises(self):
        with self.assertRaises(ValueError):
            analyze_label(model_number=KNOWN_MODEL, max_attempts=0)


class TestOcrIntegration(unittest.TestCase):
    def test_mocked_ocr_text_flows_through_parse_and_retrieval(self):
        with mock.patch(
            "scripts.label_ocr.extract_text",
            return_value=_stub_ocr("BRAND: GE\nMODEL: PCKS433ET\nSERIAL: 12345"),
        ):
            result = analyze_label(image_path="fake.png", category=KNOWN_CATEGORY)

        self.assertEqual(result["ocr"]["backend"], "stub")
        self.assertEqual(result["ocr"]["confidence"], 0.93)
        self.assertEqual(result["parsed"]["model_number"]["value"], KNOWN_MODEL)
        self.assertEqual(result["status"], "exact_match")
        self.assertEqual(result["confidence"], "high")

    def test_parsed_fixture_plate_is_honestly_not_found(self):
        dump = json.loads(DUMP_PATH.read_text(encoding="utf-8"))
        entry = next(item for item in dump if item["image_path"].endswith("dryer_label.png"))

        with mock.patch("scripts.label_ocr.extract_text", return_value=_stub_ocr(entry["raw_text"])):
            result = analyze_label(
                image_path=entry["image_path"],
                category="clothes_dryers",
                allow_fuzzy=False,
            )

        self.assertEqual(result["parsed"]["model_number"]["value"], ABSENT_MODEL)
        self.assertEqual(result["status"], "not_found")
        self.assertFalse(result["retrieval"]["energy_consumption_available"])

    def test_engine_argument_is_passed_to_backend_selection(self):
        with mock.patch("scripts.ocr_backends.get_backend") as get_backend, mock.patch(
            "scripts.label_ocr.extract_text",
            return_value=_stub_ocr("MODEL: PCKS433ET"),
        ):
            analyze_label(image_path="fake.png", engine="tesseract")

        get_backend.assert_called_once_with("tesseract")


class TestCli(unittest.TestCase):
    def test_json_output(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            exit_code = label_pipeline.main(
                [
                    "--model",
                    KNOWN_MODEL,
                    "--brand",
                    KNOWN_BRAND,
                    "--category",
                    KNOWN_CATEGORY,
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["retrieval"]["annual_energy_consumption"], 430.0)
        self.assertEqual(payload["confidence"], "high")

    def test_text_report_includes_confidence_and_attempts(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            exit_code = label_pipeline.main(
                ["--model", KNOWN_MODEL, "--brand", KNOWN_BRAND, "--category", KNOWN_CATEGORY]
            )

        report = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Confidence: high", report)
        self.assertIn("Attempts:", report)

    def test_output_file_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pipeline.json"
            with redirect_stdout(StringIO()):
                exit_code = label_pipeline.main(
                    [
                        "--model",
                        KNOWN_MODEL,
                        "--brand",
                        KNOWN_BRAND,
                        "--category",
                        KNOWN_CATEGORY,
                        "--json",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "exact_match")

    def test_no_model_found_returns_error_code(self):
        with redirect_stdout(StringIO()), mock.patch("sys.stderr", new_callable=StringIO):
            exit_code = label_pipeline.main(["--text", "ACME APPLIANCES INC.\n120V"])

        self.assertEqual(exit_code, 1)

    def test_invalid_category_returns_error_code(self):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            exit_code = label_pipeline.main(
                ["--model", KNOWN_MODEL, "--category", "not_a_category"]
            )

        self.assertEqual(exit_code, 1)

    def test_no_fuzzy_flag_is_honoured(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            exit_code = label_pipeline.main(
                [
                    "--model",
                    ABSENT_MODEL,
                    "--brand",
                    "LG",
                    "--category",
                    KNOWN_CATEGORY,
                    "--no-fuzzy",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["status"], "not_found")


if __name__ == "__main__":
    unittest.main()
