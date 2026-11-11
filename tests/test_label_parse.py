"""Tests for Phase 2 Stage 2 field parsing (``scripts/label_parse.py``)."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from scripts import label_parse
from scripts.appliance_lookup import normalize_alphanum
from scripts.label_parse import (
    KNOWN_BRANDS,
    model_score,
    parse_fields,
    parse_ocr_result,
    serial_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
DUMP_PATH = PROJECT_ROOT / "data" / "processed" / "ocr_dump.json"
GROUND_TRUTH_PATH = SAMPLES_DIR / "ground_truth.json"

CLEAN_PLATE = """ACME APPLIANCES INC.
MODEL NO: ABC-123
SERIAL NO: SN98765
120V 60Hz 5A"""

BILINGUAL_PLATE = """MARQUE: Danby
MODÈLE: DAR044A4WDD-6
SÉRIE: 4322043058424"""

RIGHT_ALIGNED_PLATE = """ABC-1234 MODEL NO.
SN98765 SERIAL NO."""

VALUE_ABOVE_PLATE = """E140302785
SERIAL
NXA630GKA100
MODEL"""

GLUED_PLATE = """KitchenAidMOD
F34203222
SER
KUDC10FXSS7"""

# Tesseract output from the original tracked fixture: noisy, but still parseable.
NOISY_TESSERACT = """#
AW
i
| :
‘ a
|
Me,
PROD = NXABZOGKAIOO
MODEL NXAGB3OGKA10@
METERING TRY NA
DEVICE INDOOR OUTDOOR
FACTORY CHARGED R410A
Gio Leenks 3509 56
INDOOR TXY SUB COOLING 1OsasF
POWER SUPPLY 299-23@ VOLTS AC
1 PH G@- Hz
PERMISSIBLE VOLTAGE AT UNIT 5
oS) lait 1 We
SUITABLE FOR OUTDOOR USE
COMPRESSOR 998/230 VOLTS AC
1 PH Bor He
vets] 64.6 LRA
FAN MOTOR 3998/7230 VOLTS Ac
1 PH 6@ 12
1716 HP Glarommaee
DESIGN/TEST PRESSURE GAGE
HI 45@ PsI 3163 ~ KPA
Lo 25@ PSI 1724 KPA
MAX DESIGN/?WORKING PRESSURE
7 OGiee ancle 4826 KPA
MINIMUM CIRCUIT AMPS 16.8
Max FUSE |MAX CKT-BKR(*) @5 A
e5A
Short Circuit Current:5kA rms, symmetrical, 230 V
FOLNOUTUULATUUM
MODEL NUMBER NXAG3@
GN
SERIAL NUMBER 302785
DATE OF Dee rtEe Toe 2614
Car ¢(UL)us LISTED
cnn CMOS
ENERGY STAR CENTRAL COOLING AIR CONDITIONER
3R38
ASSES CERTIFIED»
Mee yw _AHRIDIRECTORY.ORG
Unitary Small AC
AHRI Standard 210/240
Certification applies only when the complete
system is listed with AHRE
internatid al Comfort Progucts 337129-4903
st 3709 SA"""


def _value(parsed: dict, key: str):
    return parsed[key]["value"]


def _normalized(parsed: dict, key: str):
    return parsed[key]["normalized"]


def _serial_matches(parsed: dict, expected: str) -> bool:
    """Serial prefixes are not scored, so accept a suffix match."""
    actual = _normalized(parsed, "serial_number")
    wanted = normalize_alphanum(expected)
    return bool(actual) and (actual == wanted or actual.endswith(wanted))


class TestTokenScoring(unittest.TestCase):
    def test_model_score_accepts_mixed_tokens(self):
        self.assertGreaterEqual(model_score("NXA630GKA100"), 0.95)
        self.assertGreaterEqual(model_score("DAR044A4WDD-6"), 0.95)
        self.assertGreaterEqual(model_score("WT8400CW"), 0.95)

    def test_model_score_rejects_serial_like_tokens(self):
        self.assertLess(model_score("E140302785"), 0.95)
        self.assertLess(model_score("4322043058424"), 0.95)

    def test_model_score_rejects_noise_and_dates(self):
        self.assertEqual(model_score("120V"), 0.0)
        self.assertEqual(model_score("2022.04"), 0.0)
        self.assertEqual(model_score("R410A"), 0.0)
        self.assertEqual(model_score("MODEL"), 0.0)

    def test_serial_score_prefers_digit_heavy_tokens(self):
        self.assertGreaterEqual(serial_score("4322043058424"), 0.9)
        self.assertGreaterEqual(serial_score("503KWVQ6K991"), 0.9)
        self.assertLess(serial_score("KUDC10FXSS7"), 0.9)


class TestAnchorParsing(unittest.TestCase):
    def test_clean_english_plate(self):
        parsed = parse_fields(CLEAN_PLATE)

        self.assertEqual(_value(parsed, "brand"), "ACME APPLIANCES INC.")
        self.assertEqual(_value(parsed, "model_number"), "ABC-123")
        self.assertEqual(_value(parsed, "serial_number"), "SN98765")
        self.assertEqual(parsed["model_number"]["method"], "label_anchor")
        self.assertEqual(parsed["serial_number"]["method"], "label_anchor")
        self.assertEqual(parsed["model_number"]["confidence"], 0.9)
        self.assertEqual(parsed["serial_number"]["confidence"], 0.9)

    def test_bilingual_french_anchors(self):
        parsed = parse_fields(BILINGUAL_PLATE)

        self.assertEqual(_value(parsed, "brand"), "Danby")
        self.assertEqual(_value(parsed, "model_number"), "DAR044A4WDD-6")
        self.assertEqual(_value(parsed, "serial_number"), "4322043058424")

    def test_value_before_anchor_on_same_line(self):
        parsed = parse_fields(RIGHT_ALIGNED_PLATE)

        self.assertEqual(_value(parsed, "model_number"), "ABC-1234")
        self.assertEqual(_value(parsed, "serial_number"), "SN98765")
        self.assertEqual(parsed["model_number"]["method"], "label_anchor_before")
        self.assertEqual(parsed["serial_number"]["method"], "label_anchor_before")

    def test_value_above_anchor_line(self):
        parsed = parse_fields(VALUE_ABOVE_PLATE)

        self.assertEqual(_value(parsed, "model_number"), "NXA630GKA100")
        self.assertEqual(_value(parsed, "serial_number"), "E140302785")
        self.assertEqual(parsed["model_number"]["method"], "adjacent_line")
        self.assertEqual(parsed["serial_number"]["method"], "adjacent_line")

    def test_brand_glued_to_model_anchor(self):
        parsed = parse_fields(GLUED_PLATE)

        self.assertEqual(_value(parsed, "brand"), "KitchenAid")
        self.assertEqual(_value(parsed, "model_number"), "KUDC10FXSS7")
        self.assertEqual(_value(parsed, "serial_number"), "F34203222")

    def test_brand_anchor_value_on_next_line(self):
        parsed = parse_fields("MANUFACTURER:\nAcme Corp.")

        self.assertEqual(_value(parsed, "brand"), "Acme Corp.")
        self.assertEqual(parsed["brand"]["method"], "adjacent_line")

    def test_abbreviated_anchors(self):
        parsed = parse_fields("MOD. XYZ-100\nSN. 445566")

        self.assertEqual(_value(parsed, "model_number"), "XYZ-100")
        self.assertEqual(_value(parsed, "serial_number"), "445566")

    def test_unparsed_lines_keep_remaining_text(self):
        parsed = parse_fields(CLEAN_PLATE)

        self.assertEqual(parsed["unparsed_lines"], ["120V 60Hz 5A"])

    def test_model_and_serial_are_never_identical(self):
        parsed = parse_fields("MODEL: ABC12345\nSERIAL: ABC12345")

        self.assertEqual(_value(parsed, "model_number"), "ABC12345")
        self.assertIsNone(_value(parsed, "serial_number"))


class TestHeuristicParsing(unittest.TestCase):
    def test_noisy_tesseract_output(self):
        parsed = parse_fields(NOISY_TESSERACT)

        self.assertTrue(_normalized(parsed, "model_number").startswith("NXAG"))
        self.assertEqual(_normalized(parsed, "serial_number"), "302785")
        self.assertIsNone(_value(parsed, "brand"))

    def test_custom_known_brand(self):
        parsed = parse_fields(
            "ACME WIDGETS\nMODEL: ACM-999",
            known_brands=[("ACMEWIDGETS", "Acme Widgets")],
        )

        self.assertEqual(_value(parsed, "brand"), "Acme Widgets")
        self.assertEqual(parsed["brand"]["method"], "known_brand")

    def test_custom_brands_do_not_leak_into_module_state(self):
        before = KNOWN_BRANDS

        parse_fields("ACME WIDGETS", known_brands=[("ACMEWIDGETS", "Acme Widgets")])

        self.assertIs(KNOWN_BRANDS, before)

    def test_text_without_fields_returns_empty_values(self):
        parsed = parse_fields("hello world\nnothing to see here")

        for key in ("brand", "model_number", "serial_number"):
            self.assertIsNone(_value(parsed, key))
            self.assertEqual(parsed[key]["confidence"], 0.0)


class TestVerifiedPlates(unittest.TestCase):
    """Regression coverage for the six hand-verified rating plates."""

    @classmethod
    def setUpClass(cls):
        cls.dump = json.loads(DUMP_PATH.read_text(encoding="utf-8"))
        cls.truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))

    def test_fixture_covers_every_verified_plate(self):
        names = {Path(entry["image_path"]).name for entry in self.dump}

        self.assertEqual(names, set(self.truth))

    def test_brand_and_model_on_all_six_plates(self):
        for entry in self.dump:
            name = Path(entry["image_path"]).name
            expected = self.truth[name]
            parsed = parse_fields(entry["raw_text"])

            with self.subTest(image=name):
                self.assertEqual(_value(parsed, "brand"), expected["brand"])
                self.assertEqual(_value(parsed, "model_number"), expected["model_number"])

    def test_serial_on_all_six_plates_ignoring_prefix(self):
        for entry in self.dump:
            name = Path(entry["image_path"]).name
            expected = self.truth[name]
            parsed = parse_fields(entry["raw_text"])

            with self.subTest(image=name):
                self.assertTrue(
                    _serial_matches(parsed, expected["serial_number"]),
                    msg=f"{_value(parsed, 'serial_number')!r} vs {expected['serial_number']!r}",
                )

    def test_exact_serial_on_at_least_five_plates(self):
        exact = 0
        for entry in self.dump:
            name = Path(entry["image_path"]).name
            expected = normalize_alphanum(self.truth[name]["serial_number"])
            parsed = parse_fields(entry["raw_text"])
            if _normalized(parsed, "serial_number") == expected:
                exact += 1

        self.assertGreaterEqual(exact, 5)


class TestOcrResultBridge(unittest.TestCase):
    def test_parse_ocr_result_copies_provenance(self):
        result = {
            "image_path": "data/samples/label.png",
            "raw_text": CLEAN_PLATE,
            "confidence": 0.91,
            "backend": "paddleocr",
            "preprocessing": ["grayscale", "autocontrast"],
            "word_count": 12,
        }

        parsed = parse_ocr_result(result)

        self.assertEqual(parsed["image_path"], "data/samples/label.png")
        self.assertEqual(
            parsed["ocr"],
            {
                "backend": "paddleocr",
                "confidence": 0.91,
                "preprocessing": ["grayscale", "autocontrast"],
            },
        )
        self.assertEqual(_value(parsed, "model_number"), "ABC-123")


class TestCli(unittest.TestCase):
    def test_text_mode_emits_json(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            exit_code = label_parse.main(["--text", CLEAN_PLATE, "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["model_number"]["value"], "ABC-123")

    def test_text_mode_prints_report(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            exit_code = label_parse.main(["--text", CLEAN_PLATE])

        self.assertEqual(exit_code, 0)
        self.assertIn("model_number   ABC-123", buffer.getvalue())

    def test_output_file_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "parsed.json"
            with redirect_stdout(StringIO()):
                exit_code = label_parse.main(
                    ["--text", CLEAN_PLATE, "--json", "--output", str(output)]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["brand"]["value"], "ACME APPLIANCES INC.")

    def test_missing_text_file_returns_error_code(self):
        with redirect_stdout(StringIO()), mock.patch("sys.stderr", new_callable=StringIO):
            exit_code = label_parse.main(["--text-file", "does/not/exist.txt"])

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
