"""ENERGY STAR refrigerator snapshot, lookup, and cross-source tests."""

import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from scripts import refresh_energy_star as refresh
from scripts.appliance_lookup import lookup_appliance
from scripts.energy_star_config import ENERGY_STAR_DATASETS
from scripts.label_candidates import candidates_from_parse
from scripts.label_pipeline import _print_report, analyze_label
from scripts.test_lookup import display_result


DATASET = ENERGY_STAR_DATASETS["refrigerators"]


def refrigerator_record(record_id, model, *, brand="Fixture", annual="615", info=""):
    return {
        "pd_id": record_id,
        "brand_name": brand,
        "model_number": model,
        "additional_model_information": info,
        "upc": "012345678901",
        "type": "Side-by-Side",
        "capacity_total_volume_ft3": "22.3",
        "adjusted_volume_ft3": "28.3",
        "annual_energy_use_kwh_yr": annual,
        "us_federal_standard_kwh_yr": "674",
        "percent_less_energy_use_than_us_federal_standard": "10",
    }


def make_db(path, star_records, *, nrcan_records=()):
    refresh.replace_snapshot(star_records, "refrigerators", path)
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE refrigerators (
            REC_REF_KEY INTEGER, BRAND_NAME TEXT, MODEL_NUM_1 TEXT, AEC REAL,
            source_file TEXT, appliance_category TEXT)""")
        conn.executemany(
            "INSERT INTO refrigerators VALUES (?, ?, ?, ?, ?, ?)",
            [(index, brand, model, aec, "fixture.csv", "refrigerators")
             for index, brand, model, aec in nrcan_records],
        )


class RefrigeratorLookupTests(unittest.TestCase):
    def test_literal_precedence_wildcard_matching_and_source_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [
                refrigerator_record("family", "RF*123", annual="500",
                                    info="RFZ123 alternate retail model"),
                refrigerator_record("literal", "RFZ123", annual="615", info="Literal model"),
            ])

            literal_result = lookup_appliance("Fixture", "RFZ123", "refrigerators", path, False)
            star = literal_result["sources"]["energy_star"]
            self.assertEqual(star["status"], "exact_match")
            self.assertEqual(star["candidates"][0]["match_type"], "literal")
            self.assertEqual(star["annual_energy_consumption"], 615.0)
            self.assertEqual(star["unit"], "kWh/year")
            self.assertEqual(star["candidates"][0]["source"]["dataset_id"], "p5st-her9")
            self.assertEqual(star["candidates"][0]["source"]["url"], DATASET.dataset_url)
            self.assertEqual(star["candidates"][0]["source"]["record_id"], "literal")
            self.assertEqual(star["candidates"][0]["record"]["additional_model_information"],
                             "Literal model")
            self.assertEqual(star["candidates"][0]["energy_metrics"], [
                {"name": "Capacity (total volume)", "value": "22.3", "unit": "ft³"},
                {"name": "Adjusted volume", "value": "28.3", "unit": "ft³"},
                {"name": "U.S. federal standard", "value": "674", "unit": "kWh/year"},
                {"name": "Less energy than U.S. federal standard", "value": "10", "unit": "%"},
            ])

            wildcard_result = lookup_appliance("Fixture", "RFY123", "refrigerators", path, False)
            wildcard = wildcard_result["sources"]["energy_star"]
            self.assertEqual(wildcard["status"], "exact_match")
            self.assertEqual(wildcard["candidates"][0]["match_type"], "wildcard")
            self.assertEqual(wildcard["annual_energy_consumption"], 500.0)

            info_only = lookup_appliance("Fixture", "RFZ123 alternate retail model",
                                         "refrigerators", path, False)
            self.assertEqual(info_only["sources"]["energy_star"]["status"], "not_found")

    def test_repeated_family_records_remain_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [refrigerator_record("family-1", "CRSEH237A*"),
                           refrigerator_record("family-2", "CRSEH237A*", annual="620")])
            result = lookup_appliance("Fixture", "CRSEH237AW", "refrigerators", path, False)
            star = result["sources"]["energy_star"]
            self.assertEqual(star["status"], "ambiguous_match")
            self.assertIsNone(star["annual_energy_consumption"])
            self.assertEqual(len(star["candidates"]), 2)

    def test_missing_energy_does_not_fabricate_value(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [refrigerator_record("missing", "MISS123", annual=None)])
            result = lookup_appliance("Fixture", "MISS123", "refrigerators", path, False)
            star = result["sources"]["energy_star"]
            self.assertEqual(star["status"], "exact_match")
            self.assertIsNone(star["annual_energy_consumption"])
            self.assertIsNone(star["unit"])
            self.assertFalse(star["energy_consumption_available"])
            self.assertEqual(star["candidates"][0]["record"]["annual_energy_use_kwh_yr"], None)

    def test_annual_value_conflict_stays_source_specific_and_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [refrigerator_record("star-1", "RFZ123", annual="615")],
                    nrcan_records=[(1, "Fixture", "RFZ123", 590.0)])
            result = lookup_appliance("Fixture", "RFZ123", "refrigerators", path, False)
            self.assertEqual(result["annual_energy_consumption"], 590.0)
            self.assertEqual(result["sources"]["nrcan"]["annual_energy_consumption"], 590.0)
            self.assertEqual(result["sources"]["energy_star"]["annual_energy_consumption"], 615.0)
            self.assertTrue(result["energy_conflict"])
            self.assertIn("different annual energy values", result["energy_conflict_note"])

            payload = analyze_label(model_number="RFZ123", manufacturer="Fixture",
                                    category="refrigerators", db_path=path,
                                    allow_fuzzy=False)
            self.assertTrue(payload["retrieval"]["energy_conflict"])
            pipeline_output = io.StringIO()
            with contextlib.redirect_stdout(pipeline_output):
                _print_report(payload)
            self.assertIn("annual energy: 615.0 kWh/year", pipeline_output.getvalue())
            self.assertIn("Energy note:", pipeline_output.getvalue())
            self.assertIn("record star-1", pipeline_output.getvalue())

            cli_output = io.StringIO()
            with contextlib.redirect_stdout(cli_output):
                display_result(result)
            self.assertIn("Annual energy: 615.0 kWh/year", cli_output.getvalue())
            self.assertIn("Energy note:", cli_output.getvalue())

    def test_energy_star_confirms_ocr_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [refrigerator_record("star-target", "RFZ1234")])
            parsed = {"brand": {"value": "Fixture"},
                      "model_number": {"value": "RFZ123A", "normalized": "RFZ123A"}}
            payload = candidates_from_parse(parsed, category="refrigerators", db_path=path)
            true_candidate = next(candidate for candidate in payload["candidates"]
                                  if candidate["normalized"] == "RFZ1234")
            self.assertTrue(true_candidate["db_match"])
            self.assertEqual(true_candidate["db_sources"], ["energy_star"])


class RefrigeratorRefreshSafetyTests(unittest.TestCase):
    def test_download_failure_on_later_page_leaves_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [refrigerator_record("old", "OLD123")])
            page = mock.MagicMock()
            page.__enter__.return_value = page
            page.__exit__.return_value = False
            page.read.return_value = json.dumps([
                refrigerator_record("new", "NEW123")
            ]).encode()
            with mock.patch.object(refresh, "urlopen", side_effect=[page, OSError("offline")]):
                with self.assertRaises(OSError):
                    refresh.download_records("refrigerators", page_size=1)
            with sqlite3.connect(path) as conn:
                models = conn.execute(
                    "SELECT model_number FROM energy_star_refrigerators"
                ).fetchall()
            self.assertEqual(models, [("OLD123",)])


if __name__ == "__main__":
    unittest.main()
