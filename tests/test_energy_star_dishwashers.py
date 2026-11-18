"""ENERGY STAR dishwasher matching, context, conflicts, and refresh tests."""

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
from scripts.label_candidates import candidates_from_parse, find_energy_star_exact_matches_batch
from scripts import label_candidates
from scripts.label_pipeline import _print_report, analyze_label
from scripts.test_lookup import display_result


DATASET = ENERGY_STAR_DATASETS["dishwashers"]


def dishwasher_record(record_id, model, *, annual="234", brand="Fixture"):
    return {
        "pd_id": record_id,
        "brand_name": brand,
        "model_number": model,
        "additional_model_information": "Family/model details preserved as source data",
        "upc": "012345678901",
        "type": "Standard",
        "annual_energy_use_kwh_year": annual,
        "water_use_gallons_cycle": "3.00",
        "us_federal_standard_kwh_year": "307",
        "better_than_us_federal_standard_kwh_year": "24",
        "us_federal_standard_gallons_cycle": "5",
        "better_than_us_federal_standard_gallons_cycle": "40",
        "capacity_maximum_number_of_place_settings": "15",
        "markets": "United States, Canada",
    }


def make_db(path, star_records, *, nrcan_records=()):
    refresh.replace_snapshot(star_records, "dishwashers", path)
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE dishwashers (
            REC_REF_KEY INTEGER, BRAND_NAME TEXT, MODEL_NUM_1 TEXT, TAEC_C373_04 REAL,
            source_file TEXT, appliance_category TEXT)""")
        conn.executemany(
            "INSERT INTO dishwashers VALUES (?, ?, ?, ?, ?, ?)",
            [(index, brand, model, aec, "fixture.csv", "dishwashers")
             for index, brand, model, aec in nrcan_records],
        )


class DishwasherLookupTests(unittest.TestCase):
    def test_literal_precedence_wildcard_metrics_context_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [
                dishwasher_record("family", "DW*123", annual="250"),
                dishwasher_record("literal", "DWZ123", annual="234"),
            ])

            literal_result = lookup_appliance("Fixture", "DWZ123", "dishwashers", path, False)
            star = literal_result["sources"]["energy_star"]
            self.assertEqual(star["status"], "exact_match")
            self.assertEqual(star["candidates"][0]["match_type"], "literal")
            self.assertEqual(star["annual_energy_consumption"], 234.0)
            self.assertEqual(star["unit"], "kWh/year")
            self.assertEqual(star["dataset_id"], "q8py-6w3f")
            self.assertIn("215 loads per year", star["annual_energy_context"])
            candidate = star["candidates"][0]
            self.assertEqual(candidate["source"]["url"], DATASET.dataset_url)
            self.assertEqual(candidate["source"]["record_id"], "literal")
            self.assertEqual(candidate["record"]["additional_model_information"],
                             "Family/model details preserved as source data")
            self.assertEqual(candidate["annual_energy_context"], star["annual_energy_context"])
            self.assertEqual(candidate["energy_metrics"], [
                {"name": "Water use", "value": "3.00", "unit": "gallons/cycle"},
                {"name": "U.S. federal energy standard", "value": "307", "unit": "kWh/year"},
                {"name": "Energy use below U.S. federal standard", "value": "24", "unit": "%"},
                {"name": "U.S. federal water standard", "value": "5", "unit": "gallons/cycle"},
                {"name": "Water use below U.S. federal standard", "value": "40", "unit": "%"},
            ])

            wildcard_result = lookup_appliance("Fixture", "DWY123", "dishwashers", path, False)
            wildcard = wildcard_result["sources"]["energy_star"]
            self.assertEqual(wildcard["status"], "exact_match")
            self.assertEqual(wildcard["candidates"][0]["match_type"], "wildcard")
            self.assertEqual(wildcard["annual_energy_consumption"], 250.0)

            # A mask is not a literal just because normalization strips it.
            masked_alias = lookup_appliance("Fixture", "DW123", "dishwashers", path, False)
            self.assertEqual(masked_alias["sources"]["energy_star"]["status"], "not_found")
            self.assertEqual(find_energy_star_exact_matches_batch(
                ["DW123"], manufacturer="Fixture", category="dishwashers", db_path=path
            )["DW123"], [])

    def test_multiple_wildcard_records_are_ambiguous_without_selected_energy(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [dishwasher_record("family-1", "DW*123"),
                           dishwasher_record("family-2", "DWY12#", annual="240")])
            result = lookup_appliance("Fixture", "DWY123", "dishwashers", path, False)
            star = result["sources"]["energy_star"]
            self.assertEqual(star["status"], "ambiguous_match")
            self.assertIsNone(star["annual_energy_consumption"])
            self.assertEqual(len(star["candidates"]), 2)

    def test_missing_annual_value_stays_unavailable(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [dishwasher_record("missing", "MISS123", annual=None)])
            star = lookup_appliance("Fixture", "MISS123", "dishwashers", path, False)[
                "sources"]["energy_star"]
            self.assertEqual(star["status"], "exact_match")
            self.assertIsNone(star["annual_energy_consumption"])
            self.assertIsNone(star["unit"])
            self.assertFalse(star["energy_consumption_available"])
            self.assertEqual(star["candidates"][0]["record"]["annual_energy_use_kwh_year"], None)

    def test_nrcan_conflict_keeps_values_separate_and_reports_context(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [dishwasher_record("star-1", "DWZ123", annual="234")],
                    nrcan_records=[(1, "Fixture", "DWZ123", 260.0)])
            result = lookup_appliance("Fixture", "DWZ123", "dishwashers", path, False)
            self.assertEqual(result["annual_energy_consumption"], 260.0)
            self.assertEqual(result["sources"]["nrcan"]["annual_energy_consumption"], 260.0)
            star = result["sources"]["energy_star"]
            self.assertEqual(star["annual_energy_consumption"], 234.0)
            self.assertTrue(result["energy_conflict"])
            self.assertIn("different annual energy values", result["energy_conflict_note"])

            # A nearby fuzzy NRCan record is not evidence of a same-model conflict.
            fuzzy_db = Path(temp) / "fuzzy.db"
            make_db(fuzzy_db, [dishwasher_record("star-1", "DWZ123", annual="234")],
                    nrcan_records=[(2, "Fixture", "DWZ124", 260.0)])
            fuzzy_result = lookup_appliance("Fixture", "DWZ123", "dishwashers", fuzzy_db)
            self.assertEqual(fuzzy_result["status"], "fuzzy_match")
            self.assertFalse(fuzzy_result["energy_conflict"])

            payload = analyze_label(model_number="DWZ123", manufacturer="Fixture",
                                    category="dishwashers", db_path=path, allow_fuzzy=False)
            pipeline_output = io.StringIO()
            with contextlib.redirect_stdout(pipeline_output):
                _print_report(payload)
            self.assertIn("annual energy context:", pipeline_output.getvalue())
            self.assertIn("215 loads per year", pipeline_output.getvalue())
            self.assertIn("Energy note:", pipeline_output.getvalue())

            cli_output = io.StringIO()
            with contextlib.redirect_stdout(cli_output):
                display_result(result)
            self.assertIn("Annual energy context:", cli_output.getvalue())
            self.assertIn("215 loads per year", cli_output.getvalue())
            self.assertIn("Energy note:", cli_output.getvalue())

    def test_energy_star_exact_match_validates_ocr_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [dishwasher_record("target", "DWZ1234")])
            parsed = {"brand": {"value": "Fixture"},
                      "model_number": {"value": "DWZ123A", "normalized": "DWZ123A"}}
            candidates = candidates_from_parse(parsed, category="dishwashers", db_path=path)
            target = next(item for item in candidates["candidates"]
                          if item["normalized"] == "DWZ1234")
            self.assertTrue(target["db_match"])
            self.assertEqual(target["db_sources"], ["energy_star"])

            cli_output = io.StringIO()
            with contextlib.redirect_stdout(cli_output):
                exit_code = label_candidates.main([
                    "--model", "DWZ123A", "--brand", "Fixture", "--category", "dishwashers",
                    "--db", str(path), "--json",
                ])
            self.assertEqual(exit_code, 0)
            cli_payload = json.loads(cli_output.getvalue())
            cli_target = next(item for item in cli_payload["candidates"]
                              if item["normalized"] == "DWZ1234")
            self.assertEqual(cli_target["db_sources"], ["energy_star"])


class DishwasherRefreshSafetyTests(unittest.TestCase):
    def test_later_page_download_failure_preserves_existing_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [dishwasher_record("old", "OLD123")])
            page = mock.MagicMock()
            page.__enter__.return_value = page
            page.__exit__.return_value = False
            page.read.return_value = json.dumps([
                dishwasher_record("new", "NEW123")
            ]).encode()
            with mock.patch.object(refresh, "urlopen", side_effect=[page, OSError("offline")]):
                with self.assertRaises(OSError):
                    refresh.download_records("dishwashers", page_size=1)
            with sqlite3.connect(path) as conn:
                models = conn.execute(
                    "SELECT model_number FROM energy_star_dishwashers"
                ).fetchall()
            self.assertEqual(models, [("OLD123",)])


if __name__ == "__main__":
    unittest.main()
