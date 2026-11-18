"""Room AC coverage across ENERGY STAR refresh, lookup, candidates, and reporting."""

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


DATASET = ENERGY_STAR_DATASETS["room_air_conditioners"]


def room_record(record_id, model, *, annual="400", ceer="15", capacity="8000", savings="38"):
    return {
        "pd_id": record_id,
        "brand_name": "Midea",
        "model_number": model,
        "annual_energy_use_kwh_yr": annual,
        "combined_energy_efficiency_ratio_ceer": ceer,
        "cooling_capacity_btu_hour": capacity,
        "percent_less_energy_use_than_us_federal_standard": savings,
        "type": "Window",
        "markets": "United States, Canada",
    }


def make_db(path, records, *, nrcan_records=()):
    refresh.replace_snapshot(records, "room_air_conditioners", path)
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE room_air_conditioners (
            REC_REF_KEY INTEGER, BRAND_NAME TEXT, MODEL_NUM_1 TEXT, EE_RATIO REAL,
            source_file TEXT, appliance_category TEXT)""")
        conn.executemany(
            "INSERT INTO room_air_conditioners VALUES (?, ?, ?, ?, ?, ?)",
            [(index, brand, model, eer, "fixture.csv", "room_air_conditioners")
             for index, brand, model, eer in nrcan_records],
        )


class RoomAirConditionerTests(unittest.TestCase):
    def test_literal_then_wildcard_matches_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [room_record("star-literal", "ACX1234"),
                           room_record("star-family", "AC*123#", annual="300")])

            literal_result = lookup_appliance("Midea", "ACX1234", "room_air_conditioners",
                                               path, False)
            literal = literal_result["sources"]["energy_star"]
            self.assertEqual(literal["status"], "exact_match")
            self.assertEqual(literal["candidates"][0]["match_type"], "literal")
            self.assertEqual(literal["candidates"][0]["source"]["dataset_id"], "5xn2-dv4h")
            self.assertEqual(literal["candidates"][0]["source"]["record_id"], "star-literal")
            self.assertEqual(literal["candidates"][0]["source"]["url"], DATASET.dataset_url)
            self.assertEqual(literal["annual_energy_consumption"], 400.0)
            self.assertIn("750 compressor operating hours", literal["annual_energy_context"])

            family_result = lookup_appliance("Midea", "ACX1235", "room_air_conditioners",
                                              path, False)
            family = family_result["sources"]["energy_star"]
            self.assertEqual(family["status"], "exact_match")
            self.assertEqual(family["candidates"][0]["match_type"], "wildcard")
            self.assertEqual(family["annual_energy_consumption"], 300.0)

    def test_multiple_wildcard_records_remain_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [room_record("family-1", "AC*123#"),
                           room_record("family-2", "ACX12##", annual="500")])
            result = lookup_appliance("Midea", "ACX1235", "room_air_conditioners", path, False)
            star = result["sources"]["energy_star"]
            self.assertEqual(star["status"], "ambiguous_match")
            self.assertIsNone(star["annual_energy_consumption"])
            self.assertEqual(len(star["candidates"]), 2)

    def test_missing_annual_energy_does_not_get_inferred_from_ceer(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            record = room_record("star-no-energy", "MISSING1", annual=None, ceer="14.2")
            make_db(path, [record], nrcan_records=[(1, "Midea", "MISSING1", 12.5)])
            result = lookup_appliance("Midea", "MISSING1", "room_air_conditioners", path, False)

            star = result["sources"]["energy_star"]
            self.assertEqual(star["status"], "exact_match")
            self.assertIsNone(star["annual_energy_consumption"])
            self.assertIsNone(star["unit"])
            self.assertFalse(star["energy_consumption_available"])
            self.assertEqual(star["candidates"][0]["energy_metrics"], [
                {"name": "CEER", "value": "14.2", "unit": "Btu/Wh"},
                {"name": "Cooling capacity", "value": "8000", "unit": "Btu/hr"},
                {"name": "Less energy than U.S. federal standard", "value": "38", "unit": "%"},
            ])
            self.assertIsNone(result["annual_energy_consumption"])
            self.assertEqual(result["sources"]["nrcan"]["status"], "exact_match")
            self.assertIsNone(result["sources"]["nrcan"]["annual_energy_consumption"])

    def test_candidate_validation_and_pipeline_and_cli_report_energy_star_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [room_record("star-target", "ACX1234", annual="400"),
                           room_record("star-no-energy", "MISSING1", annual=None)],
                    nrcan_records=[(1, "Midea", "MISSING1", 12.5)])
            parsed = {"brand": {"value": "Midea"},
                      "model_number": {"value": "ACX123A", "normalized": "ACX123A"}}
            candidate = next(item for item in candidates_from_parse(
                parsed, category="room_air_conditioners", db_path=path
            )["candidates"] if item["normalized"] == "ACX1234")
            self.assertTrue(candidate["db_match"])
            self.assertEqual(candidate["db_sources"], ["energy_star"])

            result = lookup_appliance("Midea", "MISSING1", "room_air_conditioners", path, False)
            payload = analyze_label(model_number="MISSING1", manufacturer="Midea",
                                    category="room_air_conditioners", db_path=path,
                                    allow_fuzzy=False)
            self.assertEqual(payload["retrieval"]["sources"]["energy_star"]["status"],
                             "exact_match")
            self.assertIsNone(payload["retrieval"]["annual_energy_consumption"])

            pipeline_output = io.StringIO()
            with contextlib.redirect_stdout(pipeline_output):
                _print_report(payload)
            self.assertIn("ENERGY STAR: exact_match", pipeline_output.getvalue())
            self.assertIn("750 compressor operating hours", pipeline_output.getvalue())
            self.assertIn("CEER: 15 Btu/Wh", pipeline_output.getvalue())
            self.assertIn("Energy:     unavailable", pipeline_output.getvalue())
            self.assertIn("record star-no-energy", pipeline_output.getvalue())
            self.assertIn(DATASET.dataset_url, pipeline_output.getvalue())

            cli_output = io.StringIO()
            with contextlib.redirect_stdout(cli_output):
                display_result(result)
            self.assertIn("dataset 5xn2-dv4h", cli_output.getvalue())
            self.assertIn("750 compressor operating hours", cli_output.getvalue())
            self.assertIn("CEER: 15 Btu/Wh", cli_output.getvalue())
            self.assertIn("Annual energy consumption: Not reported", cli_output.getvalue())
            self.assertIn("record star-no-energy", cli_output.getvalue())
            self.assertIn(DATASET.dataset_url, cli_output.getvalue())

            star_only_output = io.StringIO()
            with contextlib.redirect_stdout(star_only_output):
                display_result(lookup_appliance("Midea", "ACX1234",
                                                 "room_air_conditioners", path, False))
            self.assertIn("ENERGY STAR: exact_match", star_only_output.getvalue())
            self.assertIn("No NRCan match found", star_only_output.getvalue())

    def test_failed_second_page_does_not_replace_existing_room_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, [room_record("old-record", "OLD1")])
            first_page = mock.MagicMock()
            first_page.__enter__.return_value = first_page
            first_page.__exit__.return_value = False
            first_page.read.return_value = json.dumps(
                [room_record("new-record", "NEW1")]
            ).encode()
            with mock.patch.object(refresh, "urlopen", side_effect=[first_page, OSError("offline")]):
                with self.assertRaises(OSError):
                    refresh.download_records("room_air_conditioners", page_size=1)
            with sqlite3.connect(path) as conn:
                models = conn.execute(
                    "SELECT model_number FROM energy_star_room_air_conditioners"
                ).fetchall()
            self.assertEqual(models, [("OLD1",)])


if __name__ == "__main__":
    unittest.main()
