"""Focused tests for ENERGY STAR washer refresh and cross-source lookup."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from scripts import refresh_energy_star_washers as refresh
from scripts.appliance_lookup import lookup_appliance
from scripts.label_pipeline import analyze_label


def star_record(record_id="1", model="WT8400C*", imef=2.76):
    return {"pd_id": record_id, "brand_name": "LG", "model_number": model,
            "additional_model_information": "family note", "upc": "123",
            "special_type": "", "integrated_modified_energy_factor_imef": imef,
            "annual_energy_use_kwh_year": "120"}


def make_db(path, *, nrcan_model=None, aec=111.0, star_models=()):
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE clothes_washers (
            REC_REF_KEY INTEGER, BRAND_NAME TEXT, MODEL_NUM_1 TEXT, MOD_J1_AEC REAL,
            source_file TEXT, appliance_category TEXT)""")
        if nrcan_model:
            conn.execute("INSERT INTO clothes_washers VALUES (1,'LG',?,?,?,?)",
                         (nrcan_model, aec, "fixture.csv", "clothes_washers"))
        conn.execute("""CREATE TABLE energy_star_clothes_washers (
            record_id TEXT PRIMARY KEY, source_url TEXT, snapshot_date TEXT,
            brand_name TEXT, model_number TEXT, additional_model_information TEXT,
            upc TEXT, special_type TEXT, raw_fields_json TEXT)""")
        for index, record in enumerate(star_models, 1):
            conn.execute("INSERT INTO energy_star_clothes_washers VALUES (?,?,?,?,?,?,?,?,?)",
                         (str(index), refresh.DATASET_URL, "2026-11-17", "LG",
                          record["model_number"], record.get("additional_model_information"),
                          record.get("upc"), record.get("special_type"), json.dumps(record)))


class RefreshTests(unittest.TestCase):
    def test_paginated_download_and_required_fields(self):
        pages = [[star_record()], []]
        responses = [mock.MagicMock(), mock.MagicMock()]
        for response, page in zip(responses, pages):
            response.__enter__.return_value = response
            response.__exit__.return_value = False
            response.read.return_value = json.dumps(page).encode()
        with mock.patch.object(refresh, "PAGE_SIZE", 1), mock.patch.object(
            refresh, "urlopen", side_effect=responses
        ):
            records = refresh.download_records()
        self.assertEqual(len(records), 1)

    def test_failed_download_does_not_replace_existing_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, star_models=[star_record()])
            with mock.patch.object(refresh, "urlopen", side_effect=OSError("offline")):
                with self.assertRaises(OSError):
                    refresh.download_records()
            with sqlite3.connect(path) as conn:
                self.assertEqual(conn.execute(
                    "SELECT model_number FROM energy_star_clothes_washers"
                ).fetchone()[0], "WT8400C*")

    def test_snapshot_replace_stores_source_and_refresh_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, star_models=[star_record(model="OLD")])
            date = refresh.replace_snapshot([star_record()], path)
            with sqlite3.connect(path) as conn:
                row = conn.execute("SELECT record_id, source_url, snapshot_date, raw_fields_json "
                                   "FROM energy_star_clothes_washers").fetchone()
                metadata = conn.execute("SELECT refresh_date, row_count "
                                        "FROM energy_star_refresh_metadata").fetchone()
            self.assertEqual(row[:3], ("1", refresh.DATASET_URL, date))
            self.assertEqual(json.loads(row[3])["model_number"], "WT8400C*")
            self.assertEqual(metadata, (date, 1))


class LookupAndPipelineTests(unittest.TestCase):
    def test_wildcard_literal_and_additional_info_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, star_models=[star_record()])
            result = lookup_appliance("LG", "WT8400CW", "clothes_washers", path, False)
            star = result["sources"]["energy_star"]
            self.assertEqual(star["status"], "exact_match")
            self.assertEqual(star["candidates"][0]["match_type"], "wildcard")
            no_family_parse = lookup_appliance("LG", "family note", "clothes_washers", path, False)
            self.assertEqual(no_family_parse["sources"]["energy_star"]["status"], "not_found")

    def test_ambiguous_family_matches_are_not_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, star_models=[star_record("1", "WT8400C*"),
                                       star_record("2", "WT8400C*")])
            result = lookup_appliance("LG", "WT8400CW", "clothes_washers", path, False)
            star = result["sources"]["energy_star"]
            self.assertEqual(star["status"], "ambiguous_match")
            self.assertEqual(len(star["candidates"]), 2)

    def test_both_source_values_remain_separate_and_pipeline_reports_star(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, nrcan_model="WT8400CW", aec=111.0,
                    star_models=[star_record()])
            result = lookup_appliance("LG", "WT8400CW", "clothes_washers", path, False)
            self.assertEqual(result["annual_energy_consumption"], 111.0)
            self.assertEqual(result["sources"]["nrcan"]["annual_energy_consumption"], 111.0)
            self.assertEqual(result["sources"]["energy_star"]["annual_energy_consumption"], 120.0)
            self.assertTrue(result["energy_conflict"])
            self.assertEqual(result["sources"]["energy_star"]["candidates"][0][
                "energy_metric"]["value"], 2.76)
            pipeline = analyze_label(model_number="WT8400CW", manufacturer="LG",
                                     category="clothes_washers", db_path=path,
                                     allow_fuzzy=False)
            self.assertEqual(pipeline["retrieval"]["sources"]["energy_star"]["status"],
                             "exact_match")
            self.assertTrue(pipeline["retrieval"]["energy_conflict"])


if __name__ == "__main__":
    unittest.main()
