"""Focused tests for ENERGY STAR washer refresh and cross-source lookup."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from scripts import refresh_energy_star_washers as refresh
from scripts import refresh_energy_star as shared_refresh
from scripts.energy_star_config import ENERGY_STAR_DATASETS
from scripts.appliance_lookup import lookup_appliance
from scripts.label_candidates import candidates_from_parse
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

    def test_shared_refresh_replaces_each_configured_snapshot_with_source_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            for category, dataset in ENERGY_STAR_DATASETS.items():
                record = {"pd_id": f"id-{dataset.dataset_id}", "brand_name": "Fixture",
                          "model_number": f"MODEL-{category}", dataset.annual_energy_field: "123",
                          "custom_source_field": "preserved"}
                date = shared_refresh.replace_snapshot([record], category, path)
                with sqlite3.connect(path) as conn:
                    row = conn.execute(
                        f"SELECT record_id, source_url, snapshot_date, raw_fields_json "
                        f"FROM {dataset.table}"
                    ).fetchone()
                    metadata = conn.execute(
                        "SELECT dataset_id, source_url, refresh_date, row_count "
                        "FROM energy_star_refresh_metadata WHERE dataset_id=?",
                        (dataset.dataset_id,),
                    ).fetchone()
                self.assertEqual(row[:3], (record["pd_id"], dataset.dataset_url, date))
                self.assertEqual(json.loads(row[3])["custom_source_field"], "preserved")
                self.assertEqual(metadata, (dataset.dataset_id, dataset.dataset_url, date, 1))

    def test_shared_refresh_rejects_invalid_snapshot_without_replacing_existing_table(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            old_record = {"pd_id": "old", "brand_name": "Fixture", "model_number": "OLD"}
            dataset = ENERGY_STAR_DATASETS["dishwashers"]
            shared_refresh.replace_snapshot([old_record], "dishwashers", path)
            with self.assertRaises(ValueError):
                shared_refresh.replace_snapshot([], "dishwashers", path)
            with sqlite3.connect(path) as conn:
                self.assertEqual(conn.execute(
                    f"SELECT model_number FROM {dataset.table}"
                ).fetchone()[0], "OLD")

    def test_shared_download_paginates_and_validates_category_schema(self):
        record = {"pd_id": "dish-1", "brand_name": "Fixture", "model_number": "D-1"}
        pages = [[record], []]
        responses = [mock.MagicMock(), mock.MagicMock()]
        for response, page in zip(responses, pages):
            response.__enter__.return_value = response
            response.__exit__.return_value = False
            response.read.return_value = json.dumps(page).encode()
        with mock.patch.object(shared_refresh, "urlopen", side_effect=responses):
            records = shared_refresh.download_records("dishwashers", page_size=1)
        self.assertEqual(records, [record])


class GeneralizedLookupTests(unittest.TestCase):
    def test_category_fields_metrics_and_provenance_are_configured(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            for category, dataset in ENERGY_STAR_DATASETS.items():
                model = f"MODEL{category.upper()}"
                metric_values = {metric.field: "42" for metric in dataset.metrics}
                record = {"pd_id": f"id-{dataset.dataset_id}", "brand_name": "Fixture",
                          "model_number": model, dataset.annual_energy_field: "321",
                          "custom_source_field": "preserved", **metric_values}
                shared_refresh.replace_snapshot([record], category, path)
                nrcan_energy = {"clothes_washers": "MOD_J1_AEC",
                                "room_air_conditioners": "EE_RATIO",
                                "refrigerators": "AEC", "dishwashers": "TAEC_C373_04"}[category]
                with sqlite3.connect(path) as conn:
                    conn.execute(
                        f"CREATE TABLE IF NOT EXISTS {category} (REC_REF_KEY INTEGER, "
                        f"BRAND_NAME TEXT, MODEL_NUM_1 TEXT, {nrcan_energy} REAL, "
                        "source_file TEXT, appliance_category TEXT)"
                    )
                result = lookup_appliance("Fixture", model, category, path, False)
                star = result["sources"]["energy_star"]
                self.assertEqual(star["dataset_id"], dataset.dataset_id)
                self.assertEqual(star["status"], "exact_match")
                self.assertEqual(star["annual_energy_consumption"], 321.0)
                candidate = star["candidates"][0]
                self.assertEqual(candidate["source"]["record_id"], record["pd_id"])
                self.assertEqual(candidate["source"]["url"], dataset.dataset_url)
                self.assertEqual(candidate["record"]["custom_source_field"], "preserved")
                self.assertEqual(candidate["energy_metrics"][0]["value"], "42")
                self.assertEqual(result["annual_energy_consumption"], None)

    def test_energy_star_exact_hit_validates_ocr_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            dataset = ENERGY_STAR_DATASETS["refrigerators"]
            record = {"pd_id": "star-1", "brand_name": "Fixture", "model_number": "ABC1234",
                      "annual_energy_use_kwh_yr": "200"}
            shared_refresh.replace_snapshot([record], "refrigerators", path)
            with sqlite3.connect(path) as conn:
                conn.execute("CREATE TABLE refrigerators (REC_REF_KEY INTEGER, BRAND_NAME TEXT, "
                             "MODEL_NUM_1 TEXT, AEC REAL, source_file TEXT, appliance_category TEXT)")
            parsed = {"brand": {"value": "Fixture"},
                      "model_number": {"value": "ABC123A", "normalized": "ABC123A"}}
            payload = candidates_from_parse(parsed, category="refrigerators", db_path=path)
            confirmed = next(candidate for candidate in payload["candidates"]
                             if candidate["normalized"] == "ABC1234")
            self.assertTrue(confirmed["db_match"])
            self.assertEqual(confirmed["db_sources"], ["energy_star"])


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

    def test_literal_match_takes_precedence_over_matching_wildcard(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.db"
            make_db(path, star_models=[star_record("1", "WT8400C*"),
                                       star_record("2", "WT8400CW")])
            result = lookup_appliance("LG", "WT8400CW", "clothes_washers", path, False)
            star = result["sources"]["energy_star"]
            self.assertEqual(star["status"], "exact_match")
            self.assertEqual(star["candidates"][0]["model_number"], "WT8400CW")
            self.assertEqual(star["candidates"][0]["match_type"], "literal")

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
