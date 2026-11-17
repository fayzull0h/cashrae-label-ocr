"""Tests for the Stage 4 end-to-end evaluation harness (``scripts/evaluate_pipeline.py``)."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from scripts import evaluate_pipeline
from scripts.evaluate_pipeline import (
    _accuracy,
    _confidence_counts,
    _rate,
    evaluate_database,
    evaluate_ground_truth,
)


def _result(
    *,
    status: str = "exact_match",
    model: str = "PCKS433ET",
    parsed_model: str | None = None,
    energy: float | None = 430.0,
    available: bool = True,
    from_original: bool = False,
    confidence: str | None = "medium",
    attempts: int = 2,
) -> dict:
    return {
        "status": status,
        "parsed": {"model_number": {"normalized": parsed_model or model}},
        "retrieval": {
            "model_used": model,
            "from_original": from_original,
            "annual_energy_consumption": energy,
            "energy_consumption_available": available,
        },
        "confidence": confidence,
        "attempts": [{}] * attempts,
    }


class TestMetricHelpers(unittest.TestCase):
    def test_accuracy_handles_zero_total(self):
        self.assertEqual(_accuracy(0, 0), {"matched": 0, "total": 0, "accuracy": 0.0})

    def test_accuracy_rounds(self):
        self.assertEqual(_accuracy(1, 3)["accuracy"], 0.3333)

    def test_rate_counts_matching_cases(self):
        cases = [{"ok": True}, {"ok": False}, {"ok": True}]
        self.assertEqual(_rate(cases, lambda case: case["ok"]), _accuracy(2, 3))

    def test_confidence_counts_are_sorted_by_frequency(self):
        counts = _confidence_counts([{"confidence": "low"}, {"confidence": "high"}, {"confidence": "low"}])
        self.assertEqual(counts, {"low": 2, "high": 1})


class TestGroundTruthSection(unittest.TestCase):
    def _write_fixtures(self, tmp: Path) -> tuple[Path, Path]:
        truth = {
            "plate.png": {
                "category": "clothes_dryers",
                "brand": "GE",
                "model_number": "PCKS433ET",
                "serial_number": "123",
            },
            "missing.png": {
                "category": "dishwashers",
                "brand": "X",
                "model_number": "ABC123",
                "serial_number": "1",
            },
        }
        dump = [
            {
                "image_path": "data/samples/plate.png",
                "raw_text": "GE\nMODEL: PCKS433ET",
            }
        ]
        truth_path = tmp / "ground_truth.json"
        dump_path = tmp / "dump.json"
        truth_path.write_text(json.dumps(truth), encoding="utf-8")
        dump_path.write_text(json.dumps(dump), encoding="utf-8")
        return truth_path, dump_path

    def test_skips_images_without_fixture_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth_path, dump_path = self._write_fixtures(Path(tmp))
            with mock.patch.object(
                evaluate_pipeline, "analyze_label", return_value=_result()
            ):
                payload = evaluate_ground_truth(truth_path, dump_path, allow_fuzzy=True)

        self.assertEqual(payload["summary"]["cases"], 1)
        self.assertEqual(payload["cases"][0]["image"], "plate.png")
        self.assertTrue(payload["cases"][0]["model_correct"])
        self.assertTrue(payload["summary"]["honest_miss"]["total"] == 1)

    def test_honest_miss_counts_unavailable_energy(self):
        miss = _result(
            status="not_found", energy=None, available=False, confidence=None
        )
        with tempfile.TemporaryDirectory() as tmp:
            truth_path, dump_path = self._write_fixtures(Path(tmp))
            with mock.patch.object(evaluate_pipeline, "analyze_label", return_value=miss):
                payload = evaluate_ground_truth(truth_path, dump_path, allow_fuzzy=False)

        summary = payload["summary"]
        self.assertEqual(summary["honest_miss"]["matched"], 1)
        self.assertEqual(summary["energy_value_returned"]["matched"], 0)
        self.assertEqual(summary["status_counts"], {"not_found": 1})
        self.assertFalse(payload["allow_fuzzy"])

    def test_wrong_parse_is_reported(self):
        wrong = _result(parsed_model="NOTTHEMODEL")
        with tempfile.TemporaryDirectory() as tmp:
            truth_path, dump_path = self._write_fixtures(Path(tmp))
            with mock.patch.object(evaluate_pipeline, "analyze_label", return_value=wrong):
                payload = evaluate_ground_truth(truth_path, dump_path, allow_fuzzy=True)

        self.assertFalse(payload["cases"][0]["model_correct"])
        self.assertEqual(payload["summary"]["model_parsed_correctly"]["matched"], 0)


class TestDatabaseSection(unittest.TestCase):
    ROWS = [
        {"model_number": "PCKS433ET", "manufacturer": "GE", "category": "clothes_dryers"},
    ]

    def _run(self, results: list[dict]) -> dict:
        with mock.patch.object(
            evaluate_pipeline, "sample_database_models", return_value=self.ROWS
        ), mock.patch.object(
            evaluate_pipeline,
            "find_exact_matches",
            return_value=[{"annual_energy_consumption": 430.0, "energy_consumption_available": True}],
        ), mock.patch.object(
            evaluate_pipeline, "corruptions", return_value=[
                {"position": 8, "from": "T", "to": "7", "corrupted": "PCKS433E7"}
            ]
        ), mock.patch.object(
            evaluate_pipeline, "analyze_label", side_effect=results
        ):
            return evaluate_database("unused.db", samples=1, seed=0, max_corruptions=1, allow_fuzzy=True)

    def test_correct_recovery_and_energy(self):
        payload = self._run([_result()])

        summary = payload["summary"]
        self.assertEqual(summary["cases"], 1)
        self.assertEqual(summary["exact_match"]["matched"], 1)
        self.assertEqual(summary["model_recovered"]["matched"], 1)
        self.assertEqual(summary["energy_value_correct"]["matched"], 1)
        self.assertEqual(summary["wrong_energy_value"]["matched"], 0)
        self.assertEqual(summary["used_a_correction"]["matched"], 1)
        self.assertEqual(summary["avg_attempts"], 2.0)

    def test_wrong_energy_is_counted_separately(self):
        payload = self._run([_result(energy=999.0)])

        self.assertEqual(payload["summary"]["energy_value_correct"]["matched"], 0)
        self.assertEqual(payload["summary"]["wrong_energy_value"]["matched"], 1)

    def test_ambiguous_result_is_not_a_wrong_value(self):
        ambiguous = _result(
            status="ambiguous_match", energy=None, available=False, confidence=None
        )
        payload = self._run([ambiguous])

        summary = payload["summary"]
        self.assertEqual(summary["exact_match"]["matched"], 0)
        self.assertEqual(summary["wrong_energy_value"]["matched"], 0)
        self.assertEqual(summary["energy_value_correct"]["matched"], 0)

    def test_models_missing_from_the_database_are_skipped(self):
        with mock.patch.object(
            evaluate_pipeline, "sample_database_models", return_value=self.ROWS
        ), mock.patch.object(evaluate_pipeline, "find_exact_matches", return_value=[]):
            payload = evaluate_database(
                "unused.db", samples=1, seed=0, max_corruptions=1, allow_fuzzy=True
            )

        self.assertEqual(payload["skipped"], ["PCKS433ET"])
        self.assertEqual(payload["summary"]["cases"], 0)


class TestCli(unittest.TestCase):
    def setUp(self):
        # Keep the CLI tests off the database: the fixture loading and report
        # plumbing are what matter here, and live retrieval is covered by
        # ``tests/test_label_pipeline.py``.
        self._patcher = mock.patch.object(
            evaluate_pipeline,
            "analyze_label",
            return_value=_result(status="not_found", energy=None, available=False, confidence=None),
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_no_db_json_run_uses_the_tracked_fixture(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            exit_code = evaluate_pipeline.main(["--no-db", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertIn("ground_truth", payload)
        self.assertNotIn("database", payload)
        self.assertEqual(payload["ground_truth"]["summary"]["cases"], 6)

    def test_text_report_lists_each_plate(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            exit_code = evaluate_pipeline.main(["--no-db"])

        report = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Verified plates", report)
        self.assertIn("airconditioner_label.png", report)

    def test_output_file_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pipeline_eval.json"
            with redirect_stdout(StringIO()):
                exit_code = evaluate_pipeline.main(
                    ["--no-db", "--json", "--output", str(output)]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["ground_truth"]["summary"]["cases"], 6)

    def test_missing_ground_truth_returns_error_code(self):
        with redirect_stdout(StringIO()), mock.patch("sys.stderr", new_callable=StringIO):
            exit_code = evaluate_pipeline.main(["--ground-truth", "nope.json", "--no-db"])

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
