"""Tests for the parsing evaluation harness (``scripts/evaluate_parse.py``)."""

from __future__ import annotations

import unittest

from scripts.evaluate_parse import _matches, _summarize


def _field(value: str | None, normalized: str | None) -> dict:
    return {"value": value, "normalized": normalized}


class TestFieldMatching(unittest.TestCase):
    def test_exact_normalized_match(self):
        self.assertTrue(_matches(_field("ABC-123", "ABC123"), "ABC 123"))

    def test_missing_value_does_not_match(self):
        self.assertFalse(_matches(_field(None, None), "ABC123"))

    def test_serial_prefix_is_tolerated(self):
        self.assertTrue(
            _matches(_field("E140302785", "E140302785"), "140302785", serial=True)
        )

    def test_serial_prefix_not_tolerated_for_models(self):
        self.assertFalse(_matches(_field("E140302785", "E140302785"), "140302785"))

    def test_wrong_serial_does_not_match(self):
        self.assertFalse(
            _matches(_field("E140302785", "E140302785"), "140302786", serial=True)
        )


class TestSummary(unittest.TestCase):
    @staticmethod
    def _result(brand: bool = True, model: bool = True, serial: bool = True) -> dict:
        return {
            "fields": {
                "brand": {"match": brand},
                "model_number": {"match": model},
                "serial_number": {"match": serial},
            },
            "all_match": brand and model and serial,
        }

    def test_accuracy_counts_per_field_and_overall(self):
        summary = _summarize([self._result(), self._result(serial=False)])

        self.assertEqual(summary["images"], 2)
        self.assertEqual(summary["fields"]["brand"]["accuracy"], 1.0)
        self.assertEqual(summary["fields"]["serial_number"]["matched"], 1)
        self.assertEqual(summary["fields"]["serial_number"]["total"], 2)
        self.assertEqual(summary["fields"]["serial_number"]["accuracy"], 0.5)
        self.assertEqual(summary["all_match"]["matched"], 1)
        self.assertEqual(summary["all_match"]["accuracy"], 0.5)

    def test_empty_results_report_zero_accuracy(self):
        summary = _summarize([])

        self.assertEqual(summary["images"], 0)
        self.assertEqual(summary["all_match"]["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
