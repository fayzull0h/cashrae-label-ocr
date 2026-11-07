"""Tests for the OCR comparison harness (``scripts/compare_ocr.py``).

These tests do not require PaddleOCR; the harness imports the Paddle backend
lazily, so the metric and reporting helpers are testable in the main venv.
"""

from __future__ import annotations

import unittest

from scripts.compare_ocr import (
    _summarize,
    _to_markdown,
    best_window_similarity,
    token_metrics,
)


class TestBestWindowSimilarity(unittest.TestCase):
    def test_exact_match_scores_one(self):
        self.assertEqual(best_window_similarity("NXAGB3OGKA100", "XXNXAGB3OGKA100YY"), 1.0)

    def test_near_miss_scores_high(self):
        score = best_window_similarity("NXAGB3OGKA100", "NXAGB3OGKA1OO")

        self.assertGreater(score, 0.85)
        self.assertLess(score, 1.0)

    def test_empty_inputs_score_zero(self):
        self.assertEqual(best_window_similarity("", "ABC"), 0.0)
        self.assertEqual(best_window_similarity("ABC", ""), 0.0)

    def test_unrelated_text_scores_low(self):
        self.assertLess(best_window_similarity("ABC123", "ZZZZZZZZZZ"), 0.3)


class TestTokenMetrics(unittest.TestCase):
    def test_scores_only_populated_fields(self):
        metrics = token_metrics(
            "MODEL NUMBER NXAGB3OGKA100 SERIAL NUMBER 302785",
            {"brand": None, "model_number": "NXAGB3OGKA100", "serial_number": "302785"},
        )

        self.assertEqual(set(metrics["fields"]), {"model_number", "serial_number"})
        self.assertEqual(metrics["exact_recall"], 1.0)
        self.assertEqual(metrics["mean_similarity"], 1.0)

    def test_partial_match_reduces_recall(self):
        metrics = token_metrics(
            "MODEL NXAGB3OGKA1OO SERIAL 302785",
            {"model_number": "NXAGB3OGKA100", "serial_number": "302785"},
        )

        self.assertFalse(metrics["fields"]["model_number"]["exact"])
        self.assertTrue(metrics["fields"]["serial_number"]["exact"])
        self.assertEqual(metrics["exact_recall"], 0.5)

    def test_no_ground_truth_returns_none_metrics(self):
        metrics = token_metrics("anything", {"brand": None})

        self.assertEqual(metrics["fields"], {})
        self.assertIsNone(metrics["exact_recall"])


class TestReporting(unittest.TestCase):
    def _payload(self):
        return {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "backend_errors": {},
            "images": [
                {
                    "image": "label.png",
                    "variants": {
                        "raw": {
                            "tesseract": {
                                "confidence": 0.7,
                                "latency_s": 1.2,
                                "metrics": {"exact_recall": 0.5, "mean_similarity": 0.8},
                            },
                            "paddleocr": {
                                "confidence": 0.9,
                                "latency_s": 2.0,
                                "metrics": {"exact_recall": 1.0, "mean_similarity": 1.0},
                            },
                        }
                    },
                }
            ],
        }

    def test_summarize_flattens_results(self):
        rows = _summarize(self._payload())

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["engine"] for row in rows}, {"tesseract", "paddleocr"})

    def test_markdown_contains_aggregate_table(self):
        payload = self._payload()
        markdown = _to_markdown(payload, _summarize(payload))

        self.assertIn("# OCR comparison", markdown)
        self.assertIn("| paddleocr | raw | 1.000 |", markdown)
        self.assertIn("| tesseract | raw | 0.500 |", markdown)


if __name__ == "__main__":
    unittest.main()
