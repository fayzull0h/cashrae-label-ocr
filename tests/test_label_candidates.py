"""Tests for Phase 2 Stage 3 OCR-error candidate generation."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from scripts import label_candidates
from scripts.appliance_lookup import find_exact_matches, find_exact_matches_batch
from scripts.label_candidates import (
    CONFUSION_MAP,
    candidates_from_parse,
    generate_candidates,
    rank_candidates,
)
from scripts.label_parse import parse_fields

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Stable fixture row: ``PCKS433ET#**`` / GE in the clothes dryers table.
KNOWN_MODEL = "PCKS433ET"
KNOWN_BRAND = "GE"
KNOWN_CATEGORY = "clothes_dryers"


def _stub_validator(hits: set[str]):
    """Return a validator that reports a hit for every model in ``hits``."""

    def validator(models, manufacturer, category):
        return {
            model: ([{"category": category or KNOWN_CATEGORY}] if model in hits else [])
            for model in models
        }

    return validator


class TestConfusionMap(unittest.TestCase):
    def test_map_is_symmetric(self):
        for character, alternatives in CONFUSION_MAP.items():
            for alternative in alternatives:
                self.assertIn(
                    character,
                    CONFUSION_MAP[alternative],
                    msg=f"{alternative!r} should map back to {character!r}",
                )

    def test_no_self_mapping(self):
        for character, alternatives in CONFUSION_MAP.items():
            self.assertNotIn(character, alternatives)

    def test_known_pairs_are_present(self):
        self.assertIn("O", CONFUSION_MAP["0"])
        self.assertIn("0", CONFUSION_MAP["O"])
        self.assertIn("I", CONFUSION_MAP["1"])
        self.assertIn("L", CONFUSION_MAP["1"])
        self.assertIn("B", CONFUSION_MAP["8"])
        self.assertIn("S", CONFUSION_MAP["5"])

    def test_map_is_uppercase_alphanumeric(self):
        for character, alternatives in CONFUSION_MAP.items():
            self.assertTrue(character.isalnum(), msg=character)
            self.assertEqual(character, character.upper())
            for alternative in alternatives:
                self.assertTrue(alternative.isalnum(), msg=alternative)
                self.assertEqual(alternative, alternative.upper())


class TestGenerateCandidates(unittest.TestCase):
    def test_empty_input_returns_empty_list(self):
        self.assertEqual(generate_candidates(None), [])
        self.assertEqual(generate_candidates(""), [])
        self.assertEqual(generate_candidates("---"), [])

    def test_original_is_first_and_unchanged(self):
        candidates = generate_candidates("pcks-433-et#**")

        original = candidates[0]
        self.assertTrue(original["is_original"])
        self.assertEqual(original["model_number"], KNOWN_MODEL)
        self.assertEqual(original["normalized"], KNOWN_MODEL)
        self.assertEqual(original["edit_count"], 0)
        self.assertEqual(original["edits"], [])
        self.assertEqual(original["rank"], None)

    def test_known_single_substitutions_are_generated(self):
        generated = {candidate["normalized"] for candidate in generate_candidates("DLE8400WE")}

        self.assertIn("D1E8400WE", generated)  # L -> 1
        self.assertIn("DIE8400WE", generated)  # L -> I
        self.assertIn("DLEB400WE", generated)  # 8 -> B
        self.assertIn("DLE840OWE", generated)  # 0 -> O
        self.assertIn("DLE84O0WE", generated)  # 0 -> O

    def test_candidates_are_deterministic(self):
        first = generate_candidates("PCKS433ET")
        second = generate_candidates("PCKS433ET")
        self.assertEqual(first, second)

    def test_candidates_are_unique(self):
        candidates = generate_candidates("DLE8400WE")
        normalized = [candidate["normalized"] for candidate in candidates]
        self.assertEqual(len(normalized), len(set(normalized)))

    def test_candidate_cap_includes_original(self):
        candidates = generate_candidates("DLE8400WE", max_candidates=3)

        self.assertEqual(len(candidates), 3)
        self.assertTrue(candidates[0]["is_original"])

    def test_cap_of_one_returns_only_original(self):
        candidates = generate_candidates("DLE8400WE", max_candidates=1)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["is_original"])

    def test_two_edits_are_opt_in(self):
        one_edit = generate_candidates("DLE8400WE", max_edits=1)
        two_edit = generate_candidates("DLE8400WE", max_edits=2)

        self.assertTrue(all(candidate["edit_count"] <= 1 for candidate in one_edit))
        self.assertTrue(any(candidate["edit_count"] == 2 for candidate in two_edit))

    def test_invalid_limits_raise(self):
        with self.assertRaises(ValueError):
            generate_candidates("ABC123", max_edits=3)
        with self.assertRaises(ValueError):
            generate_candidates("ABC123", max_edits=-1)
        with self.assertRaises(ValueError):
            generate_candidates("ABC123", max_candidates=0)

    def test_class_fit_prefers_matching_neighbours(self):
        candidates = {candidate["normalized"]: candidate for candidate in generate_candidates("DLE8400WE")}

        # L -> 1 sits between two letters: a poor class fit.
        self.assertEqual(candidates["D1E8400WE"]["edits"][0]["class_fit"], 0.0)
        # 0 -> O sits inside a digit run: also a poor class fit.
        self.assertEqual(candidates["DLE84O0WE"]["edits"][0]["class_fit"], 0.0)

    def test_first_correction_has_highest_plausibility(self):
        candidates = generate_candidates("PCKS433ET")
        plausibilities = [candidate["plausibility"] for candidate in candidates[1:]]
        self.assertEqual(plausibilities, sorted(plausibilities, reverse=True))


class TestRankCandidates(unittest.TestCase):
    def test_empty_input_returns_empty_list(self):
        self.assertEqual(rank_candidates([], validate=False), [])

    def test_validate_false_marks_db_match_unknown(self):
        ranked = rank_candidates(generate_candidates("PCKS433ET"), validate=False)

        self.assertTrue(all(candidate["db_match"] is None for candidate in ranked))
        self.assertTrue(all(candidate["db_match_count"] is None for candidate in ranked))
        self.assertTrue(all(candidate["db_categories"] == [] for candidate in ranked))

    def test_ranks_are_sequential_and_original_is_first(self):
        ranked = rank_candidates(generate_candidates("DLE8400WE"), validate=False)

        self.assertEqual([candidate["rank"] for candidate in ranked], list(range(1, len(ranked) + 1)))
        self.assertTrue(ranked[0]["is_original"])

    def test_original_stays_first_even_when_a_correction_hits(self):
        generated = generate_candidates("DLE8400WE", max_candidates=5)
        correction = generated[1]["normalized"]
        ranked = rank_candidates(
            generated,
            validate=True,
            validator=_stub_validator({correction}),
        )

        self.assertTrue(ranked[0]["is_original"])
        self.assertEqual(ranked[1]["normalized"], correction)
        self.assertTrue(ranked[1]["db_match"])

    def test_db_confirmed_ranks_above_unconfirmed(self):
        generated = generate_candidates("DLE8400WE", max_candidates=6)
        correction = generated[3]["normalized"]
        ranked = rank_candidates(
            generated,
            validate=True,
            validator=_stub_validator({correction}),
        )

        confirmed = [candidate for candidate in ranked if candidate["db_match"]]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["normalized"], correction)
        self.assertEqual(confirmed[0]["rank"], 2)  # right after the original

    def test_validator_receives_lookup_context(self):
        seen: dict[str, object] = {}

        def validator(models, manufacturer, category):
            seen["models"] = models
            seen["manufacturer"] = manufacturer
            seen["category"] = category
            return {model: [] for model in models}

        rank_candidates(
            generate_candidates("PCKS433ET"),
            manufacturer=KNOWN_BRAND,
            category=KNOWN_CATEGORY,
            validate=True,
            validator=validator,
        )

        self.assertIn(KNOWN_MODEL, seen["models"])
        self.assertEqual(seen["manufacturer"], KNOWN_BRAND)
        self.assertEqual(seen["category"], KNOWN_CATEGORY)

    def test_ranking_does_not_mutate_input(self):
        generated = generate_candidates("PCKS433ET")
        snapshot = json.dumps(generated, sort_keys=True)
        rank_candidates(generated, validate=False)
        self.assertEqual(json.dumps(generated, sort_keys=True), snapshot)


class TestExactLookup(unittest.TestCase):
    def test_known_model_is_found(self):
        hits = find_exact_matches(KNOWN_MODEL, category=KNOWN_CATEGORY)

        self.assertTrue(hits)
        self.assertTrue(all(hit["normalized_model"] == KNOWN_MODEL for hit in hits))
        self.assertTrue(all(hit["category"] == KNOWN_CATEGORY for hit in hits))

    def test_brand_filter_is_applied(self):
        self.assertTrue(
            find_exact_matches(KNOWN_MODEL, manufacturer=KNOWN_BRAND, category=KNOWN_CATEGORY)
        )
        self.assertEqual(
            find_exact_matches(KNOWN_MODEL, manufacturer="Not A Brand", category=KNOWN_CATEGORY),
            [],
        )

    def test_one_edit_variant_is_not_an_exact_match(self):
        self.assertEqual(find_exact_matches("PCKS433E7", category=KNOWN_CATEGORY), [])

    def test_empty_model_returns_empty(self):
        self.assertEqual(find_exact_matches(""), [])
        self.assertEqual(find_exact_matches(None), [])

    def test_invalid_category_raises(self):
        with self.assertRaises(ValueError):
            find_exact_matches(KNOWN_MODEL, category="not_a_category")

    def test_batch_keys_cover_every_requested_model(self):
        matches = find_exact_matches_batch(
            [KNOWN_MODEL, "NOPE123"],
            category=KNOWN_CATEGORY,
        )

        self.assertEqual(set(matches), {KNOWN_MODEL, "NOPE123"})
        self.assertTrue(matches[KNOWN_MODEL])
        self.assertEqual(matches["NOPE123"], [])

    def test_batch_normalizes_keys(self):
        matches = find_exact_matches_batch(["pcks-433-et"], category=KNOWN_CATEGORY)
        self.assertTrue(matches[KNOWN_MODEL])

    def test_batch_empty_input_returns_empty(self):
        self.assertEqual(find_exact_matches_batch([]), {})


class TestCandidatesFromParse(unittest.TestCase):
    def test_uses_parsed_model_and_brand(self):
        parsed = parse_fields("BRAND: LG\nMODEL: DLE8400WE\nSERIAL: 503KWVQ6K991")
        payload = candidates_from_parse(parsed, validate=False)

        self.assertEqual(payload["original"]["value"], "DLE8400WE")
        self.assertEqual(payload["original"]["normalized"], "DLE8400WE")
        self.assertEqual(payload["manufacturer"], "LG")
        self.assertFalse(payload["validated"])
        self.assertTrue(payload["candidates"][0]["is_original"])
        self.assertTrue(any(c["normalized"] == "D1E8400WE" for c in payload["candidates"]))

    def test_missing_model_returns_no_candidates(self):
        parsed = parse_fields("ACME APPLIANCES INC.\n120V 60Hz")
        payload = candidates_from_parse(parsed, validate=False)

        self.assertIsNone(payload["original"]["value"])
        self.assertEqual(payload["original"]["normalized"], "")
        self.assertEqual(payload["candidates"], [])

    def test_explicit_manufacturer_overrides_parsed_brand(self):
        parsed = parse_fields("BRAND: LG\nMODEL: DLE8400WE")
        payload = candidates_from_parse(parsed, manufacturer="Whirlpool", validate=False)

        self.assertEqual(payload["manufacturer"], "Whirlpool")


class TestCli(unittest.TestCase):
    def test_json_output(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            exit_code = label_candidates.main(
                ["--model", KNOWN_MODEL, "--no-validate", "--json"]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["original"]["normalized"], KNOWN_MODEL)
        self.assertFalse(payload["validated"])
        self.assertTrue(payload["candidates"][0]["is_original"])
        self.assertIsNone(payload["candidates"][0]["db_match"])

    def test_text_report_marks_the_original(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            exit_code = label_candidates.main(["--model", KNOWN_MODEL, "--no-validate"])

        self.assertEqual(exit_code, 0)
        self.assertIn("(original)", buffer.getvalue())

    def test_output_file_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "candidates.json"
            with redirect_stdout(StringIO()):
                exit_code = label_candidates.main(
                    ["--model", KNOWN_MODEL, "--no-validate", "--json", "--output", str(output)]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["original"]["normalized"], KNOWN_MODEL)

    def test_empty_model_returns_error_code(self):
        with redirect_stdout(StringIO()), mock.patch("sys.stderr", new_callable=StringIO):
            exit_code = label_candidates.main(["--model=---"])

        self.assertEqual(exit_code, 1)

    def test_invalid_category_returns_error_code(self):
        with redirect_stdout(StringIO()), mock.patch("sys.stderr", new_callable=StringIO):
            exit_code = label_candidates.main(
                ["--model", KNOWN_MODEL, "--category", "not_a_category"]
            )

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
