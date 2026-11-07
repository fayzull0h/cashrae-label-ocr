"""Integration and Unit Test Suite for NRCan Appliance Retrieval Engine.

Covers all 12 test requirements specified in Phase 1:
1. Exact manufacturer and model match.
2. Exact model match with different capitalization.
3. Model match with spaces or hyphens.
4. Model-only search without a manufacturer.
5. Fuzzy model-number matching.
6. Multiple possible matches (ambiguous).
7. Model number that does not exist.
8. Invalid or empty input.
9. Appliance found but energy consumption missing.
10. Searching within a specified appliance category.
11. Searching across all categories.
12. Categories with different energy-consumption field names.
"""

from pathlib import Path
import unittest

from scripts.appliance_lookup import (
    get_annual_energy_consumption,
    lookup_appliance,
    normalize_category,
    normalize_manufacturer,
    normalize_model_number,
)

DB_PATH = Path("database/appliances.db")


class TestApplianceLookup(unittest.TestCase):
    """Test suite for appliance retrieval functionality."""

    @classmethod
    def setUpClass(cls):
        """Ensure database exists before running integration tests."""
        if not DB_PATH.is_file():
            raise FileNotFoundError(f"Database not found at {DB_PATH.resolve()}")

    # -------------------------------------------------------------------------
    # 1. Exact manufacturer and model match
    # -------------------------------------------------------------------------
    def test_01_exact_manufacturer_and_model_match(self):
        """Test exact match when both manufacturer and model are provided."""
        res = lookup_appliance(
            manufacturer="GE",
            model_number="PCKS433ET#**",
            category="clothes_dryers",
            db_path=DB_PATH,
        )
        self.assertEqual(res["status"], "exact_match")
        self.assertTrue(res["is_exact"])
        self.assertEqual(res["match_confidence"], 1.0)
        self.assertEqual(res["manufacturer"], "GE")
        self.assertEqual(res["category"], "clothes_dryers")
        self.assertEqual(res["annual_energy_consumption"], 430.0)
        self.assertEqual(res["unit"], "kWh/year")
        self.assertTrue(res["energy_consumption_available"])
        self.assertEqual(res["source"]["table"], "clothes_dryers")

    # -------------------------------------------------------------------------
    # 2. Exact model match with different capitalization
    # -------------------------------------------------------------------------
    def test_02_exact_model_match_different_capitalization(self):
        """Test exact match with lowercase and mixed-case inputs."""
        res_lower = lookup_appliance(
            manufacturer="ge",
            model_number="pcks433et#**",
            category="clothes_dryers",
            db_path=DB_PATH,
        )
        self.assertEqual(res_lower["status"], "exact_match")
        self.assertEqual(res_lower["annual_energy_consumption"], 430.0)

        res_mixed = lookup_appliance(
            manufacturer="Ge",
            model_number="PcKs433Et#**",
            category="clothes_dryers",
            db_path=DB_PATH,
        )
        self.assertEqual(res_mixed["status"], "exact_match")
        self.assertEqual(res_mixed["annual_energy_consumption"], 430.0)

    # -------------------------------------------------------------------------
    # 3. Model match with spaces or hyphens
    # -------------------------------------------------------------------------
    def test_03_model_match_with_spaces_or_hyphens(self):
        """Test that hyphenated, spaced, and underscored inputs normalize to match."""
        variants = [
            "PCKS-433-ET",
            "pcks 433 et",
            "PCKS_433_ET",
            "  pcks-433-et  ",
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                res = lookup_appliance(
                    manufacturer="GE",
                    model_number=variant,
                    category="clothes_dryers",
                    db_path=DB_PATH,
                )
                self.assertEqual(res["status"], "exact_match", f"Failed for {variant}")
                self.assertEqual(res["annual_energy_consumption"], 430.0)

    # -------------------------------------------------------------------------
    # 4. Model-only search without a manufacturer
    # -------------------------------------------------------------------------
    def test_04_model_only_search_without_manufacturer(self):
        """Test model-only search where manufacturer is omitted."""
        # F1051 is a unique model in freezers (Liebherr)
        res = lookup_appliance(
            manufacturer=None,
            model_number="F1051",
            db_path=DB_PATH,
        )
        self.assertEqual(res["status"], "exact_match")
        self.assertEqual(res["manufacturer"], "Liebherr")
        self.assertEqual(res["category"], "freezers")
        self.assertEqual(res["annual_energy_consumption"], 465.0)
        self.assertIn("warning", res)
        self.assertIn("Manufacturer was not specified", res["warning"])

    # -------------------------------------------------------------------------
    # 5. Fuzzy model-number matching
    # -------------------------------------------------------------------------
    def test_05_fuzzy_model_number_matching(self):
        """Test fuzzy matching with minor typing/OCR error in model number."""
        # Target model is PCKS433ET#**; search with missing last letter 'T'
        res = lookup_appliance(
            manufacturer="GE",
            model_number="PCKS433E",
            category="clothes_dryers",
            db_path=DB_PATH,
        )
        self.assertEqual(res["status"], "fuzzy_match")
        self.assertFalse(res["is_exact"])
        self.assertGreater(res["match_confidence"], 0.85)
        self.assertEqual(res["model_number"], "PCKS433ET#**")
        self.assertEqual(res["annual_energy_consumption"], 430.0)
        self.assertIn("candidates", res)

    # -------------------------------------------------------------------------
    # 6. Multiple possible matches
    # -------------------------------------------------------------------------
    def test_06_multiple_possible_matches(self):
        """Test that multiple matching candidates return ambiguous status rather than picking one arbitrarily."""
        # Master Chef model 043-0288-4 has multiple records in refrigerators
        res = lookup_appliance(
            manufacturer="Master Chef",
            model_number="043-0288-4",
            db_path=DB_PATH,
        )
        self.assertEqual(res["status"], "ambiguous_match")
        self.assertIsNone(res["annual_energy_consumption"])
        self.assertGreaterEqual(len(res["candidates"]), 2)
        for cand in res["candidates"]:
            self.assertEqual(cand["manufacturer"], "Master Chef")

    # -------------------------------------------------------------------------
    # 7. Model number that does not exist
    # -------------------------------------------------------------------------
    def test_07_model_number_does_not_exist(self):
        """Test searching for a non-existent model number."""
        res = lookup_appliance(
            manufacturer="GE",
            model_number="COMPLETELY_FABRICATED_MODEL_99999",
            db_path=DB_PATH,
        )
        self.assertEqual(res["status"], "not_found")
        self.assertIsNone(res["annual_energy_consumption"])
        self.assertFalse(res["energy_consumption_available"])
        self.assertEqual(res["match_confidence"], 0.0)

    # -------------------------------------------------------------------------
    # 8. Invalid or empty input
    # -------------------------------------------------------------------------
    def test_08_invalid_or_empty_input(self):
        """Test handling of None, empty strings, whitespace, and non-alphanumeric inputs."""
        invalid_inputs = ["", None, "   ", "---", "***###"]
        for inp in invalid_inputs:
            with self.subTest(inp=inp):
                res = lookup_appliance(
                    manufacturer="GE",
                    model_number=inp,
                    db_path=DB_PATH,
                )
                self.assertEqual(res["status"], "invalid_input")
                self.assertIsNone(res["annual_energy_consumption"])

        # Test invalid category
        res_cat = lookup_appliance(
            manufacturer="GE",
            model_number="PCKS433ET#**",
            category="microwave_ovens",
            db_path=DB_PATH,
        )
        self.assertEqual(res_cat["status"], "invalid_input")
        self.assertIn("Invalid appliance category", res_cat["message"])

    # -------------------------------------------------------------------------
    # 9. Appliance found but energy consumption missing
    # -------------------------------------------------------------------------
    def test_09_appliance_found_but_energy_consumption_missing(self):
        """Test finding an appliance record when annual energy consumption is not reported."""
        # Case A: Room air conditioners do not have an annual energy consumption field
        res_ac = lookup_appliance(
            manufacturer="Wallmate",
            model_number="SCA09LS",
            category="room_air_conditioners",
            db_path=DB_PATH,
        )
        self.assertEqual(res_ac["status"], "exact_match")
        self.assertIsNone(res_ac["annual_energy_consumption"])
        self.assertFalse(res_ac["energy_consumption_available"])
        self.assertIsNone(res_ac["unit"])
        self.assertIn("notes", res_ac)
        self.assertIn("does not report Annual Energy Consumption", res_ac["notes"])

        # Case B: clothes_washer_dryers row with NULL AEC
        res_cwd = lookup_appliance(
            manufacturer="LG",
            model_number="WM3455H*",
            category="clothes_washer_dryers",
            db_path=DB_PATH,
        )
        self.assertEqual(res_cwd["status"], "exact_match")
        self.assertIsNone(res_cwd["annual_energy_consumption"])
        self.assertFalse(res_cwd["energy_consumption_available"])
        self.assertIn("notes", res_cwd)

    # -------------------------------------------------------------------------
    # 10. Searching within a specified appliance category
    # -------------------------------------------------------------------------
    def test_10_searching_within_specified_category(self):
        """Test restricting search to a specific category."""
        # Kenmore 253.22052* is in both freezers and refrigerators.
        # Restricting to 'freezers' should yield an exact unique match.
        res = lookup_appliance(
            manufacturer="Kenmore",
            model_number="253.22052*",
            category="freezers",
            db_path=DB_PATH,
        )
        self.assertEqual(res["status"], "exact_match")
        self.assertEqual(res["category"], "freezers")
        self.assertEqual(res["annual_energy_consumption"], 534.0)

    # -------------------------------------------------------------------------
    # 11. Searching across all categories
    # -------------------------------------------------------------------------
    def test_11_searching_across_all_categories(self):
        """Test searching across all categories when category is omitted."""
        # Kenmore 253.22052* across all categories should be flagged as ambiguous
        res = lookup_appliance(
            manufacturer="Kenmore",
            model_number="253.22052*",
            category=None,
            db_path=DB_PATH,
        )
        self.assertEqual(res["status"], "ambiguous_match")
        matched_categories = {c["category"] for c in res["candidates"]}
        self.assertIn("freezers", matched_categories)
        self.assertIn("refrigerators", matched_categories)

    # -------------------------------------------------------------------------
    # 12. Categories with different energy-consumption field names
    # -------------------------------------------------------------------------
    def test_12_categories_with_different_energy_field_names(self):
        """Test retrieval across categories that use distinct energy column names."""
        test_cases = [
            # clothes_washers uses MOD_J1_AEC
            ("Staber", "HXW2304", "clothes_washers", 180.0),
            # cooktops uses AEC_03
            ("Jenn-Air", "CVEX4270", "cooktops", 235.0),
            # dishwashers uses TAEC_C373_04
            ("Smeg", "PLA8743XU", "dishwashers", 298.0),
            # refrigerators uses AEC
            ("Daewoo", "FR-091", "refrigerators", 318.0),
        ]

        for mfg, model, cat, expected_aec in test_cases:
            with self.subTest(category=cat):
                # Test lookup_appliance
                res = lookup_appliance(
                    manufacturer=mfg,
                    model_number=model,
                    category=cat,
                    db_path=DB_PATH,
                )
                self.assertEqual(res["status"], "exact_match")
                self.assertEqual(res["annual_energy_consumption"], expected_aec)
                self.assertEqual(res["unit"], "kWh/year")

                # Test get_annual_energy_consumption reuses lookup
                aec_res = get_annual_energy_consumption(
                    manufacturer=mfg,
                    model_number=model,
                    category=cat,
                    db_path=DB_PATH,
                )
                self.assertEqual(aec_res["status"], "exact_match")
                self.assertEqual(aec_res["annual_energy_consumption"], expected_aec)


if __name__ == "__main__":
    unittest.main()
