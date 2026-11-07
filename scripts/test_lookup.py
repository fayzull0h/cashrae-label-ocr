"""Command-line demonstration script for NRCan Appliance Retrieval Engine.

Allows interactive input of manufacturer and model number, or automated demonstration
against actual database records.

Usage:
    # Interactive mode:
    python scripts/test_lookup.py

    # Command-line arguments:
    python scripts/test_lookup.py --manufacturer "GE" --model "PCKS-433-ET"
    python scripts/test_lookup.py -m "F1051"

    # Built-in demonstration:
    python scripts/test_lookup.py --demo
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.appliance_lookup import (
    CATEGORY_DISPLAY_NAMES,
    DEFAULT_DB_PATH,
    lookup_appliance,
)


def display_result(result: dict) -> None:
    """Format and print a lookup result to stdout."""
    status = result.get("status")

    if status == "invalid_input":
        print("\n[!] Invalid input:")
        print(f"    {result.get('message')}")
        return

    if status == "not_found":
        print("\nNo match found.")
        print(f"Message: {result.get('message')}")
        return

    if status == "ambiguous_match":
        print(f"\n[?] Ambiguous match: {result.get('message')}")
        candidates = result.get("candidates", [])
        print(f"    Found {len(candidates)} candidate records:")
        for idx, cand in enumerate(candidates, 1):
            aec = cand.get("annual_energy_consumption")
            unit = cand.get("unit") or ""
            energy_str = f"{aec} {unit}" if aec is not None else "Not reported"
            cat_disp = cand.get("category_display", cand.get("category"))
            conf = cand.get("match_confidence", 0.0)
            print(f"    {idx}. [{cat_disp}] {cand.get('manufacturer')} - {cand.get('model_number')} (Energy: {energy_str}, Confidence: {conf:.2f})")
        return

    if status in ("exact_match", "fuzzy_match"):
        print("\nMatch found!\n")
        mfg = result.get("manufacturer")
        model = result.get("model_number")
        cat_disp = result.get("category_display", result.get("category"))
        aec = result.get("annual_energy_consumption")
        unit = result.get("unit") or ""
        match_type = "Exact" if result.get("is_exact") else f"Fuzzy (confidence: {result.get('match_confidence', 0.0):.2f})"

        print(f"Manufacturer:              {mfg}")
        print(f"Model:                     {model}")
        print(f"Category:                  {cat_disp}")
        if aec is not None:
            print(f"Annual energy consumption: {aec} {unit}")
        else:
            print(f"Annual energy consumption: Not reported (or N/A)")
            if result.get("notes"):
                print(f"Notes:                     {result.get('notes')}")
        print(f"Match type:                {match_type}")
        if result.get("warning"):
            print(f"Notice:                    {result.get('warning')}")


def run_demo() -> None:
    """Run demonstration with actual NRCan records."""
    print("=" * 65)
    print("NRCan Appliance Retrieval Engine Demonstration")
    print("=" * 65)

    demos = [
        {
            "desc": "1. Exact match with punctuation and lowercase (Clothes dryer)",
            "mfg": "ge",
            "model": "pcks-433-et",
            "cat": "clothes_dryers",
        },
        {
            "desc": "2. Exact match using different energy column (Dishwasher - TAEC_C373_04)",
            "mfg": "Smeg",
            "model": "PLA8743XU",
            "cat": "dishwashers",
        },
        {
            "desc": "3. Model-only search without manufacturer (Freezer)",
            "mfg": None,
            "model": "F1051",
            "cat": None,
        },
        {
            "desc": "4. Fuzzy match with typing error (missing last character 'T')",
            "mfg": "GE",
            "model": "PCKS433E",
            "cat": "clothes_dryers",
        },
        {
            "desc": "5. Appliance found but energy consumption not reported (Room AC)",
            "mfg": "Wallmate",
            "model": "SCA09LS",
            "cat": "room_air_conditioners",
        },
        {
            "desc": "6. Ambiguous match across categories (Freezers & Refrigerators)",
            "mfg": "Kenmore",
            "model": "253.22052*",
            "cat": None,
        },
    ]

    for d in demos:
        print("\n" + "-" * 65)
        print(f"Demo {d['desc']}")
        print(f"Input: Manufacturer='{d['mfg']}', Model='{d['model']}', Category='{d['cat']}'")
        res = lookup_appliance(
            manufacturer=d["mfg"],
            model_number=d["model"],
            category=d["cat"],
        )
        display_result(res)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="NRCan Appliance Retrieval CLI")
    parser.add_argument("-m", "--model", help="Model number to search for")
    parser.add_argument("-b", "--manufacturer", help="Brand/manufacturer name (optional)")
    parser.add_argument("-c", "--category", help="Appliance category (optional)")
    parser.add_argument("--demo", action="store_true", help="Run automated demonstration")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to appliances.db")

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    # If arguments provided on CLI
    if args.model:
        print(f"Manufacturer: {args.manufacturer or '(Not specified)'}")
        print(f"Model number: {args.model}")
        if args.category:
            print(f"Category:     {args.category}")
        print("\nSearching NRCan database...")
        res = lookup_appliance(
            manufacturer=args.manufacturer,
            model_number=args.model,
            category=args.category,
            db_path=args.db,
        )
        display_result(res)
        return

    # Interactive prompt mode
    print("=" * 55)
    print("NRCan Appliance Energy Retrieval (Interactive Mode)")
    print("=" * 55)

    try:
        mfg_input = input("Manufacturer (optional, press Enter to skip): ").strip()
        manufacturer = mfg_input if mfg_input else None

        model = input("Model number: ").strip()

        cat_input = input("Category (optional, press Enter for all): ").strip()
        category = cat_input if cat_input else None

        print("\nSearching NRCan database...")
        res = lookup_appliance(
            manufacturer=manufacturer,
            model_number=model,
            category=category,
            db_path=args.db,
        )
        display_result(res)

    except (KeyboardInterrupt, EOFError):
        print("\nLookup cancelled.")
        sys.exit(0)


if __name__ == "__main__":
    main()
