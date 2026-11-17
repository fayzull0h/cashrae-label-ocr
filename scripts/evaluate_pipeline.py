"""End-to-end pipeline evaluation (Phase 2, Stage 4).

Measures the whole chain — text → parse → candidate search → NRCan retrieval →
energy value — rather than one stage at a time.

Two corpora:

1. **Verified plates.** The tracked PaddleOCR fixture (``data/processed/ocr_dump.json``)
   is fed straight into the pipeline for the six hand-verified plates. Those models are
   not present in the NRCan data, so the useful signals here are parse accuracy and
   whether the pipeline reports the miss honestly instead of inventing an energy value.
2. **Database-derived cases.** Real NRCan models are corrupted with a single realistic
   OCR character swap and pushed through the pipeline. Because the true model *is* in the
   database, this measures end-to-end recovery: does the pipeline find the right model and
   return the right kWh/year value?

OCR itself is not exercised here (the fixture supplies text); use
``scripts/evaluate_parse.py`` for OCR + parsing accuracy.

Typical usage::

    python scripts/evaluate_pipeline.py
    python scripts/evaluate_pipeline.py --no-db
    python scripts/evaluate_pipeline.py --db-samples 20 --json
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import argparse
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow direct execution as ``python scripts/evaluate_pipeline.py``.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.appliance_lookup import (  # noqa: E402
    DEFAULT_DB_PATH,
    find_exact_matches,
    normalize_alphanum,
)
from scripts.evaluate_candidates import (  # noqa: E402
    _limit,
    corruptions,
    sample_database_models,
)
from scripts.label_pipeline import analyze_label  # noqa: E402

DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "data" / "samples" / "ground_truth.json"
DEFAULT_DUMP = PROJECT_ROOT / "data" / "processed" / "ocr_dump.json"
DEFAULT_DB_SAMPLES = 30
DEFAULT_MAX_CORRUPTIONS = 4
DEFAULT_SEED = 0


def _accuracy(matched: int, total: int) -> dict[str, Any]:
    return {
        "matched": matched,
        "total": total,
        "accuracy": round(matched / total, 4) if total else 0.0,
    }


def _rate(cases: list[dict[str, Any]], predicate) -> dict[str, Any]:
    return _accuracy(sum(1 for case in cases if predicate(case)), len(cases))


def _confidence_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(case["confidence"]) for case in cases).most_common())


def evaluate_ground_truth(
    ground_truth_path: Path,
    dump_path: Path,
    *,
    allow_fuzzy: bool,
) -> dict[str, Any]:
    truth = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    dump = json.loads(Path(dump_path).read_text(encoding="utf-8"))
    texts = {Path(entry["image_path"]).name: entry["raw_text"] for entry in dump}

    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    for name, expected in sorted(truth.items()):
        raw_text = texts.get(name)
        if raw_text is None:
            continue
        expected_model = normalize_alphanum(expected.get("model_number"))
        result = analyze_label(
            raw_text=raw_text,
            category=expected.get("category"),
            allow_fuzzy=allow_fuzzy,
        )
        parsed_model = result["parsed"]["model_number"]["normalized"]
        retrieval = result.get("retrieval") or {}
        cases.append(
            {
                "image": name,
                "expected_model": expected_model,
                "parsed_model": parsed_model,
                "model_correct": parsed_model == expected_model,
                "status": result["status"],
                "model_used": retrieval.get("model_used"),
                "from_original": retrieval.get("from_original"),
                "energy_available": bool(retrieval.get("energy_consumption_available")),
                "annual_energy_consumption": retrieval.get("annual_energy_consumption"),
                "confidence": result.get("confidence"),
                "attempts": len(result.get("attempts") or []),
            }
        )

    return {
        "source": str(ground_truth_path),
        "dump": str(dump_path),
        "allow_fuzzy": allow_fuzzy,
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "model_parsed_correctly": _rate(cases, lambda case: case["model_correct"]),
            "found_in_database": _rate(
                cases, lambda case: case["status"] in ("exact_match", "fuzzy_match")
            ),
            "energy_value_returned": _rate(cases, lambda case: case["energy_available"]),
            "honest_miss": _rate(
                cases,
                lambda case: case["status"] in ("not_found", "ambiguous_match", "no_model_found")
                and not case["energy_available"],
            ),
            "status_counts": dict(Counter(case["status"] for case in cases).most_common()),
            "confidence_counts": _confidence_counts(cases),
        },
        "latency_s": round(time.perf_counter() - started, 3),
    }


def evaluate_database(
    db_path: Path | str,
    *,
    samples: int,
    seed: int,
    max_corruptions: int,
    allow_fuzzy: bool,
) -> dict[str, Any]:
    selected = sample_database_models(db_path, count=samples, seed=seed)
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    skipped: list[str] = []

    for row in selected:
        true_model = row["model_number"]
        hits = find_exact_matches(true_model, category=row["category"], db_path=db_path)
        if not hits:
            skipped.append(true_model)
            continue
        expected_energy = hits[0].get("annual_energy_consumption")
        expected_available = bool(hits[0].get("energy_consumption_available"))

        for corruption in _limit(corruptions(true_model), max_corruptions):
            raw_text = f"{row['manufacturer']}\nMODEL: {corruption['corrupted']}"
            result = analyze_label(
                raw_text=raw_text,
                category=row["category"],
                allow_fuzzy=allow_fuzzy,
                db_path=db_path,
            )
            retrieval = result.get("retrieval") or {}
            energy = retrieval.get("annual_energy_consumption")
            cases.append(
                {
                    "category": row["category"],
                    "true_model": true_model,
                    "corrupted": corruption["corrupted"],
                    "parsed_model": result["parsed"]["model_number"]["normalized"],
                    "status": result["status"],
                    "model_used": retrieval.get("model_used"),
                    "model_recovered": retrieval.get("model_used") == true_model,
                    "from_original": retrieval.get("from_original"),
                    "expected_energy": expected_energy,
                    "annual_energy_consumption": energy,
                    "energy_correct": (
                        energy == expected_energy if expected_available else energy is None
                    ),
                    "confidence": result.get("confidence"),
                    "attempts": len(result.get("attempts") or []),
                }
            )

    def is_exact(case: dict[str, Any]) -> bool:
        return case["status"] == "exact_match"

    def wrong_energy(case: dict[str, Any]) -> bool:
        energy = case["annual_energy_consumption"]
        return energy is not None and energy != case["expected_energy"]

    return {
        "source": str(db_path),
        "samples": selected,
        "skipped": skipped,
        "allow_fuzzy": allow_fuzzy,
        "cases": cases,
        "summary": {
            "models": len(selected) - len(skipped),
            "cases": len(cases),
            "exact_match": _rate(cases, is_exact),
            "fuzzy_match": _rate(cases, lambda case: case["status"] == "fuzzy_match"),
            "model_recovered": _rate(cases, lambda case: case["model_recovered"]),
            "energy_value_correct": _rate(cases, lambda case: case["energy_correct"]),
            "wrong_energy_value": _rate(cases, wrong_energy),
            "used_a_correction": _rate(cases, lambda case: case["from_original"] is False),
            "confidence_counts": _confidence_counts(cases),
            "avg_attempts": round(
                sum(case["attempts"] for case in cases) / len(cases), 2
            )
            if cases
            else 0.0,
        },
        "latency_s": round(time.perf_counter() - started, 3),
    }


def _print_rate(label: str, section: dict[str, Any]) -> None:
    print(
        f"  {label:26} {section['matched']}/{section['total']} "
        f"({section['accuracy'] * 100:.1f}%)"
    )


def _print_ground_truth(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print(f"  plates:       {summary['cases']}")
    print(f"  fuzzy allowed:{' yes' if payload['allow_fuzzy'] else ' no'}")
    _print_rate("model parsed correctly", summary["model_parsed_correctly"])
    _print_rate("found in database", summary["found_in_database"])
    _print_rate("energy value returned", summary["energy_value_returned"])
    _print_rate("honest miss (no value)", summary["honest_miss"])
    print(f"  statuses:     {summary['status_counts']}")
    print(f"  confidence:   {summary['confidence_counts']}")
    print(f"  latency:      {payload['latency_s']:.2f}s")
    print()
    print("  Per plate:")
    for case in payload["cases"]:
        mark = "ok " if case["model_correct"] else "BAD"
        print(
            f"    [{mark}] {case['image']:28} {case['expected_model']:16} "
            f"-> {case['status']:16} energy={case['annual_energy_consumption']}"
        )


def _print_database(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print(f"  models:       {summary['models']} (skipped {len(payload['skipped'])})")
    print(f"  cases:        {summary['cases']}")
    print(f"  fuzzy allowed:{' yes' if payload['allow_fuzzy'] else ' no'}")
    _print_rate("exact match", summary["exact_match"])
    _print_rate("fuzzy match", summary["fuzzy_match"])
    _print_rate("true model recovered", summary["model_recovered"])
    _print_rate("energy value correct", summary["energy_value_correct"])
    _print_rate("wrong energy value", summary["wrong_energy_value"])
    _print_rate("used a correction", summary["used_a_correction"])
    print(f"  avg attempts: {summary['avg_attempts']:.2f}")
    print(f"  confidence:   {summary['confidence_counts']}")
    print(f"  latency:      {payload['latency_s']:.2f}s")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Evaluate the label → energy pipeline end to end"
    )
    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH),
        help="Ground-truth JSON keyed by image filename",
    )
    parser.add_argument(
        "--dump",
        default=str(DEFAULT_DUMP),
        help="Tracked OCR raw-text fixture used in place of live OCR",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the SQLite database")
    parser.add_argument(
        "--db-samples",
        type=int,
        default=DEFAULT_DB_SAMPLES,
        help=f"Database models to corrupt for the end-to-end section (default: {DEFAULT_DB_SAMPLES})",
    )
    parser.add_argument(
        "--max-corruptions",
        type=int,
        default=DEFAULT_MAX_CORRUPTIONS,
        help=f"Corruptions scored per model (default: {DEFAULT_MAX_CORRUPTIONS})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed")
    parser.add_argument(
        "--no-fuzzy",
        dest="allow_fuzzy",
        action="store_false",
        help="Disable the pipeline's fuzzy fallback",
    )
    parser.add_argument("--no-db", action="store_true", help="Skip the database-derived section")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report")
    parser.add_argument("--output", help="Write JSON results to this file")
    args = parser.parse_args(argv)

    try:
        payload: dict[str, Any] = {
            "ground_truth": evaluate_ground_truth(
                Path(args.ground_truth),
                Path(args.dump),
                allow_fuzzy=args.allow_fuzzy,
            )
        }
        if not args.no_db:
            payload["database"] = evaluate_database(
                args.db,
                samples=args.db_samples,
                seed=args.seed,
                max_corruptions=args.max_corruptions,
                allow_fuzzy=args.allow_fuzzy,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {output_path}")

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("=" * 78)
        print("Label → energy pipeline — end-to-end evaluation")
        print("=" * 78)
        print()
        print("Verified plates (tracked OCR fixture, parsed and retrieved)")
        _print_ground_truth(payload["ground_truth"])
        if "database" in payload:
            print()
            print("Database models (one OCR character swap, real energy values)")
            _print_database(payload["database"])
        print()

    total = payload["ground_truth"]["summary"]["cases"] + payload.get("database", {}).get(
        "summary", {}
    ).get("cases", 0)
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
