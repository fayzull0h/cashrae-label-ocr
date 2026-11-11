"""Candidate-generation evaluation (Phase 2, Stage 3).

Measures how well :mod:`scripts.label_candidates` recovers the true model number
after a single realistic OCR character swap.

Two corpora are used:

1. **Ground-truth plates** — the six hand-verified models in
   ``data/samples/ground_truth.json``, corrupted with every single confusion
   substitution at every position. These models are not present in the NRCan
   database, so this section scores string ranking only (no validation).
2. **Database models** — a deterministic sample of real NRCan model numbers,
   corrupted the same way and ranked with exact database validation. This is the
   signal that matters for Stage 4: after validation, does the true model
   surface as the top database-confirmed candidate?

Typical usage::

    python scripts/evaluate_candidates.py
    python scripts/evaluate_candidates.py --db-samples 20 --json
    python scripts/evaluate_candidates.py --no-db
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import argparse
import json
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow direct execution as ``python scripts/evaluate_candidates.py``.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.appliance_lookup import (  # noqa: E402
    ALL_APPLIANCE_TABLES,
    DEFAULT_DB_PATH,
    get_db_connection,
    normalize_alphanum,
)
from scripts.label_candidates import (  # noqa: E402
    CONFUSION_MAP,
    DEFAULT_MAX_CANDIDATES,
    generate_candidates,
    rank_candidates,
)

DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "data" / "samples" / "ground_truth.json"
DEFAULT_DB_SAMPLES = 40
DEFAULT_SEED = 0
DEFAULT_MAX_CORRUPTIONS = 25


def corruptions(model_number: str) -> list[dict[str, Any]]:
    """Every single confusion substitution that can be applied to ``model_number``."""
    normalized = normalize_alphanum(model_number)
    cases: list[dict[str, Any]] = []
    for position, character in enumerate(normalized):
        for replacement in CONFUSION_MAP.get(character, ()):
            cases.append(
                {
                    "position": position,
                    "from": character,
                    "to": replacement,
                    "corrupted": normalized[:position] + replacement + normalized[position + 1 :],
                }
            )
    return cases


def _limit(items: list[Any], limit: int) -> list[Any]:
    """Deterministically spread ``limit`` picks across ``items``."""
    if limit <= 0:
        return []
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


def _score_cases(cases: list[dict[str, Any]], k: int) -> dict[str, Any]:
    total = len(cases)
    if not total:
        return {
            "k": k,
            "cases": 0,
            "recall_at_1": {"matched": 0, "total": 0, "accuracy": 0.0},
            "recall_at_5": {"matched": 0, "total": 0, "accuracy": 0.0},
            f"recall_at_{k}": {"matched": 0, "total": 0, "accuracy": 0.0},
            "mrr": 0.0,
            "avg_candidates": 0.0,
            "original_first_violations": 0,
        }

    def recall(limit: int) -> dict[str, Any]:
        matched = sum(
            1
            for case in cases
            if case["correction_rank"] is not None and case["correction_rank"] <= limit
        )
        return {"matched": matched, "total": total, "accuracy": round(matched / total, 4)}

    return {
        "k": k,
        "cases": total,
        "recall_at_1": recall(1),
        "recall_at_5": recall(5),
        f"recall_at_{k}": recall(k),
        "mrr": round(
            sum(1 / case["correction_rank"] for case in cases if case["correction_rank"]) / total,
            4,
        ),
        "avg_candidates": round(
            sum(case["candidate_count"] for case in cases) / total,
            2,
        ),
        "original_first_violations": sum(1 for case in cases if not case["original_first"]),
    }


def build_validation_index(db_path: Path | str) -> dict[str, list[dict[str, str]]]:
    """Load every normalized model -> (brand, category) once, for fast validation."""
    conn = get_db_connection(db_path)
    index: dict[str, list[dict[str, str]]] = {}
    try:
        for table in ALL_APPLIANCE_TABLES:
            for row in conn.execute(f'SELECT MODEL_NUM_1, BRAND_NAME FROM "{table}"'):
                model = normalize_alphanum(row["MODEL_NUM_1"])
                if not model:
                    continue
                index.setdefault(model, []).append(
                    {
                        "manufacturer": normalize_alphanum(row["BRAND_NAME"]),
                        "category": table,
                    }
                )
    finally:
        conn.close()
    return index


def make_validator(index: dict[str, list[dict[str, str]]]):
    """Wrap a prebuilt index in the ``rank_candidates`` validator signature."""

    def validator(
        models: list[str],
        manufacturer: str | None,
        category: str | None,
    ) -> dict[str, list[dict[str, str]]]:
        brand = normalize_alphanum(manufacturer)
        matches: dict[str, list[dict[str, str]]] = {model: [] for model in models}
        for model in models:
            for hit in index.get(model, []):
                if category and hit["category"] != category:
                    continue
                if brand and hit["manufacturer"] != brand:
                    continue
                matches[model].append(hit)
        return matches

    return validator


def _rank_case(
    corrupted: str,
    true_model: str,
    *,
    manufacturer: str | None,
    category: str | None,
    validate: bool,
    max_candidates: int,
    validator=None,
) -> dict[str, Any]:
    generated = generate_candidates(corrupted, max_candidates=max_candidates)
    ranked = rank_candidates(
        generated,
        manufacturer=manufacturer,
        category=category,
        validate=validate,
        validator=validator,
    )
    true_normalized = normalize_alphanum(true_model)
    true_candidate = next(
        (candidate for candidate in ranked if candidate["normalized"] == true_normalized),
        None,
    )
    rank = true_candidate["rank"] if true_candidate else None
    corrections = [candidate for candidate in ranked if not candidate["is_original"]]
    correction_rank = next(
        (
            position
            for position, candidate in enumerate(corrections, start=1)
            if candidate["normalized"] == true_normalized
        ),
        None,
    )
    db_hits = [candidate for candidate in ranked if candidate["db_match"]]
    top_db = db_hits[0] if db_hits else None

    return {
        "true_model": true_normalized,
        "corrupted": corrupted,
        "rank": rank,
        "correction_rank": correction_rank,
        "candidate_count": len(ranked),
        "original_first": bool(ranked) and ranked[0]["is_original"],
        "db_hit_count": len(db_hits) if validate else None,
        "true_confirmed": rank is not None and any(
            candidate["normalized"] == true_normalized and candidate["db_match"]
            for candidate in ranked
        )
        if validate
        else None,
        "true_is_top_db": bool(top_db and top_db["normalized"] == true_normalized)
        if validate
        else None,
    }


def evaluate_ground_truth(
    ground_truth_path: Path,
    *,
    max_candidates: int,
    max_corruptions: int,
) -> dict[str, Any]:
    truth = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []

    for name, expected in sorted(truth.items()):
        true_model = normalize_alphanum(expected.get("model_number"))
        model_cases = corruptions(true_model)
        selected = _limit(model_cases, max_corruptions)
        for corruption in selected:
            cases.append(
                _rank_case(
                    corruption["corrupted"],
                    true_model,
                    manufacturer=expected.get("brand"),
                    category=expected.get("category"),
                    validate=False,
                    max_candidates=max_candidates,
                )
            )
        models.append(
            {
                "image": name,
                "model_number": true_model,
                "corruptions_available": len(model_cases),
                "corruptions_scored": len(selected),
            }
        )

    return {
        "source": str(ground_truth_path),
        "models": models,
        "cases": cases,
        "summary": _score_cases(cases, min(10, max_candidates)),
        "latency_s": round(time.perf_counter() - started, 3),
    }


def sample_database_models(
    db_path: Path | str,
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically sample model-like rows from the NRCan database."""
    conn = get_db_connection(db_path)
    try:
        rows: list[dict[str, Any]] = []
        for table in ALL_APPLIANCE_TABLES:
            for row in conn.execute(f'SELECT MODEL_NUM_1, BRAND_NAME FROM "{table}"'):
                normalized = normalize_alphanum(row["MODEL_NUM_1"])
                if not 5 <= len(normalized) <= 25:
                    continue
                if not any(character.isalpha() for character in normalized):
                    continue
                if not any(character.isdigit() for character in normalized):
                    continue
                rows.append(
                    {
                        "model_number": normalized,
                        "manufacturer": row["BRAND_NAME"],
                        "category": table,
                    }
                )
    finally:
        conn.close()

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["model_number"], row["category"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    unique.sort(key=lambda row: (row["category"], row["model_number"]))
    random.Random(seed).shuffle(unique)
    return unique[:count]


def evaluate_database(
    db_path: Path | str,
    *,
    samples: int,
    seed: int,
    max_candidates: int,
    max_corruptions: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    selected_models = sample_database_models(db_path, count=samples, seed=seed)
    validator = make_validator(build_validation_index(db_path))
    cases: list[dict[str, Any]] = []

    for row in selected_models:
        model_cases = _limit(corruptions(row["model_number"]), max_corruptions)
        for corruption in model_cases:
            cases.append(
                _rank_case(
                    corruption["corrupted"],
                    row["model_number"],
                    manufacturer=row["manufacturer"],
                    category=row["category"],
                    validate=True,
                    max_candidates=max_candidates,
                    validator=validator,
                )
            )

    summary = _score_cases(cases, min(10, max_candidates))
    scored = [case for case in cases if case["true_confirmed"] is not None]
    confirmed = sum(1 for case in scored if case["true_confirmed"])
    top = sum(1 for case in scored if case["true_is_top_db"])
    unique = sum(1 for case in scored if case["db_hit_count"] == 1)
    summary["validation"] = {
        "true_confirmed": {
            "matched": confirmed,
            "total": len(scored),
            "accuracy": round(confirmed / len(scored), 4) if scored else 0.0,
        },
        "true_is_top_db_confirmed": {
            "matched": top,
            "total": len(scored),
            "accuracy": round(top / len(scored), 4) if scored else 0.0,
        },
        "unique_db_hit": {
            "matched": unique,
            "total": len(scored),
            "accuracy": round(unique / len(scored), 4) if scored else 0.0,
        },
        "avg_db_hits": round(
            sum(case["db_hit_count"] for case in scored) / len(scored), 2
        )
        if scored
        else 0.0,
    }

    return {
        "source": str(db_path),
        "samples": selected_models,
        "cases": cases,
        "summary": summary,
        "latency_s": round(time.perf_counter() - started, 3),
    }


def _print_section(title: str, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    k = summary["k"]
    print(title)
    print(f"  models:          {len(payload.get('models') or payload.get('samples') or [])}")
    print(f"  cases:           {summary['cases']}")
    print("  ranks below are among corrections; the original always ranks first")
    print(
        f"  recall@1:        {summary['recall_at_1']['matched']}/{summary['recall_at_1']['total']} "
        f"({summary['recall_at_1']['accuracy'] * 100:.1f}%)"
    )
    print(
        f"  recall@5:        {summary['recall_at_5']['matched']}/{summary['recall_at_5']['total']} "
        f"({summary['recall_at_5']['accuracy'] * 100:.1f}%)"
    )
    if k not in (1, 5):
        section = summary[f"recall_at_{k}"]
        print(
            f"  recall@{k:<2}       {section['matched']}/{section['total']} "
            f"({section['accuracy'] * 100:.1f}%)"
        )
    print(f"  MRR:             {summary['mrr']:.4f}")
    print(f"  avg candidates:  {summary['avg_candidates']:.2f}")
    if summary["original_first_violations"]:
        print(f"  ORIGINAL-NOT-FIRST: {summary['original_first_violations']}")

    validation = summary.get("validation")
    if validation:
        print()
        print("  Exact database validation:")
        for label, key in (
            ("true model confirmed", "true_confirmed"),
            ("true model is top DB hit", "true_is_top_db_confirmed"),
            ("only one DB hit", "unique_db_hit"),
        ):
            section = validation[key]
            print(
                f"    {label:24} {section['matched']}/{section['total']} "
                f"({section['accuracy'] * 100:.1f}%)"
            )
        print(f"    {'avg DB hits per case':24} {validation['avg_db_hits']:.2f}")
    print(f"  latency:         {payload['latency_s']:.2f}s")
    print()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Evaluate OCR-error candidate generation against synthetic corruptions"
    )
    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH),
        help="Ground-truth JSON keyed by image filename",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the SQLite database")
    parser.add_argument(
        "--db-samples",
        type=int,
        default=DEFAULT_DB_SAMPLES,
        help=f"Database models to sample for validation scoring (default: {DEFAULT_DB_SAMPLES})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help=f"Candidate cap per case (default: {DEFAULT_MAX_CANDIDATES})",
    )
    parser.add_argument(
        "--max-corruptions",
        type=int,
        default=DEFAULT_MAX_CORRUPTIONS,
        help=f"Corruptions scored per model (default: {DEFAULT_MAX_CORRUPTIONS})",
    )
    parser.add_argument("--no-db", action="store_true", help="Skip the database validation section")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report")
    parser.add_argument("--output", help="Write JSON results to this file")
    args = parser.parse_args(argv)

    try:
        payload: dict[str, Any] = {
            "ground_truth": evaluate_ground_truth(
                Path(args.ground_truth),
                max_candidates=args.max_candidates,
                max_corruptions=args.max_corruptions,
            )
        }
        if not args.no_db:
            payload["database"] = evaluate_database(
                args.db,
                samples=args.db_samples,
                seed=args.seed,
                max_candidates=args.max_candidates,
                max_corruptions=args.max_corruptions,
            )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
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
        print("OCR-error candidate generation — evaluation")
        print("=" * 78)
        print()
        _print_section(
            "Ground-truth plates (synthetic single-character corruption, no DB validation)",
            payload["ground_truth"],
        )
        if "database" in payload:
            _print_section(
                "Database models (synthetic corruption, exact validation)",
                payload["database"],
            )

    total_cases = payload["ground_truth"]["summary"]["cases"] + sum(
        section["summary"]["cases"] for key, section in payload.items() if key != "ground_truth"
    )
    return 0 if total_cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
