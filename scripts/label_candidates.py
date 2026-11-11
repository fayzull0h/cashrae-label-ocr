"""OCR-error candidate generation for model numbers (Phase 2, Stage 3).

Rating-plate OCR reliably confuses a small set of characters: ``0``/``O``,
``1``/``I``/``L``, ``5``/``S``, ``8``/``B`` and a handful more. This module takes
a parsed model number, generates the plausible single-character corrections,
and ranks them so a later stage can try the most likely model first.

Design rules:

- **The parsed value is never rewritten.** It is always returned first with
  ``is_original: True``. Candidate generation is a suggestion layer, not a
  silent correction.
- **Explainable.** Every candidate carries its edit trail (position, ``from``,
  ``to``) and a ``plausibility`` score based on whether the replacement keeps
  the character class (digit vs letter) of its neighbours.
- **Bounded.** One edit by default, substitutions only, hard candidate cap.
- **Database validation is a ranking signal, not a rewrite.** When validation is
  enabled, a candidate that exists in the NRCan database ranks above one that
  does not. Validation uses the exact-only helper, so a fuzzy hit can never
  promote a candidate.

Retrieval (turning a corrected model into an AEC value) belongs to Stage 4.

Typical usage::

    from scripts.label_parse import parse_fields
    from scripts.label_candidates import candidates_from_parse

    parsed = parse_fields(raw_text)
    result = candidates_from_parse(parsed, category="clothes_dryers")
    result["candidates"][1]["model_number"]

CLI::

    python scripts/label_candidates.py --model "PCKS433ET" --brand GE
    python scripts/label_candidates.py --model "PCKS433ET" --category clothes_dryers --json
    python scripts/label_candidates.py --model "PCKS433ET" --no-validate
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow direct execution as ``python scripts/label_candidates.py``.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.appliance_lookup import (  # noqa: E402
    DEFAULT_DB_PATH,
    find_exact_matches_batch,
    normalize_alphanum,
)

#: One substitution by default: it covers the dominant single-character OCR error.
DEFAULT_MAX_EDITS = 1

#: Hard cap on the returned list (the original always counts as one entry).
DEFAULT_MAX_CANDIDATES = 64

#: Signature of an alternative validator for :func:`rank_candidates`.
Validator = Callable[
    [list[str], str | None, str | None],
    dict[str, list[dict[str, Any]]],
]

#: Characters OCR engines confuse on dense, embossed rating plates.
#: Deliberately narrow: every extra group multiplies the candidate list.
CONFUSION_GROUPS: tuple[tuple[str, ...], ...] = (
    ("0", "O"),
    ("1", "I", "L"),
    ("2", "Z"),
    ("4", "A"),
    ("5", "S"),
    ("6", "G"),
    ("7", "T"),
    ("8", "B"),
    ("C", "G"),
    ("M", "N"),
    ("U", "V"),
)


def _build_confusion_map(groups: Iterable[Iterable[str]]) -> dict[str, tuple[str, ...]]:
    """Build a symmetric character -> alternatives map from confusion groups."""
    peers: dict[str, set[str]] = {}
    for group in groups:
        members = {character.upper() for character in group}
        for character in members:
            peers.setdefault(character, set()).update(members)
    return {
        character: tuple(sorted(options - {character}))
        for character, options in peers.items()
    }


#: Read-only character -> alternative characters lookup.
CONFUSION_MAP: dict[str, tuple[str, ...]] = _build_confusion_map(CONFUSION_GROUPS)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def _character_class(character: str) -> str:
    return "digit" if character.isdigit() else "letter"


def _class_fit(text: str, position: int, replacement: str) -> float:
    """How well ``replacement`` fits the character class of its neighbours.

    1.0 means the replacement matches the class of both neighbours (or the only
    neighbour), 0.0 means it matches neither. This is a cheap, explainable way to
    prefer ``PCKS4330T`` over ``PCKS433OT`` inside a digit run.
    """
    neighbours = [
        text[index]
        for index in (position - 1, position + 1)
        if 0 <= index < len(text)
    ]
    if not neighbours:
        return 0.5
    replacement_class = _character_class(replacement)
    matching = sum(_character_class(neighbour) == replacement_class for neighbour in neighbours)
    return round(matching / len(neighbours), 3)


def _make_candidate(text: str, edits: list[dict[str, Any]], *, is_original: bool = False) -> dict[str, Any]:
    plausibility = (
        1.0
        if not edits
        else round(sum(edit["class_fit"] for edit in edits) / len(edits), 3)
    )
    return {
        "model_number": text,
        "normalized": text,
        "edit_count": len(edits),
        "edits": edits,
        "is_original": is_original,
        "plausibility": plausibility,
        "db_match": None,
        "db_match_count": 0,
        "db_categories": [],
        "rank": None,
    }


def _single_substitutions(text: str):
    """Yield ``(position, from, to, candidate)`` for every confusion swap."""
    for position, character in enumerate(text):
        for replacement in CONFUSION_MAP.get(character, ()):
            yield (
                position,
                character,
                replacement,
                text[:position] + replacement + text[position + 1 :],
            )


def _order_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -candidate["plausibility"],
        candidate["edit_count"],
        tuple(edit["position"] for edit in candidate["edits"]),
        candidate["normalized"],
    )


def generate_candidates(
    model_number: str | None,
    *,
    max_edits: int = DEFAULT_MAX_EDITS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[dict[str, Any]]:
    """Generate ranked-by-plausibility candidate corrections for a model number.

    The original normalized value is always the first entry. ``max_edits`` is
    capped at 2 because two-edit neighbourhoods grow quickly and get noisy;
    ``max_candidates`` includes the original.
    """
    if not 0 <= max_edits <= 2:
        raise ValueError("max_edits must be 0, 1 or 2")
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")

    original = normalize_alphanum(model_number)
    if not original:
        return []

    candidates = [_make_candidate(original, [], is_original=True)]
    seen = {original}

    def add(text: str, edits: list[dict[str, Any]]) -> None:
        if text in seen:
            return
        seen.add(text)
        candidates.append(_make_candidate(text, edits))

    for position, character, replacement, text in _single_substitutions(original):
        add(
            text,
            [
                {
                    "position": position,
                    "from": character,
                    "to": replacement,
                    "class_fit": _class_fit(original, position, replacement),
                }
            ],
        )

    if max_edits >= 2:
        first_round = [candidate for candidate in candidates if candidate["edit_count"] == 1]
        for base in first_round:
            for position, character, replacement, text in _single_substitutions(base["normalized"]):
                # Do not undo an edit already applied to this branch.
                if any(edit["position"] == position for edit in base["edits"]):
                    continue
                add(
                    text,
                    [
                        *base["edits"],
                        {
                            "position": position,
                            "from": character,
                            "to": replacement,
                            "class_fit": _class_fit(base["normalized"], position, replacement),
                        },
                    ],
                )

    ordered = sorted(candidates[1:], key=_order_key)
    return candidates[:1] + ordered[: max_candidates - 1]


# ---------------------------------------------------------------------------
# Ranking and database validation
# ---------------------------------------------------------------------------


def _rank_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if candidate["is_original"] else 1,
        0 if candidate["db_match"] else 1,
        candidate["edit_count"],
        -candidate["plausibility"],
        candidate["normalized"],
    )


def rank_candidates(
    candidates: list[dict[str, Any]],
    *,
    manufacturer: str | None = None,
    category: str | None = None,
    validate: bool = True,
    db_path: Path | str = DEFAULT_DB_PATH,
    validator: Validator | None = None,
) -> list[dict[str, Any]]:
    """Rank candidates: original first, then database-confirmed, then plausibility.

    With ``validate=False`` every ``db_match`` is ``None`` (meaning "not
    checked"), which callers can distinguish from a checked ``False``.

    ``validator`` optionally replaces the database lookup, for callers that
    already hold an in-memory index (batch evaluation) or need to stub it out.
    It receives ``(normalized_models, manufacturer, category)`` and returns a
    mapping of normalized model -> matching records with a ``category`` key.
    """
    ranked = [dict(candidate) for candidate in candidates]
    if not ranked:
        return []

    if validate:
        models = [candidate["normalized"] for candidate in ranked]
        if validator is not None:
            matches = validator(models, manufacturer, category)
        else:
            matches = find_exact_matches_batch(
                models,
                manufacturer=manufacturer,
                category=category,
                db_path=db_path,
            )
        for candidate in ranked:
            hits = matches.get(candidate["normalized"], [])
            candidate["db_match"] = bool(hits)
            candidate["db_match_count"] = len(hits)
            candidate["db_categories"] = sorted({hit["category"] for hit in hits})
    else:
        for candidate in ranked:
            candidate["db_match"] = None
            candidate["db_match_count"] = None
            candidate["db_categories"] = []

    ranked.sort(key=_rank_key)
    for position, candidate in enumerate(ranked, start=1):
        candidate["rank"] = position
    return ranked


def candidates_from_parse(
    parsed_fields: dict[str, Any],
    *,
    manufacturer: str | None = None,
    category: str | None = None,
    validate: bool = True,
    max_edits: int = DEFAULT_MAX_EDITS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Build a candidate payload from :func:`scripts.label_parse.parse_fields` output."""
    model_field = parsed_fields.get("model_number") or {}
    brand_field = parsed_fields.get("brand") or {}

    original_value = model_field.get("value")
    original_normalized = model_field.get("normalized") or normalize_alphanum(original_value)
    manufacturer = manufacturer or brand_field.get("value")

    generated = generate_candidates(
        original_normalized,
        max_edits=max_edits,
        max_candidates=max_candidates,
    )
    ranked = rank_candidates(
        generated,
        manufacturer=manufacturer,
        category=category,
        validate=validate,
        db_path=db_path,
    )

    return {
        "original": {
            "value": original_value,
            "normalized": original_normalized,
            "confidence": model_field.get("confidence"),
            "method": model_field.get("method"),
        },
        "manufacturer": manufacturer,
        "category": category,
        "validated": validate,
        "candidates": ranked,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_edits(candidate: dict[str, Any]) -> str:
    if candidate["is_original"]:
        return "(original)"
    return ", ".join(
        f"{edit['position']}:{edit['from']}->{edit['to']}" for edit in candidate["edits"]
    )


def _print_report(payload: dict[str, Any]) -> None:
    original = payload["original"]
    print("=" * 78)
    print("OCR-error candidates for model number")
    print("=" * 78)
    print(f"Model:     {original['value']} (normalized: {original['normalized'] or '(empty)'})")
    print(f"Brand:     {payload['manufacturer'] or '(not given)'}")
    print(f"Category:  {payload['category'] or '(all)'}")
    print(f"Validated: {'yes' if payload['validated'] else 'no'}")
    print()

    if not payload["candidates"]:
        print("No model number supplied; no candidates generated.")
        print()
        return

    header = f"{'Rank':>4}  {'Candidate':22} {'Edits':>5}  {'DB':3}  {'Fit':>4}  Correction"
    print(header)
    print("-" * len(header))
    for candidate in payload["candidates"]:
        if candidate["db_match"] is None:
            database = "n/a"
        else:
            database = "hit" if candidate["db_match"] else "--"
        print(
            f"{candidate['rank']:>4}  {candidate['model_number']:22} "
            f"{candidate['edit_count']:>5}  {database:3}  {candidate['plausibility']:>4.2f}  "
            f"{_format_edits(candidate)}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Generate and rank OCR-error candidates for an appliance model number"
    )
    parser.add_argument("--model", required=True, help="Parsed model number")
    parser.add_argument("--brand", help="Manufacturer/brand used to narrow database validation")
    parser.add_argument("--category", help="Appliance category used to narrow database validation")
    parser.add_argument(
        "--max-edits",
        type=int,
        default=DEFAULT_MAX_EDITS,
        choices=(0, 1, 2),
        help="Maximum substitutions per candidate (default: 1)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help=f"Maximum number of candidates, including the original (default: {DEFAULT_MAX_CANDIDATES})",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip NRCan database validation (db_match becomes null)",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the SQLite database")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report")
    parser.add_argument("--output", help="Write JSON results to this file")
    args = parser.parse_args(argv)

    normalized = normalize_alphanum(args.model)
    if not normalized:
        print(f"Model number '{args.model}' contains no alphanumeric characters.", file=sys.stderr)
        return 1

    try:
        generated = generate_candidates(
            args.model,
            max_edits=args.max_edits,
            max_candidates=args.max_candidates,
        )
        candidates = rank_candidates(
            generated,
            manufacturer=args.brand,
            category=args.category,
            validate=args.validate,
            db_path=args.db,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = {
        "original": {
            "value": args.model,
            "normalized": normalized,
            "confidence": None,
            "method": None,
        },
        "manufacturer": args.brand,
        "category": args.category,
        "validated": args.validate,
        "candidates": candidates,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {output_path}")

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
