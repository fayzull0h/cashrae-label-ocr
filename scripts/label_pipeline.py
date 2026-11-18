"""Label-to-energy pipeline (Phase 2, Stage 4).

Connects the pieces built in Stages 1–3 into one call:

1. OCR a rating plate (or accept raw text / a manual model number).
2. Parse brand, model number, and serial number.
3. Try the parsed model against the NRCan database.
4. If it is not an exact hit, try the ranked OCR-error candidates.
5. Fall back to fuzzy matching on the original reading when allowed.

The parsed value is never rewritten. Every model that was considered is recorded
in ``attempts``, and ``retrieval.model_used`` says which one actually produced
the energy value. Confidence is a coarse band (``high`` / ``medium`` / ``low``)
with its inputs reported alongside it, not an invented blended score.

Typical usage::

    from scripts.label_pipeline import analyze_label

    result = analyze_label("data/samples/dryer_label.png", category="clothes_dryers")
    result["retrieval"]["annual_energy_consumption"]

CLI::

    python scripts/label_pipeline.py --image data/samples/dryer_label.png
    python scripts/label_pipeline.py --image ... --category clothes_dryers --json
    python scripts/label_pipeline.py --text "LG\\nMODEL: DLE8400WE" --brand LG
    python scripts/label_pipeline.py --model "DLE8400WE" --brand LG
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow direct execution as ``python scripts/label_pipeline.py``.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.appliance_lookup import (  # noqa: E402
    DEFAULT_DB_PATH,
    HIGH_CONFIDENCE_THRESHOLD,
    lookup_appliance,
    normalize_alphanum,
    normalize_category,
)
from scripts.label_candidates import (  # noqa: E402
    DEFAULT_MAX_CANDIDATES,
    candidates_from_parse,
)

#: How many ranked candidates to examine for an exact hit before fuzzy matching.
DEFAULT_MAX_ATTEMPTS = 8


def _ocr_summary(ocr: dict[str, Any] | None) -> dict[str, Any] | None:
    if not ocr:
        return None
    return {
        "backend": ocr.get("backend"),
        "confidence": ocr.get("confidence"),
        "preprocessing": ocr.get("preprocessing"),
    }


def _summarize_lookup(
    result: dict[str, Any],
    model_used: str,
    *,
    from_original: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": result.get("status"),
        "model_used": model_used,
        "matched_model": result.get("model_number"),
        "manufacturer": result.get("manufacturer"),
        "category": result.get("category"),
        "annual_energy_consumption": result.get("annual_energy_consumption"),
        "unit": result.get("unit"),
        "energy_consumption_available": bool(result.get("energy_consumption_available")),
        "match_type": result.get("match_type"),
        "match_confidence": result.get("match_confidence"),
        "from_original": from_original,
        "source": result.get("source"),
        "sources": result.get("sources"),
        "energy_conflict": result.get("energy_conflict", False),
        "energy_conflict_note": result.get("energy_conflict_note"),
    }
    for key in ("message", "notes", "warning"):
        if result.get(key):
            summary[key] = result[key]
    if result.get("candidates"):
        summary["candidates"] = [
            {
                "model_number": candidate.get("model_number"),
                "manufacturer": candidate.get("manufacturer"),
                "category": candidate.get("category"),
                "match_confidence": candidate.get("match_confidence"),
            }
            for candidate in result["candidates"]
        ]
    return summary


def _confidence_band(
    status: str | None,
    *,
    from_original: bool,
    match_confidence: float | None,
) -> str | None:
    """Coarse, explainable confidence band for the retrieval result."""
    if status == "exact_match":
        return "high" if from_original else "medium"
    if status == "fuzzy_match":
        if from_original and (match_confidence or 0.0) >= HIGH_CONFIDENCE_THRESHOLD:
            return "medium"
        return "low"
    return None


def _finish(
    payload: dict[str, Any],
    lookup: dict[str, Any],
    attempts: list[dict[str, Any]],
    model_used: str,
    *,
    from_original: bool,
    rank: int | None,
) -> dict[str, Any]:
    payload["status"] = lookup.get("status")
    payload["attempts"] = attempts
    payload["retrieval"] = _summarize_lookup(lookup, model_used, from_original=from_original)
    payload["confidence"] = _confidence_band(
        lookup.get("status"),
        from_original=from_original,
        match_confidence=lookup.get("match_confidence"),
    )
    payload["confidence_components"] = {
        "parse_confidence": payload["confidence_components"]["parse_confidence"],
        "candidate_rank": rank,
        "from_original": from_original,
        "match_type": lookup.get("match_type"),
        "match_confidence": lookup.get("match_confidence"),
    }
    return payload


def analyze_label(
    image_path: str | Path | None = None,
    *,
    raw_text: str | None = None,
    model_number: str | None = None,
    manufacturer: str | None = None,
    category: str | None = None,
    engine: str = "auto",
    backend: Any = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    allow_fuzzy: bool = True,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Run the full label → energy flow and return a structured result.

    Provide one of ``image_path`` (OCR + parse), ``raw_text`` (parse only), or
    ``model_number`` (manual override, no OCR or parsing). When ``model_number``
    is given it takes precedence and ``image_path``/``raw_text`` are ignored.
    """
    if image_path is None and raw_text is None and not model_number:
        raise ValueError("Provide an image path, raw OCR text, or a model number.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    # Validate the category up front so a typo fails fast instead of surfacing
    # as an invalid_input retrieval status after OCR has already run.
    category = normalize_category(category)

    ocr: dict[str, Any] | None = None
    parsed: dict[str, Any] | None = None

    if model_number:
        parsed_model = normalize_alphanum(model_number)
        parsed_brand = manufacturer
        parse_confidence = None
    else:
        if raw_text is not None:
            from scripts.label_parse import parse_fields

            parsed = parse_fields(raw_text)
        else:
            from scripts.label_ocr import extract_text

            ocr_backend = backend
            if ocr_backend is None and engine != "auto":
                from scripts.ocr_backends import get_backend

                ocr_backend = get_backend(engine)
            ocr = extract_text(image_path, backend=ocr_backend)

            from scripts.label_parse import parse_ocr_result

            parsed = parse_ocr_result(ocr)

        parsed_model = parsed["model_number"]["normalized"]
        parsed_brand = manufacturer or parsed["brand"]["value"]
        parse_confidence = parsed["model_number"]["confidence"]

    payload: dict[str, Any] = {
        "status": "no_model_found",
        "image_path": str(image_path) if image_path else None,
        "ocr": _ocr_summary(ocr),
        "parsed": parsed,
        "category": category,
        "retrieval": None,
        "attempts": [],
        "confidence": None,
        "confidence_components": {
            "parse_confidence": parse_confidence,
            "candidate_rank": None,
            "from_original": None,
            "match_type": None,
            "match_confidence": None,
        },
    }

    if not parsed_model:
        return payload

    # Manual override: retrieve directly, no correction search.
    if model_number:
        lookup = lookup_appliance(
            manufacturer=parsed_brand,
            model_number=parsed_model,
            category=category,
            db_path=db_path,
            allow_fuzzy=allow_fuzzy,
        )
        return _finish(
            payload,
            lookup,
            [
                {
                    "rank": 1,
                    "model": parsed_model,
                    "is_original": True,
                    "db_match": None,
                    "result": lookup.get("status"),
                }
            ],
            parsed_model,
            from_original=True,
            rank=1,
        )

    candidate_payload = candidates_from_parse(
        parsed,
        manufacturer=manufacturer,
        category=category,
        validate=True,
        max_candidates=max_candidates,
        db_path=db_path,
    )
    candidates = candidate_payload["candidates"]

    attempts: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    for candidate in candidates[:max_attempts]:
        source_probe = lookup_appliance(
            manufacturer=parsed_brand, model_number=candidate["model_number"],
            category=category, db_path=db_path, allow_fuzzy=False,
        )
        entry = {
            "rank": candidate["rank"],
            "model": candidate["model_number"],
            "is_original": candidate["is_original"],
            "db_match": candidate["db_match"],
            "source_matches": {
                name: {"status": detail.get("status"),
                       "match_type": detail.get("match_type"),
                       "candidates": detail.get("candidates", [])}
                for name, detail in source_probe.get("sources", {}).items()
            },
        }
        if candidate["db_match"]:
            entry["result"] = "exact_match"
            attempts.append(entry)
            if accepted is None:
                accepted = candidate
            break
        entry["result"] = "no exact match"
        attempts.append(entry)

    if accepted is not None:
        lookup = lookup_appliance(
            manufacturer=parsed_brand,
            model_number=accepted["model_number"],
            category=category,
            db_path=db_path,
        )
        return _finish(
            payload,
            lookup,
            attempts,
            accepted["model_number"],
            from_original=accepted["is_original"],
            rank=accepted["rank"],
        )

    # No candidate is an exact database model. Optionally fall back to fuzzy
    # matching on the literal reading, which is the only value we actually saw.
    lookup = lookup_appliance(
        manufacturer=parsed_brand,
        model_number=parsed_model,
        category=category,
        db_path=db_path,
        allow_fuzzy=allow_fuzzy,
    )
    if not any(attempt["model"] == parsed_model for attempt in attempts):
        attempts.append(
            {
                "rank": None,
                "model": parsed_model,
                "is_original": True,
                "db_match": False,
                "result": lookup.get("status"),
                "match_confidence": lookup.get("match_confidence"),
            }
        )
    return _finish(
        payload,
        lookup,
        attempts,
        parsed_model,
        from_original=True,
        rank=None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_report(payload: dict[str, Any]) -> None:
    print("=" * 78)
    print("Label → energy pipeline")
    print("=" * 78)

    if payload.get("image_path"):
        print(f"Image:      {payload['image_path']}")
    ocr = payload.get("ocr")
    if ocr:
        confidence = ocr.get("confidence")
        confidence_text = "n/a" if confidence is None else f"{confidence:.2f}"
        print(f"OCR:        {ocr.get('backend')} (confidence {confidence_text})")

    parsed = payload.get("parsed")
    if parsed:
        print(
            "Parsed:     "
            f"brand={parsed['brand']['value'] or '(not found)'}  "
            f"model={parsed['model_number']['value'] or '(not found)'}  "
            f"serial={parsed['serial_number']['value'] or '(not found)'}"
        )
    print(f"Category:   {payload['category'] or '(all)'}")
    print()

    retrieval = payload.get("retrieval")
    if not retrieval:
        print("Retrieval:  no model number to look up")
        print("Confidence: -")
    else:
        source = retrieval.get("source") or {}
        origin = "original" if retrieval.get("from_original") else "correction"
        print(f"Retrieval:  {retrieval['status']} (model {retrieval['model_used']}, {origin})")
        if retrieval.get("matched_model") and retrieval["matched_model"] != retrieval["model_used"]:
            print(f"Matched:    {retrieval['matched_model']}")
        if retrieval.get("energy_consumption_available"):
            print(f"Energy:     {retrieval['annual_energy_consumption']} {retrieval.get('unit') or ''}".rstrip())
        else:
            print("Energy:     unavailable")
        if source:
            print(f"Source:     {source.get('database')} / {source.get('table')}")
        energy_star = (retrieval.get("sources") or {}).get("energy_star")
        if energy_star and energy_star.get("status") not in ("not_found", "not_available"):
            print(f"ENERGY STAR: {energy_star['status']} "
                  f"(dataset {energy_star.get('dataset_id')})")
            if energy_star.get("annual_energy_consumption") is not None:
                print(f"             annual energy: {energy_star['annual_energy_consumption']} "
                      f"{energy_star.get('unit')}")
            if energy_star.get("annual_energy_context"):
                print(f"             annual energy context: "
                      f"{energy_star['annual_energy_context']}")
            for candidate in energy_star.get("candidates", []):
                print(f"             {candidate.get('manufacturer')} {candidate.get('model_number')} "
                      f"({candidate.get('match_type')})")
                source = candidate.get("source") or {}
                if source:
                    print(f"             Source: {source.get('url')} "
                          f"(record {source.get('record_id')}, snapshot {source.get('snapshot_date')})")
                metrics = candidate.get("energy_metrics")
                if metrics is None:
                    metrics = [candidate.get("energy_metric") or {}]
                for metric in metrics:
                    if metric.get("value") is not None:
                        print(f"             {metric.get('name')}: {metric['value']} "
                              f"{metric.get('unit')}")
        if retrieval.get("energy_conflict"):
            print(f"Energy note: {retrieval.get('energy_conflict_note')}")
        print(f"Confidence: {payload['confidence'] or '-'}")
        if retrieval.get("notes"):
            print(f"Note:       {retrieval['notes']}")
        if retrieval.get("message"):
            print(f"Message:    {retrieval['message']}")

    if payload.get("attempts"):
        print()
        print("Attempts:")
        header = f"  {'Rank':>4}  {'Model':22} {'Result':18} DB"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for attempt in payload["attempts"]:
            rank = "-" if attempt["rank"] is None else attempt["rank"]
            database = "n/a" if attempt["db_match"] is None else ("hit" if attempt["db_match"] else "--")
            print(
                f"  {rank:>4}  {attempt['model']:22} {attempt['result']:18} {database}"
            )
            for source_name, source_match in attempt.get("source_matches", {}).items():
                if source_match.get("status") not in ("not_found", "not_available"):
                    print(f"             {source_name}: {source_match['status']} "
                          f"({source_match.get('match_type') or 'multiple matches'})")
    print()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Run OCR + parsing + NRCan retrieval on a label photo"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Label photo to OCR and parse")
    source.add_argument("--text", help="Raw OCR text to parse (skips OCR)")
    source.add_argument("--model", help="Manual model number override (skips OCR and parsing)")
    parser.add_argument("--brand", help="Manufacturer override for retrieval")
    parser.add_argument("--category", help="Appliance category to restrict retrieval")
    parser.add_argument(
        "--engine",
        default="auto",
        choices=("auto", "paddleocr", "tesseract"),
        help="OCR engine for --image: auto, paddleocr, or tesseract",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help=f"Candidate cap (default: {DEFAULT_MAX_CANDIDATES})",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Candidates examined before fuzzy fallback (default: {DEFAULT_MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--no-fuzzy",
        dest="allow_fuzzy",
        action="store_false",
        help="Disable fuzzy fallback when no candidate is an exact match",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the SQLite database")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report")
    parser.add_argument("--output", help="Write JSON results to this file")
    args = parser.parse_args(argv)

    try:
        payload = analyze_label(
            args.image,
            raw_text=args.text,
            model_number=args.model,
            manufacturer=args.brand,
            category=args.category,
            engine=args.engine,
            max_candidates=args.max_candidates,
            max_attempts=args.max_attempts,
            allow_fuzzy=args.allow_fuzzy,
            db_path=args.db,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - OCR init failures get a clear message
        print(f"Pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {output_path}")

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_report(payload)

    return 0 if payload.get("retrieval") else 1


if __name__ == "__main__":
    raise SystemExit(main())
