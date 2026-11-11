"""Field parsing for appliance rating plates (Phase 2, Stage 2).

Turns the raw OCR text produced by :mod:`scripts.label_ocr` into structured
``brand``, ``model_number`` and ``serial_number`` fields. Values are normalized
with :func:`scripts.appliance_lookup.normalize_alphanum` so they can be passed
straight to the Phase 1 retrieval engine.

The parser is deliberately rule-based and explainable:

1. **Label anchors** — ``MODEL`` / ``MODÈLE``, ``SERIAL`` / ``SÉRIE``,
   ``BRAND`` / ``MANUFACTURER`` and their common abbreviations.
2. **Adjacent lines** — OCR frequently puts the value on the line before or
   after the anchor, and sometimes before the anchor on the same line.
3. **Heuristics** — model-like and serial-like token scoring plus a known-brand
   list, used when anchors are missing or garbled.

Confidence is assigned per field from the *method* that produced it. The
aggregate OCR confidence is not used as a gate: measured cases show correct
text at lower confidence than incorrect text.

Typical usage::

    from scripts.label_ocr import extract_text
    from scripts.label_parse import parse_fields, parse_ocr_result

    result = extract_text("data/samples/airconditioner_label.png")
    fields = parse_ocr_result(result)
    print(fields["model_number"]["value"])

CLI::

    python scripts/label_parse.py --image data/samples/airconditioner_label.png
    python scripts/label_parse.py --text "MODEL: ABC-123"
    python scripts/label_parse.py --image data/samples/airconditioner_label.png --json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence
import argparse
import json
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow direct execution as ``python scripts/label_parse.py``.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.appliance_lookup import normalize_alphanum  # noqa: E402

# ---------------------------------------------------------------------------
# Anchor patterns
# ---------------------------------------------------------------------------

MODEL_ANCHOR_RE = re.compile(
    r"(?:MODEL|MODELE|MODÈLE|MOD)(?![A-Za-z])"
    r"\s*(?:/\s*(?:MODEL|MODELE|MODÈLE))?"
    r"\s*(?:NO\.?|NUMBER|NUM\.?|#)?",
    re.IGNORECASE,
)
GLUED_MODEL_RE = re.compile(r"^(?P<prefix>[A-Za-z][A-Za-z&'.\- ]{2,})MOD$", re.IGNORECASE)
SERIAL_ANCHOR_RE = re.compile(
    r"(?:SERIAL|SERIE|SÉRIE)(?![A-Za-z])"
    r"\s*(?:/\s*(?:SERIAL|SERIE|SÉRIE))?"
    r"\s*(?:NO\.?|NUMBER|NUM\.?|#)?"
    r"|S\s*/\s*N(?:O)?\.?"
    r"|\bSER\.?(?![A-Za-z])"
    r"|\bSN\b",
    re.IGNORECASE,
)
BRAND_ANCHOR_RE = re.compile(
    r"(?:BRAND|MANUFACTURER|MARQUE|MFG\.?)\s*(?:NAME)?\s*[:#]?",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/\-]*")

_DATE_RE = re.compile(r"(?:\d{2}/\d{2}/\d{2}|\d{4}-\d{2}-\d{2})")
_URL_RE = re.compile(r"(?:WWW|HTTPS?|\.COM\b|\.CA\b|\.ORG\b)", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\b1[-.]?8\d{2}[-.]?\d{3}[-.]?\d{4}\b|\d{3}[-.]\d{3}[-.]\d{4})")

#: Tokens that are never model or serial numbers.
NOISE_SUBSTRINGS = (
    "VOLT",
    "HERTZ",
    "AMP",
    "PSI",
    "KPA",
    "MPA",
    "BTU",
    "WATT",
    "PHASE",
    "FUSE",
    "CIRCUIT",
    "PRESSURE",
    "REFRIGERANT",
    "R410",
    "R134",
    "R600",
    "SERIAL",
    "MODEL",
    "MODELE",
    "DATE",
    "YEAR",
    "MANUFACTUR",
    "WARRANTY",
    "PATENT",
    "WWW",
    "HTTP",
    "ENERGY",
    "LISTED",
    "STANDARD",
    "CAPACITY",
    "DIMENSION",
    "WEIGHT",
    "PSIG",
    "RLA",
    "LRA",
    "FLA",
    "AHRI",
    "SEER",
)

#: (normalized pattern, canonical display name). Longer patterns win.
KNOWN_BRANDS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            ("INTERNATIONALCOMFORTPRODUCTS", "International Comfort Products"),
            ("GENERALELECTRICCOMPANY", "General Electric Company"),
            ("GENERALELECTRIC", "General Electric Company"),
            ("DANBYPRODUCTSLTD", "Danby Products Ltd"),
            ("DANBYPRODUCTS", "Danby Products Ltd"),
            ("DANBYDESIGNER", "Danby Designer"),
            ("DANBY", "Danby"),
            ("KITCHENAID", "KitchenAid"),
            ("FRIGIDAIRE", "Frigidaire"),
            ("WHIRLPOOL", "Whirlpool"),
            ("MAYTAG", "Maytag"),
            ("JENNAIR", "JennAir"),
            ("ELECTROLUX", "Electrolux"),
            ("THERMADOR", "Thermador"),
            ("BERTAZZONI", "Bertazzoni"),
            ("FISHERPAYKEL", "Fisher & Paykel"),
            ("SAMSUNG", "Samsung"),
            ("PANASONIC", "Panasonic"),
            ("HISENSE", "Hisense"),
            ("TOSHIBA", "Toshiba"),
            ("INSIGNIA", "Insignia"),
            ("KENMORE", "Kenmore"),
            ("MAGICCHEF", "Magic Chef"),
            ("SUBZERO", "Sub-Zero"),
            ("BOSCH", "Bosch"),
            ("SIEMENS", "Siemens"),
            ("MIELE", "Miele"),
            ("HAIER", "Haier"),
            ("AVANTI", "Avanti"),
            ("VIKING", "Viking"),
            ("SHARP", "Sharp"),
            ("AMANA", "Amana"),
            ("SMEG", "Smeg"),
            ("BEKO", "Beko"),
            ("LG", "LG"),
        ),
        key=lambda item: -len(item[0]),
    )
)

MODEL_MIN_SCORE = 0.95
SERIAL_MIN_SCORE = 0.9
ADJACENT_DISTANCE = 4

#: Confidence assigned to each extraction method, per field family.
CONFIDENCE = {
    "label_anchor": 0.9,
    "label_anchor_before": 0.85,
    "adjacent_line": 0.8,
    "known_brand": 0.8,
    "company_suffix": 0.7,
    "glued_anchor": 0.7,
    "heuristic_model": 0.6,
    "heuristic_serial": 0.5,
}


# ---------------------------------------------------------------------------
# Token scoring
# ---------------------------------------------------------------------------


def _is_noise(normalized: str) -> bool:
    return any(noise in normalized for noise in NOISE_SUBSTRINGS)


def _date_like(token: str) -> bool:
    return bool(_DATE_RE.search(token))


def _letter_digit_counts(normalized: str) -> tuple[int, int]:
    letters = sum(character.isalpha() for character in normalized)
    return letters, len(normalized) - letters


def model_score(token: str) -> float:
    """Score how much ``token`` looks like a model number (0 = not a candidate)."""
    normalized = normalize_alphanum(token)
    if not 5 <= len(normalized) <= 25:
        return 0.0
    if not any(character.isdigit() for character in normalized):
        return 0.0
    if not any(character.isalpha() for character in normalized):
        return 0.0
    if _is_noise(normalized) or _date_like(token):
        return 0.0

    letters, digits = _letter_digit_counts(normalized)
    score = 1.0
    if letters >= 2:
        score += 0.2
    if digits >= 3:
        score += 0.1
    if letters == 1 and digits >= 6:
        # Serial-like: one leading letter followed by a long digit run.
        score -= 0.3
    if len(normalized) < 6:
        score -= 0.2
    return round(score, 3)


def serial_score(token: str) -> float:
    """Score how much ``token`` looks like a serial number (0 = not a candidate)."""
    normalized = normalize_alphanum(token)
    if not 5 <= len(normalized) <= 30:
        return 0.0
    if not any(character.isdigit() for character in normalized):
        return 0.0
    if _is_noise(normalized) or _date_like(token):
        return 0.0

    letters, digits = _letter_digit_counts(normalized)
    score = 1.0
    if digits >= 5:
        score += 0.1
    if letters <= 2 and digits >= 5:
        score += 0.2
    if letters == 0:
        score += 0.3
    if letters >= 5 and digits <= 3:
        # Model-like: many letters, few digits.
        score -= 0.2
    return round(score, 3)


def _best_token(text: str, scorer, minimum: float) -> tuple[float, str] | None:
    best: tuple[float, str] | None = None
    for raw in TOKEN_RE.findall(text):
        token = raw.strip("._-/")
        if not token:
            continue
        score = scorer(token)
        if score < minimum:
            continue
        if best is None or score > best[0]:
            best = (score, token)
    return best


def _prefix_token(line: str, anchor_start: int, scorer, minimum: float) -> tuple[float, str] | None:
    """Best candidate on the same line *before* the anchor (right-aligned labels)."""
    return _best_token(line[:anchor_start], scorer, minimum)


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _field(
    value: str | None,
    confidence: float,
    source_line: str | None,
    method: str | None,
) -> dict[str, Any]:
    return {
        "value": value,
        "normalized": normalize_alphanum(value) if value else None,
        "confidence": round(confidence, 3),
        "source_line": source_line,
        "method": method,
    }


def _empty_field() -> dict[str, Any]:
    return _field(None, 0.0, None, None)


def _split_lines(raw_text: str) -> list[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def _pick(candidates: Sequence[tuple[float, int, str, str]]) -> tuple[float, int, str, str] | None:
    """Pick the highest score, breaking ties toward the earliest line."""
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], -item[1]))


# ---------------------------------------------------------------------------
# Model number
# ---------------------------------------------------------------------------


def _find_model(lines: Sequence[str]) -> dict[str, Any]:
    candidates: list[tuple[float, int, str, str]] = []

    for index, line in enumerate(lines):
        anchors = list(MODEL_ANCHOR_RE.finditer(line))
        for match in anchors:
            inline = _best_token(line[match.end() :], model_score, MODEL_MIN_SCORE)
            if inline:
                score, token = inline
                candidates.append((score, index, token, "label_anchor"))
                continue

            before = _prefix_token(line, match.start(), model_score, MODEL_MIN_SCORE)
            if before:
                score, token = before
                candidates.append((score - 0.01, index, token, "label_anchor_before"))
                continue

            found_forward = False
            for distance in range(1, ADJACENT_DISTANCE + 1):
                look = index + distance
                if look >= len(lines):
                    break
                found = _best_token(lines[look], model_score, MODEL_MIN_SCORE)
                if found:
                    score, token = found
                    candidates.append((score - distance * 0.02, look, token, "adjacent_line"))
                    found_forward = True
                    break

            # Some plates print the value above the anchor (``VALUE`` then ``MODEL``).
            # Only fall back to this when nothing follows the anchor, because an
            # adjacent serial can also score as a model.
            if not found_forward:
                for distance in (1, 2):
                    back = index - distance
                    if back < 0:
                        break
                    found = _best_token(lines[back], model_score, MODEL_MIN_SCORE)
                    if found:
                        score, token = found
                        candidates.append((score - distance * 0.03, back, token, "adjacent_line"))
                        break

        glued = GLUED_MODEL_RE.match(line)
        if glued and not anchors:
            for distance in range(1, ADJACENT_DISTANCE + 1):
                look = index + distance
                if look >= len(lines):
                    break
                found = _best_token(lines[look], model_score, MODEL_MIN_SCORE)
                if found:
                    score, token = found
                    candidates.append((score - distance * 0.02, look, token, "adjacent_line"))
                    break

    if not candidates:
        for index, line in enumerate(lines):
            found = _best_token(line, model_score, MODEL_MIN_SCORE)
            if found:
                score, token = found
                candidates.append((score, index, token, "heuristic_model"))

    picked = _pick(candidates)
    if picked is None:
        return _empty_field()

    _, index, token, method = picked
    return _field(token, CONFIDENCE[method], lines[index], method)


# ---------------------------------------------------------------------------
# Serial number
# ---------------------------------------------------------------------------


def _find_serial(lines: Sequence[str], exclude: set[str] | None = None) -> dict[str, Any]:
    exclude = exclude or set()
    candidates: list[tuple[float, int, str, str]] = []

    def acceptable(token: str, scorer, minimum: float) -> tuple[float, str] | None:
        if normalize_alphanum(token) in exclude:
            return None
        score = scorer(token)
        return (score, token) if score >= minimum else None

    for index, line in enumerate(lines):
        anchors = list(SERIAL_ANCHOR_RE.finditer(line))
        for match in anchors:
            inline = _best_token(line[match.end() :], serial_score, SERIAL_MIN_SCORE)
            if inline and normalize_alphanum(inline[1]) not in exclude:
                score, token = inline
                candidates.append((score, index, token, "label_anchor"))
                continue

            before = _prefix_token(line, match.start(), serial_score, SERIAL_MIN_SCORE)
            if before and normalize_alphanum(before[1]) not in exclude:
                score, token = before
                candidates.append((score - 0.01, index, token, "label_anchor_before"))
                continue

            for distance in range(1, ADJACENT_DISTANCE + 1):
                look = index + distance
                if look >= len(lines):
                    break
                found = acceptable(lines[look], serial_score, SERIAL_MIN_SCORE)
                if found:
                    score, token = found
                    candidates.append((score - distance * 0.03, look, token, "adjacent_line"))
                    break

            for distance in (1, 2):
                back = index - distance
                if back < 0:
                    break
                found = acceptable(lines[back], serial_score, SERIAL_MIN_SCORE)
                if found:
                    score, token = found
                    candidates.append((score - distance * 0.03, back, token, "adjacent_line"))
                    break

    if not candidates:
        for index, line in enumerate(lines):
            if SERIAL_ANCHOR_RE.search(line) or MODEL_ANCHOR_RE.search(line):
                continue
            found = acceptable(line, serial_score, SERIAL_MIN_SCORE)
            if found:
                score, token = found
                candidates.append((score, index, token, "heuristic_serial"))

    picked = _pick(candidates)
    if picked is None:
        return _empty_field()

    _, index, token, method = picked
    return _field(token, CONFIDENCE[method], lines[index], method)


# ---------------------------------------------------------------------------
# Brand
# ---------------------------------------------------------------------------


def _brand_line_ok(line: str) -> bool:
    if not 2 <= len(line) <= 60:
        return False
    if _URL_RE.search(line) or _PHONE_RE.search(line):
        return False
    upper = line.upper()
    if any(
        noise in upper
        for noise in (
            "MODEL",
            "MODELE",
            "SERIAL",
            "DATE",
            "VOLT",
            "REFRIGERANT",
            "PATENT",
            "ENERGY",
            "LISTED",
            "WARRANTY",
        )
    ):
        return False
    if sum(character.isdigit() for character in line) >= 4:
        return False
    return True


def _match_known_brand(line: str, brands: Sequence[tuple[str, str]]) -> tuple[int, str] | None:
    upper = line.upper()
    normalized = normalize_alphanum(line)
    for pattern, canonical in brands:
        if len(pattern) <= 3:
            if re.search(rf"(?<![A-Z0-9]){re.escape(pattern)}(?![A-Z0-9])", upper):
                return len(pattern), canonical
        elif pattern in normalized:
            return len(pattern), canonical
    return None


def _company_line(line: str) -> bool:
    if not _brand_line_ok(line):
        return False
    return bool(
        re.search(
            r"\b(?:COMPANY|LTD\.?|LIMITED|INC\.?|CORP\.?|CORPORATION|PRODUCTS)\s*$",
            line.upper(),
        )
    )


def _find_brand(lines: Sequence[str], brands: Sequence[tuple[str, str]]) -> dict[str, Any]:
    # 1. Explicit brand / manufacturer anchors.
    for index, line in enumerate(lines):
        match = BRAND_ANCHOR_RE.search(line)
        if not match:
            continue
        rest = line[match.end() :].strip(" :#-")
        if rest and _brand_line_ok(rest):
            return _field(rest, CONFIDENCE["label_anchor"], line, "label_anchor")
        for distance in (1, -1):
            look = index + distance
            if 0 <= look < len(lines) and _brand_line_ok(lines[look]):
                return _field(lines[look], CONFIDENCE["adjacent_line"], lines[look], "adjacent_line")

    # 2. Known brand names, preferring the longest match.
    best: tuple[int, int, str, str] | None = None
    for index, line in enumerate(lines):
        if not _brand_line_ok(line):
            continue
        matched = _match_known_brand(line, brands)
        if not matched:
            continue
        pattern_length, canonical = matched
        if best is None or (pattern_length, -index) > (best[0], -best[1]):
            best = (pattern_length, index, canonical, line)

    if best is not None:
        return _field(best[2], CONFIDENCE["known_brand"], best[3], "known_brand")

    # 3. Lines ending in a company suffix (unknown brands).
    for line in lines:
        if _company_line(line):
            return _field(line.strip(), CONFIDENCE["company_suffix"], line, "company_suffix")

    # 4. A brand glued to the MODEL anchor, e.g. ``KitchenAidMOD``.
    for line in lines:
        glued = GLUED_MODEL_RE.match(line)
        if glued and _brand_line_ok(glued.group("prefix")):
            return _field(
                glued.group("prefix").strip(), CONFIDENCE["glued_anchor"], line, "glued_anchor"
            )

    return _empty_field()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _brand_table(extra: Iterable[tuple[str, str]] | None) -> tuple[tuple[str, str], ...]:
    if not extra:
        return KNOWN_BRANDS
    merged = dict(KNOWN_BRANDS)
    merged.update({pattern.upper(): canonical for pattern, canonical in extra})
    return tuple(sorted(merged.items(), key=lambda item: -len(item[0])))


def parse_fields(
    raw_text: str,
    *,
    known_brands: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Parse structured fields from raw OCR text.

    ``known_brands`` optionally extends the built-in brand list with
    ``(normalized_pattern, canonical_name)`` pairs.
    """
    brands = _brand_table(known_brands)
    lines = _split_lines(raw_text or "")

    brand = _find_brand(lines, brands)
    model_number = _find_model(lines)

    exclude = {model_number["normalized"]} if model_number["normalized"] else set()
    serial_number = _find_serial(lines, exclude=exclude)

    used = {
        field["source_line"]
        for field in (brand, model_number, serial_number)
        if field["source_line"]
    }
    return {
        "brand": brand,
        "model_number": model_number,
        "serial_number": serial_number,
        "unparsed_lines": [line for line in lines if line not in used],
    }


def parse_ocr_result(
    result: dict[str, Any],
    *,
    known_brands: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Parse fields from an :func:`scripts.label_ocr.extract_text` result."""
    parsed = parse_fields(result.get("raw_text", ""), known_brands=known_brands)
    parsed["image_path"] = result.get("image_path")
    parsed["ocr"] = {
        "backend": result.get("backend"),
        "confidence": result.get("confidence"),
        "preprocessing": result.get("preprocessing"),
    }
    return parsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_fields(payload: dict[str, Any]) -> None:
    if payload.get("image_path"):
        print(f"Image:  {payload['image_path']}")
        ocr = payload.get("ocr") or {}
        print(f"OCR:    {ocr.get('backend')} (confidence {ocr.get('confidence')})")
        print()

    for key in ("brand", "model_number", "serial_number"):
        field = payload[key]
        value = field["value"] if field["value"] is not None else "(not found)"
        print(f"{key:14} {value}")
        if field["value"] is not None:
            print(
                f"{'':14} confidence {field['confidence']:.2f} · "
                f"{field['method']} · {field['source_line']!r}"
            )


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Parse brand/model/serial from label OCR text")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Run OCR on this image, then parse it")
    source.add_argument("--text", help="Parse this raw OCR text")
    source.add_argument("--text-file", help="Parse raw OCR text from a file")
    parser.add_argument(
        "--engine",
        default="auto",
        choices=("auto", "paddleocr", "tesseract"),
        help="OCR engine for --image: auto, paddleocr, or tesseract",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report")
    parser.add_argument("--output", help="Write JSON results to this file")
    args = parser.parse_args(argv)

    try:
        if args.image:
            from scripts.label_ocr import extract_text
            from scripts.ocr_backends import get_backend

            backend = get_backend(args.engine)
            payload = parse_ocr_result(extract_text(args.image, backend=backend))
        elif args.text_file:
            text = Path(args.text_file).read_text(encoding="utf-8")
            payload = parse_fields(text)
        else:
            payload = parse_fields(args.text)
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
        _print_fields(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
