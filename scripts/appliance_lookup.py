"""NRCan Appliance Retrieval Engine (Phase 1).

Retrieves appliance records and annual energy consumption from Natural Resources
Canada (NRCan) datasets stored in SQLite (`database/appliances.db`).

Provides:
- Normalization of model numbers and manufacturer names (whitespace, hyphens, punctuation).
- Two-stage matching: Stage A (Exact) and Stage B (Fuzzy using SequenceMatcher).
- Handling of heterogeneous energy column names across all 10 appliance categories.
- Structured, transparent return format.
"""

from difflib import SequenceMatcher
from pathlib import Path
import re
import sqlite3
from typing import Any

# Default database path (relative to project root)
DEFAULT_DB_PATH = Path("database/appliances.db")

# All 10 appliance data tables
ALL_APPLIANCE_TABLES = [
    "clothes_dryers",
    "clothes_washer_dryers",
    "clothes_washers",
    "cooktops",
    "dishwashers",
    "freezers",
    "ovens",
    "ranges",
    "refrigerators",
    "room_air_conditioners",
]

# Mapping from table name to its primary annual energy consumption column
# Note: room_air_conditioners reports EE_RATIO (EER) and cooling capacity; it has no AEC in kWh/year.
ENERGY_COLUMNS: dict[str, str | None] = {
    "clothes_dryers": "AEC",
    "clothes_washer_dryers": "AEC",
    "clothes_washers": "MOD_J1_AEC",
    "cooktops": "AEC_03",
    "dishwashers": "TAEC_C373_04",
    "freezers": "AEC",
    "ovens": "AEC_03",
    "ranges": "AEC_03",
    "refrigerators": "AEC",
    "room_air_conditioners": None,
}

# Standard unit for annual energy consumption across NRCan datasets
DEFAULT_ENERGY_UNIT = "kWh/year"

# Display names for appliance categories
CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "clothes_dryers": "Clothes dryers",
    "clothes_washer_dryers": "Clothes washer-dryers",
    "clothes_washers": "Clothes washers",
    "cooktops": "Cooktops",
    "dishwashers": "Dishwashers",
    "freezers": "Freezers",
    "ovens": "Ovens",
    "ranges": "Ranges",
    "refrigerators": "Refrigerators and wine chillers",
    "room_air_conditioners": "Room air conditioners",
}

# Fuzzy matching thresholds
# MIN_FUZZY_CONFIDENCE: Candidates with similarity below this score are discarded
MIN_FUZZY_CONFIDENCE = 0.70

# HIGH_CONFIDENCE_THRESHOLD: Score indicating a reliable fuzzy match (e.g. single character typo/OCR error)
HIGH_CONFIDENCE_THRESHOLD = 0.85

# AMBIGUITY_MARGIN: If the score difference between candidate 1 and candidate 2 is <= this margin,
# results are flagged as ambiguous rather than arbitrarily selecting one.
AMBIGUITY_MARGIN = 0.04


def normalize_alphanum(text: str | None) -> str:
    """Normalize text by converting to uppercase and stripping all non-alphanumeric characters.

    Handles None and empty inputs safely.
    Examples:
        'ABC-123' -> 'ABC123'
        'ABC 123' -> 'ABC123'
        'abc_123' -> 'ABC123'
        'PCKS433ET#**' -> 'PCKS433ET'
    """
    if text is None:
        return ""
    # Strip whitespace, convert to uppercase, keep only letters and digits
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def normalize_model_number(model: str | None) -> str:
    """Normalize an appliance model number."""
    return normalize_alphanum(model)


def normalize_manufacturer(manufacturer: str | None) -> str:
    """Normalize an appliance manufacturer or brand name."""
    return normalize_alphanum(manufacturer)


def normalize_category(category: str | None) -> str | None:
    """Normalize category input into a valid database table name.

    Handles case variations, hyphens, numbers, and common aliases.
    Returns None if category is None or empty.
    Raises ValueError if category string is unrecognized.
    """
    if category is None:
        return None

    cleaned = str(category).strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"^0*\d+_", "", cleaned)  # strip numeric prefixes like '001_'

    # Direct match
    if cleaned in ALL_APPLIANCE_TABLES:
        return cleaned

    # Alias mappings
    alias_map = {
        "dryer": "clothes_dryers",
        "dryers": "clothes_dryers",
        "clothes_dryer": "clothes_dryers",
        "washer": "clothes_washers",
        "washers": "clothes_washers",
        "clothes_washer": "clothes_washers",
        "washer_dryer": "clothes_washer_dryers",
        "washer_dryers": "clothes_washer_dryers",
        "clothes_washer_dryer": "clothes_washer_dryers",
        "cooktop": "cooktops",
        "dishwasher": "dishwashers",
        "freezer": "freezers",
        "oven": "ovens",
        "range": "ranges",
        "refrigerator": "refrigerators",
        "refrigerators_and_wine_chillers": "refrigerators",
        "refrigerators_wine_chillers": "refrigerators",
        "air_conditioner": "room_air_conditioners",
        "air_conditioners": "room_air_conditioners",
        "room_air_conditioner": "room_air_conditioners",
    }

    if cleaned in alias_map:
        return alias_map[cleaned]

    raise ValueError(
        f"Invalid appliance category '{category}'. "
        f"Valid categories: {', '.join(ALL_APPLIANCE_TABLES)}"
    )


def get_db_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection configured with the NORMALIZE custom function and Row factory."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"Database not found at: {path.resolve()}")

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    # Register custom SQL function so queries can compare normalized values on the fly
    conn.create_function("NORMALIZE", 1, normalize_alphanum)
    return conn


def _extract_energy_consumption(row_dict: dict[str, Any], table: str) -> tuple[float | None, str | None, bool, str | None]:
    """Extract annual energy consumption, unit, availability flag, and notes for a record."""
    col = ENERGY_COLUMNS.get(table)

    if col is None:
        note = (
            f"Category '{table}' does not report Annual Energy Consumption (kWh/year) in NRCan datasets. "
            f"NRCan reports efficiency ratio (EE_RATIO = {row_dict.get('EE_RATIO')}) instead."
        )
        return None, None, False, note

    val = row_dict.get(col)

    if val is None or str(val).strip() == "" or str(val).lower() == "none":
        note = f"Annual energy consumption column '{col}' is missing or null for this record."
        return None, None, False, note

    try:
        numeric_val = float(val)
        return numeric_val, DEFAULT_ENERGY_UNIT, True, None
    except (ValueError, TypeError):
        note = f"Could not convert energy consumption value '{val}' to numeric."
        return None, None, False, note


def _format_candidate(
    row: sqlite3.Row | dict[str, Any],
    table: str,
    confidence: float,
    match_type: str,
    db_name: str,
) -> dict[str, Any]:
    """Format an appliance row into a candidate dictionary."""
    row_dict = dict(row)
    brand = row_dict.get("BRAND_NAME", "")
    raw_model = row_dict.get("MODEL_NUM_1", "")
    source_file = row_dict.get("source_file", "")
    rec_ref_key = row_dict.get("REC_REF_KEY")

    aec, unit, available, note = _extract_energy_consumption(row_dict, table)

    candidate = {
        "manufacturer": brand,
        "model_number": raw_model,
        "normalized_model": normalize_alphanum(raw_model),
        "category": table,
        "category_display": CATEGORY_DISPLAY_NAMES.get(table, table),
        "annual_energy_consumption": aec,
        "energy_consumption_available": available,
        "unit": unit,
        "match_confidence": round(confidence, 4),
        "match_type": match_type,
        "source": {
            "database": db_name,
            "table": table,
            "source_file": source_file,
            "rec_ref_key": rec_ref_key,
        },
        "record": row_dict,
    }
    if note:
        candidate["notes"] = note
    return candidate


def _search_exact(
    conn: sqlite3.Connection,
    target_tables: list[str],
    norm_mfg: str,
    norm_model: str,
    db_name: str,
) -> list[dict[str, Any]]:
    """Stage A: Execute exact normalized search across target tables."""
    matches: list[dict[str, Any]] = []

    for table in target_tables:
        cur = conn.cursor()
        if norm_mfg:
            query = f"""
                SELECT * FROM "{table}"
                WHERE NORMALIZE(MODEL_NUM_1) = ?
                  AND NORMALIZE(BRAND_NAME) = ?;
            """
            cur.execute(query, (norm_model, norm_mfg))
        else:
            query = f"""
                SELECT * FROM "{table}"
                WHERE NORMALIZE(MODEL_NUM_1) = ?;
            """
            cur.execute(query, (norm_model,))

        rows = cur.fetchall()
        for row in rows:
            match_type = "exact" if norm_mfg else "exact_model_only"
            matches.append(_format_candidate(row, table, 1.0, match_type, db_name))

    return matches


def _search_fuzzy(
    conn: sqlite3.Connection,
    target_tables: list[str],
    norm_mfg: str,
    norm_model: str,
    db_name: str,
) -> list[dict[str, Any]]:
    """Stage B: Execute fuzzy similarity search across target tables using SequenceMatcher."""
    candidates: list[dict[str, Any]] = []

    for table in target_tables:
        cur = conn.cursor()
        # If manufacturer provided, first prioritize models matching that manufacturer
        if norm_mfg:
            cur.execute(f'SELECT * FROM "{table}" WHERE NORMALIZE(BRAND_NAME) = ?;', (norm_mfg,))
            rows = cur.fetchall()
            # If no rows under this exact normalized manufacturer, broaden to all rows in table
            if not rows:
                cur.execute(f'SELECT * FROM "{table}";')
                rows = cur.fetchall()
        else:
            cur.execute(f'SELECT * FROM "{table}";')
            rows = cur.fetchall()

        for row in rows:
            row_dict = dict(row)
            db_model_norm = normalize_alphanum(row_dict.get("MODEL_NUM_1"))
            db_mfg_norm = normalize_alphanum(row_dict.get("BRAND_NAME"))

            # Calculate model number similarity
            model_score = SequenceMatcher(None, norm_model, db_model_norm).ratio()

            if model_score < MIN_FUZZY_CONFIDENCE:
                continue

            # Weight by manufacturer alignment if manufacturer was requested
            final_confidence = model_score
            if norm_mfg:
                if db_mfg_norm == norm_mfg:
                    final_confidence = model_score
                else:
                    mfg_score = SequenceMatcher(None, norm_mfg, db_mfg_norm).ratio()
                    # Slight penalty if manufacturer differs
                    final_confidence = (model_score * 0.85) + (mfg_score * 0.15)

            if final_confidence >= MIN_FUZZY_CONFIDENCE:
                candidates.append(_format_candidate(row, table, final_confidence, "fuzzy", db_name))

    # Sort descending by confidence score
    candidates.sort(key=lambda c: c["match_confidence"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Exact-only lookups
# ---------------------------------------------------------------------------
# ``lookup_appliance`` falls through to the fuzzy stage, which is the right
# behaviour for retrieval but the wrong signal for candidate validation: a fuzzy
# hit does not mean "this corrected model exists". The helpers below expose the
# exact stage on its own and never run SequenceMatcher.


def find_exact_matches(
    model_number: str,
    manufacturer: str | None = None,
    category: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return exact normalized matches only (no fuzzy fallback).

    Mirrors Stage A of :func:`lookup_appliance`. Returns an empty list for an
    empty model number; raises ``ValueError`` for an unknown category.
    """
    norm_model = normalize_model_number(model_number)
    if not norm_model:
        return []

    norm_category = normalize_category(category)
    target_tables = [norm_category] if norm_category else ALL_APPLIANCE_TABLES
    norm_mfg = normalize_manufacturer(manufacturer)
    db_path_obj = Path(db_path)

    conn = get_db_connection(db_path_obj)
    try:
        return _search_exact(conn, target_tables, norm_mfg, norm_model, db_path_obj.name)
    finally:
        conn.close()


def find_exact_matches_batch(
    model_numbers: list[str] | tuple[str, ...] | set[str],
    manufacturer: str | None = None,
    category: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, list[dict[str, Any]]]:
    """Exact-match many model numbers in one pass, keyed by normalized model.

    Every requested normalized model is present in the result, mapped to an
    (often empty) list of matching records. Useful for validating a batch of
    OCR-error candidates without opening a connection per candidate.
    """
    normalized: dict[str, list[dict[str, Any]]] = {}
    for model in model_numbers:
        norm_model = normalize_model_number(model)
        if norm_model:
            normalized.setdefault(norm_model, [])

    if not normalized:
        return {}

    norm_category = normalize_category(category)
    target_tables = [norm_category] if norm_category else ALL_APPLIANCE_TABLES
    norm_mfg = normalize_manufacturer(manufacturer)
    db_path_obj = Path(db_path)

    placeholders = ", ".join("?" for _ in normalized)
    conn = get_db_connection(db_path_obj)
    try:
        for table in target_tables:
            # Two-phase lookup: find matching rowids first (narrow scan), then
            # fetch the full rows. ``SELECT *`` combined with a NORMALIZE()
            # predicate scans wide NRCan tables far more slowly.
            key_query = f'SELECT rowid FROM "{table}" WHERE NORMALIZE(MODEL_NUM_1) IN ({placeholders})'
            params: list[Any] = list(normalized)
            if norm_mfg:
                key_query += " AND NORMALIZE(BRAND_NAME) = ?"
                params.append(norm_mfg)

            rowids = [row[0] for row in conn.execute(key_query, params)]
            if not rowids:
                continue

            match_type = "exact" if norm_mfg else "exact_model_only"
            for start in range(0, len(rowids), 900):
                chunk = rowids[start : start + 900]
                row_placeholders = ", ".join("?" for _ in chunk)
                rows = conn.execute(
                    f'SELECT * FROM "{table}" WHERE rowid IN ({row_placeholders})',
                    chunk,
                )
                for row in rows:
                    row_dict = dict(row)
                    key = normalize_alphanum(row_dict.get("MODEL_NUM_1"))
                    if key in normalized:
                        normalized[key].append(
                            _format_candidate(row, table, 1.0, match_type, db_path_obj.name)
                        )
    finally:
        conn.close()

    return normalized


def lookup_appliance(
    manufacturer: str | None = None,
    model_number: str = "",
    category: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Look up an appliance in the NRCan database by model number and optional manufacturer.

    Args:
        manufacturer: Brand/manufacturer name (e.g. "GE", "Whirlpool"). Optional.
        model_number: Model number string (e.g. "PCKS-433-ET", "ABC 123"). Required.
        category: Appliance category to restrict search (e.g. "clothes_dryers"). Optional.
        db_path: Path to the SQLite database.

    Returns:
        Structured dictionary containing match status, confidence, appliance details,
        annual energy consumption, and source references.
    """
    # 1. Validate inputs
    if model_number is None or str(model_number).strip() == "":
        return {
            "status": "invalid_input",
            "message": "Model number is required and cannot be empty.",
            "annual_energy_consumption": None,
            "energy_consumption_available": False,
            "match_confidence": 0.0,
        }

    try:
        norm_category = normalize_category(category)
    except ValueError as e:
        return {
            "status": "invalid_input",
            "message": str(e),
            "annual_energy_consumption": None,
            "energy_consumption_available": False,
            "match_confidence": 0.0,
        }

    norm_model = normalize_model_number(model_number)
    norm_mfg = normalize_manufacturer(manufacturer)

    if not norm_model:
        return {
            "status": "invalid_input",
            "message": f"Model number '{model_number}' contains no alphanumeric characters.",
            "annual_energy_consumption": None,
            "energy_consumption_available": False,
            "match_confidence": 0.0,
        }

    # Determine tables to search
    target_tables = [norm_category] if norm_category else ALL_APPLIANCE_TABLES
    db_path_obj = Path(db_path)
    db_name = db_path_obj.name

    conn = get_db_connection(db_path_obj)

    try:
        # =====================================================================
        # Stage A: Exact Matching
        # =====================================================================
        exact_matches = _search_exact(conn, target_tables, norm_mfg, norm_model, db_name)

        if exact_matches:
            # If exactly 1 match found
            if len(exact_matches) == 1:
                result = exact_matches[0]
                result["status"] = "exact_match"
                result["is_exact"] = True
                if not norm_mfg:
                    result["warning"] = "Manufacturer was not specified; match was made on model number only."
                return result

            # Multiple exact matches found (e.g., model shared across categories or duplicate records)
            return {
                "status": "ambiguous_match",
                "message": f"Found {len(exact_matches)} exact matches for model '{model_number}'.",
                "is_exact": True,
                "match_confidence": 1.0,
                "match_type": "exact",
                "annual_energy_consumption": None,
                "energy_consumption_available": False,
                "candidates": exact_matches,
            }

        # =====================================================================
        # Stage B: Fuzzy Matching
        # =====================================================================
        fuzzy_candidates = _search_fuzzy(conn, target_tables, norm_mfg, norm_model, db_name)

        if not fuzzy_candidates:
            return {
                "status": "not_found",
                "message": f"No matching appliance found for model '{model_number}'.",
                "annual_energy_consumption": None,
                "energy_consumption_available": False,
                "match_confidence": 0.0,
            }

        top_candidate = fuzzy_candidates[0]
        top_score = top_candidate["match_confidence"]

        # Check if the match is sufficiently confident
        if top_score < MIN_FUZZY_CONFIDENCE:
            return {
                "status": "not_found",
                "message": f"No matching appliance found above confidence threshold ({MIN_FUZZY_CONFIDENCE}).",
                "annual_energy_consumption": None,
                "energy_consumption_available": False,
                "match_confidence": top_score,
            }

        # Check for ambiguity among top candidates
        if len(fuzzy_candidates) > 1:
            second_score = fuzzy_candidates[1]["match_confidence"]
            if (top_score - second_score) <= AMBIGUITY_MARGIN:
                return {
                    "status": "ambiguous_match",
                    "message": (
                        f"Multiple candidate appliances found with similar confidence scores "
                        f"({top_score:.2f} vs {second_score:.2f})."
                    ),
                    "is_exact": False,
                    "match_confidence": top_score,
                    "match_type": "fuzzy",
                    "annual_energy_consumption": None,
                    "energy_consumption_available": False,
                    "candidates": fuzzy_candidates[:10],
                }

        # Single clear fuzzy match
        result = dict(top_candidate)
        result["status"] = "fuzzy_match"
        result["is_exact"] = False
        result["candidates"] = fuzzy_candidates[:5]
        return result

    finally:
        conn.close()


def get_annual_energy_consumption(
    manufacturer: str | None = None,
    model_number: str = "",
    category: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Retrieve an appliance's annual energy consumption (kWh/year).

    Reuses `lookup_appliance()` to perform normalization, exact matching, and fuzzy matching.

    Args:
        manufacturer: Brand/manufacturer name. Optional.
        model_number: Model number string. Required.
        category: Appliance category restriction. Optional.
        db_path: Path to database.

    Returns:
        Structured dictionary containing:
            - status: 'exact_match', 'fuzzy_match', 'ambiguous_match', 'not_found', or 'invalid_input'
            - annual_energy_consumption: Float kWh/year value if available, else None
            - unit: 'kWh/year' or None
            - energy_consumption_available: Boolean flag
            - match_confidence: Similarity score (0.0 to 1.0)
            - category, manufacturer, model_number, and source metadata
    """
    return lookup_appliance(
        manufacturer=manufacturer,
        model_number=model_number,
        category=category,
        db_path=db_path,
    )
