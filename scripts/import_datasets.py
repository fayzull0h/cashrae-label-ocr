"""NRCan Appliance Dataset Importer.

Imports Natural Resources Canada (NRCan) appliance CSV datasets into SQLite.
Organizes data into category-specific tables and shared metadata tables.
"""

from pathlib import Path
import re
import sqlite3
import pandas as pd

# Project paths (relative to project root)
RAW_DIR = Path("data/raw")
DB_PATH = Path("database/appliances.db")

# Ensure database directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Known NRCan dataset prefixes and canonical category names
CATEGORY_MAP = {
    # NRCan standard dataset numeric prefixes
    "001": "clothes_dryers",
    "002": "clothes_washer_dryers",
    "003": "clothes_washers",
    "004": "cooktops",
    "005": "dishwashers",
    "006": "freezers",
    "007": "ovens",
    "008": "ranges",
    "009": "refrigerators",
    "010": "room_air_conditioners",
    # English names and slug variations
    "clothes_dryers": "clothes_dryers",
    "clothes_dryer": "clothes_dryers",
    "clothes_washer_dryers": "clothes_washer_dryers",
    "clothes_washers_dryers": "clothes_washer_dryers",
    "clothes_washer_dryer": "clothes_washer_dryers",
    "clothes_washers": "clothes_washers",
    "clothes_washer": "clothes_washers",
    "cooktops": "cooktops",
    "cooktop": "cooktops",
    "dishwashers": "dishwashers",
    "dishwasher": "dishwashers",
    "freezers": "freezers",
    "freezer": "freezers",
    "ovens": "ovens",
    "oven": "ovens",
    "ranges": "ranges",
    "range": "ranges",
    "refrigerators": "refrigerators",
    "refrigerator": "refrigerators",
    "refrigerators_wine_chillers": "refrigerators",
    "room_air_conditioners": "room_air_conditioners",
    "room_air_conditioner": "room_air_conditioners",
    "air_conditioners": "room_air_conditioners",
    "air_conditioner": "room_air_conditioners",
}

# Metadata column mappings
INFO_COLUMN_MAPPING = {
    "Filename / Nom du fichier": "filename",
    "Product Name English / Nom du produit - anglais": "product_name_english",
    "Product Name French / Nom du produit - français": "product_name_french",
    "Structure Change / Modification - structure": "structure_change_date",
    "Data Change / Modification - données": "data_change_date",
}

DICT_COLUMN_MAPPING = {
    "Field Name / Nom du champ": "field_name",
    "Type / Type": "data_type",
    "English Label / Étiquette - anglais": "english_label",
    "French Label / Étiquette - français": "french_label",
    "Language / Langue": "language",
}


def clean_name(name: str) -> str:
    """Convert a filename, folder, or column name into a SQLite-friendly snake_case identifier."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def read_csv_file(file_path: Path) -> pd.DataFrame:
    """Read a CSV file trying utf-8, cp1252, and latin1 encodings in order.

    Preserves accents and French characters across varied dataset encodings.
    """
    encodings = ["utf-8", "cp1252", "latin1"]

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not determine encoding for {file_path} after trying {encodings}")


def detect_category(file_path: Path, raw_dir: Path = RAW_DIR) -> str:
    """Determine the normalized appliance category from directory structure and file context.

    Handles nested extraction directories (e.g. 001_clothes-dryers_secheuses/001_clothes-dryers_secheuses/...)
    by inspecting directory components from deepest to highest, matching known NRCan prefixes or names,
    and falling back to the filename prefix if needed.
    """
    try:
        rel_path = file_path.resolve().relative_to(raw_dir.resolve())
        dir_parts = list(rel_path.parts[:-1])
    except ValueError:
        dir_parts = list(file_path.parent.parts)

    # 1. Search directory parts (from innermost to outermost)
    for part in reversed(dir_parts):
        cleaned = clean_name(part)
        if cleaned in CATEGORY_MAP:
            return CATEGORY_MAP[cleaned]

        # Match NRCan pattern: e.g. "001_clothes_dryers_secheuses"
        match = re.match(r"^(\d{3})_(.+)$", cleaned)
        if match:
            prefix, rest = match.groups()
            if prefix in CATEGORY_MAP:
                return CATEGORY_MAP[prefix]

            # Try prefix tokens before French translation suffix
            tokens = rest.split("_")
            for length in range(len(tokens), 0, -1):
                candidate = "_".join(tokens[:length])
                if candidate in CATEGORY_MAP:
                    return CATEGORY_MAP[candidate]
            return rest

    # 2. Check for known category substring in any directory part
    for part in reversed(dir_parts):
        cleaned = clean_name(part)
        for key, cat in CATEGORY_MAP.items():
            if not key.isdigit() and key in cleaned:
                return cat

    # 3. Fallback: inspect filename prefix (e.g., '001_Data.csv' -> '001')
    fn_match = re.match(r"^(\d{3})_", file_path.name)
    if fn_match and fn_match.group(1) in CATEGORY_MAP:
        return CATEGORY_MAP[fn_match.group(1)]

    # Final fallback: use cleaned immediate parent folder name
    return clean_name(file_path.parent.name) or "unknown"


def classify_file_type(file_path: Path) -> str:
    """Classify NRCan CSV files into DATA, DICTIONARY, INFORMATION, or UNKNOWN."""
    name = file_path.name.lower()
    if re.search(r"_data(_donn[eé]es)?\.csv$", name) or name.endswith("_data.csv"):
        return "DATA"
    elif name.endswith("_dictionary_dictionnaire.csv") or "dictionary" in name:
        return "DICTIONARY"
    elif name.endswith("_information.csv") or "information" in name:
        return "INFORMATION"
    return "UNKNOWN"


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the SQLite database."""
    cur = connection.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


def cleanup_legacy_tables(connection: sqlite3.Connection) -> None:
    """Drop obsolete per-file tables (e.g., 001_data, 001_information) from earlier importer versions."""
    cur = connection.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    legacy_pattern = re.compile(r"^\d{3}_(data|dictionary|information)", re.IGNORECASE)
    legacy_tables = [t for t in tables if legacy_pattern.match(t)]

    if legacy_tables:
        print(f"Cleaning up {len(legacy_tables)} obsolete tables from earlier schema...")
        with connection:
            for table in legacy_tables:
                connection.execute(f'DROP TABLE IF EXISTS "{table}"')
        print("Legacy tables removed.")


def import_data_file(file_path: Path, category: str, connection: sqlite3.Connection) -> int:
    """Import an appliance _Data.csv file into its category-specific table.

    Preserves original column names, adds source_file and appliance_category columns,
    and replaces existing table to ensure repeatability without duplication.
    """
    df = read_csv_file(file_path)

    # Preserve original columns and append metadata columns
    df["source_file"] = file_path.name
    df["appliance_category"] = category

    with connection:
        df.to_sql(category, connection, if_exists="replace", index=False)

    return len(df)


def import_information_file(file_path: Path, category: str, connection: sqlite3.Connection) -> int:
    """Import an _Information.csv file into the shared `datasets` metadata table.

    Avoids duplication by refreshing records for the appliance category.
    """
    df = read_csv_file(file_path)

    # Map column headers flexibly
    mapped_columns = {}
    for col in df.columns:
        col_lower = col.lower()
        for orig, target in INFO_COLUMN_MAPPING.items():
            if orig.lower() in col_lower or col_lower in orig.lower():
                mapped_columns[col] = target
                break

    df = df.rename(columns=mapped_columns).copy()
    df["appliance_category"] = category
    df["source_file"] = file_path.name

    desired_columns = [
        "appliance_category",
        "filename",
        "product_name_english",
        "product_name_french",
        "structure_change_date",
        "data_change_date",
        "source_file",
    ]
    df = df[[c for c in desired_columns if c in df.columns]]

    with connection:
        if table_exists(connection, "datasets"):
            connection.execute("DELETE FROM datasets WHERE appliance_category = ?", (category,))
        df.to_sql("datasets", connection, if_exists="append", index=False)

    return len(df)


def import_dictionary_file(file_path: Path, category: str, connection: sqlite3.Connection) -> int:
    """Import a _Dictionary_Dictionnaire.csv file into the shared `field_dictionary` metadata table.

    Associates each row with the appliance category and refreshes without duplicating records.
    """
    df = read_csv_file(file_path)

    # Map column headers flexibly
    mapped_columns = {}
    for col in df.columns:
        col_lower = col.lower()
        for orig, target in DICT_COLUMN_MAPPING.items():
            if orig.lower() in col_lower or col_lower in orig.lower():
                mapped_columns[col] = target
                break

    df = df.rename(columns=mapped_columns).copy()
    df["appliance_category"] = category
    df["source_file"] = file_path.name

    desired_columns = [
        "appliance_category",
        "field_name",
        "data_type",
        "english_label",
        "french_label",
        "language",
        "source_file",
    ]
    df = df[[c for c in desired_columns if c in df.columns]]

    with connection:
        if table_exists(connection, "field_dictionary"):
            connection.execute("DELETE FROM field_dictionary WHERE appliance_category = ?", (category,))
        df.to_sql("field_dictionary", connection, if_exists="append", index=False)

    return len(df)


def import_file(file_path: Path, connection: sqlite3.Connection) -> None:
    """Import a single CSV or Excel file into SQLite based on its detected type."""
    category = detect_category(file_path)
    ext = file_path.suffix.lower()

    if ext == ".csv":
        file_type = classify_file_type(file_path)

        if file_type == "DATA":
            print(f"Importing appliance data: {file_path.name} -> table '{category}'")
            count = import_data_file(file_path, category, connection)
            print(f"  -> Populated '{category}' with {count} records")

        elif file_type == "DICTIONARY":
            print(f"Importing field dictionary: {file_path.name} -> table 'field_dictionary'")
            count = import_dictionary_file(file_path, category, connection)
            print(f"  -> Added {count} field definitions for '{category}'")

        elif file_type == "INFORMATION":
            print(f"Importing dataset metadata: {file_path.name} -> table 'datasets'")
            count = import_information_file(file_path, category, connection)
            print(f"  -> Added {count} metadata record for '{category}'")

        else:
            print(f"Skipping unrecognized CSV format: {file_path}")

    elif ext in [".xlsx", ".xls"]:
        print(f"Reading Excel workbook: {file_path.name}")
        sheets = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, df in sheets.items():
            table_name = clean_name(f"{category}_{file_path.stem}_{sheet_name}")
            df["source_file"] = file_path.name
            df["source_sheet"] = sheet_name
            df["appliance_category"] = category
            with connection:
                df.to_sql(table_name, connection, if_exists="replace", index=False)
            print(f"  -> Imported sheet '{sheet_name}' into table '{table_name}' ({len(df)} records)")


def main() -> None:
    """Main importer entry point."""
    print("=" * 60)
    print("Starting NRCan Appliance Dataset Importer")
    print(f"Source Directory:   {RAW_DIR.resolve()}")
    print(f"SQLite Database:    {DB_PATH.resolve()}")
    print("=" * 60)

    connection = sqlite3.connect(DB_PATH)

    try:
        # Clean up legacy per-file tables from older runs if present
        cleanup_legacy_tables(connection)

        supported_extensions = {".csv", ".xlsx", ".xls"}
        files_to_import = [
            p for p in RAW_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in supported_extensions
        ]

        if not files_to_import:
            print(f"No supported files found in {RAW_DIR}")
            return

        print(f"Found {len(files_to_import)} files to process.\n")

        # Sort files so data, dictionary, and information are processed in a consistent order
        files_to_import.sort(key=lambda p: (detect_category(p), p.name))

        success_count = 0
        error_count = 0

        for file_path in files_to_import:
            try:
                import_file(file_path, connection)
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"[ERROR] Failed to import {file_path}: {e}")

        print("\n" + "=" * 60)
        print("Import Summary")
        print(f"Successfully processed: {success_count} files")
        if error_count:
            print(f"Errors encountered:     {error_count} files")
        print(f"Database location:      {DB_PATH.resolve()}")
        print("=" * 60)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
