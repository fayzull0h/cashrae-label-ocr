"""Refresh a category's ENERGY STAR certified product snapshot."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.energy_star_config import ENERGY_STAR_DATASETS, EnergyStarDataset


DB_PATH = Path("database/appliances.db")
PAGE_SIZE = 1000
REQUIRED_FIELDS = {"pd_id", "brand_name", "model_number"}


def _dataset(category: str) -> EnergyStarDataset:
    try:
        return ENERGY_STAR_DATASETS[category]
    except KeyError as exc:
        raise ValueError(f"No ENERGY STAR dataset configured for category: {category}") from exc


def validate_records(records, dataset: EnergyStarDataset) -> list[dict]:
    records = list(records)
    if not records:
        raise ValueError("ENERGY STAR API returned no records; existing snapshot was not changed")
    required_fields = {dataset.record_id_field, dataset.brand_field, dataset.model_field}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("ENERGY STAR API returned a non-object record")
        missing = required_fields - record.keys()
        if missing:
            raise ValueError(f"ENERGY STAR record is missing required fields: {sorted(missing)}")
        if not record[dataset.record_id_field] or not record[dataset.model_field]:
            raise ValueError("ENERGY STAR record has an empty ID or model number")
    ids = [str(record[dataset.record_id_field]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("ENERGY STAR API returned duplicate record IDs")
    return records


def download_records(category: str, *, page_size: int = PAGE_SIZE, opener=None) -> list[dict]:
    dataset = _dataset(category)
    opener = opener or urlopen
    records = []
    offset = 0
    while True:
        url = f"{dataset.api_url}?$limit={page_size}&$offset={offset}"
        request = Request(url, headers={"User-Agent": "cashrae-label-ocr/1.0"})
        with opener(request, timeout=60) as response:
            page = json.load(response)
        if not isinstance(page, list):
            raise ValueError("ENERGY STAR API returned a non-list page")
        if not page:
            break
        records.extend(page)
        if len(page) < page_size:
            break
        offset += len(page)
    return validate_records(records, dataset)


def replace_snapshot(records, category: str, db_path=DB_PATH) -> str:
    dataset = _dataset(category)
    records = validate_records(records, dataset)
    refreshed_at = datetime.now(timezone.utc).date().isoformat()
    table = dataset.table
    with sqlite3.connect(db_path) as conn:
        conn.execute("BEGIN")
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"""CREATE TABLE {table} (
            record_id TEXT PRIMARY KEY,
            source_url TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            brand_name TEXT,
            model_number TEXT,
            additional_model_information TEXT,
            upc TEXT,
            special_type TEXT,
            raw_fields_json TEXT NOT NULL
        )""")
        conn.executemany(
            f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(str(record[dataset.record_id_field]), dataset.dataset_url, refreshed_at,
              record.get(dataset.brand_field), record.get(dataset.model_field),
              record.get("additional_model_information"), record.get("upc"),
              record.get("special_type"),
              json.dumps(record, ensure_ascii=False, sort_keys=True)) for record in records],
        )
        conn.execute("""CREATE TABLE IF NOT EXISTS energy_star_refresh_metadata (
            dataset_id TEXT PRIMARY KEY, source_url TEXT NOT NULL,
            refresh_date TEXT NOT NULL, row_count INTEGER NOT NULL
        )""")
        conn.execute("""INSERT OR REPLACE INTO energy_star_refresh_metadata
            VALUES (?, ?, ?, ?)""",
                     (dataset.dataset_id, dataset.dataset_url, refreshed_at, len(records)))
    return refreshed_at


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Refresh an ENERGY STAR product snapshot")
    parser.add_argument("category", choices=sorted(ENERGY_STAR_DATASETS))
    parser.add_argument("--db", default=str(DB_PATH), help="Path to appliances.db")
    args = parser.parse_args(argv)
    records = download_records(args.category)
    refreshed_at = replace_snapshot(records, args.category, args.db)
    print(f"Imported {len(records)} ENERGY STAR {args.category} records (snapshot {refreshed_at}).")


if __name__ == "__main__":
    main()
