"""Refresh the ENERGY STAR certified residential clothes washer snapshot."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from urllib.request import Request, urlopen


DATASET_URL = "https://data.energystar.gov/Active-Specifications/ENERGY-STAR-Certified-Residential-Clothes-Washers/bghd-e2wd"
API_URL = "https://data.energystar.gov/resource/bghd-e2wd.json"
DB_PATH = Path("database/appliances.db")
TABLE = "energy_star_clothes_washers"
PAGE_SIZE = 1000
REQUIRED_FIELDS = {"pd_id", "brand_name", "model_number"}


def download_records():
    records = []
    offset = 0
    while True:
        url = f"{API_URL}?$limit={PAGE_SIZE}&$offset={offset}"
        request = Request(url, headers={"User-Agent": "cashrae-label-ocr/1.0"})
        with urlopen(request, timeout=60) as response:
            page = json.load(response)
        if not isinstance(page, list):
            raise ValueError("ENERGY STAR API returned a non-list page")
        if not page:
            break
        for record in page:
            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                raise ValueError(f"ENERGY STAR record is missing required fields: {sorted(missing)}")
            if not record["pd_id"] or not record["model_number"]:
                raise ValueError("ENERGY STAR record has an empty ID or model number")
        records.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += len(page)
    if not records:
        raise ValueError("ENERGY STAR API returned no records; existing snapshot was not changed")
    ids = [str(record["pd_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("ENERGY STAR API returned duplicate record IDs")
    return records


def replace_snapshot(records, db_path=DB_PATH):
    refreshed_at = datetime.now(timezone.utc).date().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute("BEGIN")
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
        conn.execute(f"""CREATE TABLE {TABLE} (
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
            f"INSERT INTO {TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(str(r["pd_id"]), DATASET_URL, refreshed_at, r.get("brand_name"),
              r.get("model_number"), r.get("additional_model_information"),
              r.get("upc"), r.get("special_type"),
              json.dumps(r, ensure_ascii=False, sort_keys=True)) for r in records],
        )
        conn.execute("""CREATE TABLE IF NOT EXISTS energy_star_refresh_metadata (
            dataset_id TEXT PRIMARY KEY, source_url TEXT NOT NULL,
            refresh_date TEXT NOT NULL, row_count INTEGER NOT NULL
        )""")
        conn.execute("""INSERT OR REPLACE INTO energy_star_refresh_metadata
            VALUES (?, ?, ?, ?)""", ("bghd-e2wd", DATASET_URL, refreshed_at, len(records)))
    return refreshed_at


def main():
    records = download_records()
    refreshed_at = replace_snapshot(records)
    print(f"Imported {len(records)} ENERGY STAR washer records (snapshot {refreshed_at}).")


if __name__ == "__main__":
    main()
