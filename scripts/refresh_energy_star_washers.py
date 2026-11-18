"""Backward-compatible washer snapshot refresh command and API."""

from urllib.request import urlopen
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.energy_star_config import ENERGY_STAR_DATASETS
from scripts.refresh_energy_star import (
    DB_PATH,
    PAGE_SIZE,
    REQUIRED_FIELDS,
    download_records as _download_records,
    replace_snapshot as _replace_snapshot,
)


DATASET = ENERGY_STAR_DATASETS["clothes_washers"]
DATASET_URL = DATASET.dataset_url
API_URL = DATASET.api_url
TABLE = DATASET.table


def download_records():
    # Keep these module-level seams so existing tests and callers can patch them.
    return _download_records("clothes_washers", page_size=PAGE_SIZE, opener=urlopen)


def replace_snapshot(records, db_path=DB_PATH):
    return _replace_snapshot(records, "clothes_washers", db_path)


def main():
    records = download_records()
    refreshed_at = replace_snapshot(records)
    print(f"Imported {len(records)} ENERGY STAR washer records (snapshot {refreshed_at}).")


if __name__ == "__main__":
    main()
