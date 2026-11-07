"""Verification script to test and inspect the imported SQLite database."""

import sqlite3
import pandas as pd

DB_PATH = "database/appliances.db"

def main():
    conn = sqlite3.connect(DB_PATH)

    print("=" * 60)
    print("Query 1: List all tables")
    print("=" * 60)
    q1 = "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name;"
    tables_df = pd.read_sql(q1, conn)
    print(tables_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Query 2: Inspect appliance records (clothes_dryers LIMIT 10)")
    print("=" * 60)
    q2 = "SELECT * FROM clothes_dryers LIMIT 10;"
    dryers_df = pd.read_sql(q2, conn)
    print(f"Columns: {list(dryers_df.columns)}")
    print(dryers_df.head(3).to_string())

    print("\n" + "=" * 60)
    print("Query 3: Count appliance records (clothes_dryers)")
    print("=" * 60)
    q3 = "SELECT COUNT(*) FROM clothes_dryers;"
    print(pd.read_sql(q3, conn).to_string(index=False))

    print("\n" + "=" * 60)
    print("Query 4: Inspect the field dictionary (clothes_dryers)")
    print("=" * 60)
    q4 = "SELECT * FROM field_dictionary WHERE appliance_category = 'clothes_dryers';"
    dict_df = pd.read_sql(q4, conn)
    print(dict_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Query 5: Inspect dataset metadata (datasets)")
    print("=" * 60)
    q5 = "SELECT * FROM datasets;"
    datasets_df = pd.read_sql(q5, conn)
    print(datasets_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Row counts for all appliance tables:")
    print("=" * 60)
    for name in tables_df["name"]:
        cnt = pd.read_sql(f'SELECT COUNT(*) as count FROM "{name}";', conn).iloc[0]["count"]
        print(f"  {name:25}: {cnt:>6} records")

    conn.close()

if __name__ == "__main__":
    main()
