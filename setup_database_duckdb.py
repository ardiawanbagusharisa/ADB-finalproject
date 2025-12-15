import os

import duckdb
import pandas as pd

# Config
CSV_FILE = 'GameRecord_Short.csv'
DB_FILE = 'sumobot.duckdb'  # DuckDB database file
TABLE_NAME = 'game_records'


def setup_database():
    """Reads the GameRecord_Short.csv and loads it into a DuckDB database."""

    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please ensure the file is in the current directory.")
        return

    print(f"Reading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)

        # Rename 'Name' column to 'Action' for better clarity and LLM understanding
        if 'Name' in df.columns:
            print("Renaming column 'Name' to 'Action'...")
            df.rename(columns={'Name': 'Action'}, inplace=True)

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Creating DuckDB database {DB_FILE}...")
    try:
        conn = duckdb.connect(DB_FILE)

        print(f"Loading {len(df)} rows into table '{TABLE_NAME}'...")
        # Fast path: register the DataFrame as a temporary view, then create table from it
        conn.register("df_view", df)
        conn.execute(f"CREATE OR REPLACE TABLE {TABLE_NAME} AS SELECT * FROM df_view")
        conn.unregister("df_view")

        # Optional: DuckDB index support depends on version/workload; safe to try and fall back.
        print("Creating indexes (optional)...")
        try:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_game_index ON {TABLE_NAME} (GameIndex)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_game_winner ON {TABLE_NAME} (GameWinner)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_actor ON {TABLE_NAME} (Actor)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_action ON {TABLE_NAME} (Action)")
        except Exception as idx_e:
            print(f"Skipping indexes (not supported or not needed): {idx_e}")

        conn.close()
        print("Database setup complete successfully.")

    except Exception as e:
        print(f"Error setting up database: {e}")


if __name__ == "__main__":
    setup_database()
