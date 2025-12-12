import pandas as pd
import sqlite3
import os

# Config
CSV_FILE = 'GameRecord_Short.csv'
DB_FILE = 'sumobot.db'
TABLE_NAME = 'game_records'

def setup_database():
    """
    Reads the GameRecord_Short.csv and loads it into a SQLite database.
    """
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
    
    # Data Type Optimization (Optional but good for large files)
    # Convert timestamps if needed, but keeping as string is often safer for simple SQL queries unless date math is needed.
    
    print(f"Creating database {DB_FILE}...")
    try:
        conn = sqlite3.connect(DB_FILE)
        
        print(f"Loading {len(df)} rows into table '{TABLE_NAME}'...")
        # if_exists='replace' will drop the table if it exists and recreate it
        df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        
        # Create indices for common lookup columns to improve query performance
        print("Creating indices...")
        cursor = conn.cursor()
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_game_index ON {TABLE_NAME} (GameIndex)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_game_winner ON {TABLE_NAME} (GameWinner)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_actor ON {TABLE_NAME} (Actor)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_action ON {TABLE_NAME} (Action)")
        
        conn.commit()
        conn.close()
        print("Database setup complete successfully.")
        
    except Exception as e:
        print(f"Error setting up database: {e}")

if __name__ == "__main__":
    setup_database()
