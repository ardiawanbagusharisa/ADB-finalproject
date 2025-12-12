import sqlite3
import pandas as pd

DB_FILE = 'sumobot.db'

def inspect():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get table info
    # cursor.execute("PRAGMA table_info(game_records)")
    # columns = cursor.fetchall()
    # print("Columns:")
    # for col in columns:
    #     print(col)
    
    from langchain_community.utilities import SQLDatabase
    db = SQLDatabase.from_uri(f"sqlite:///{DB_FILE}")
    print("LangChain Table Info:")
    print(db.get_table_info())


if __name__ == "__main__":
    inspect()
