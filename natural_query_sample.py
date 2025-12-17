import os
import sys
import time
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM

# Configuration
DB_FILE = 'sample_game.sqlite'
MODEL_NAME = "sqlcoder:7b"

def get_engine():
    """
    Initializes the Database and LLM.
    """
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Database {DB_FILE} not found. Please ensure the file exists in the current directory.")

    # Connect to the SQLite database
    db = SQLDatabase.from_uri(f"sqlite:///{DB_FILE}")
    
    print(f"Connecting to local LLM via Ollama (model: {MODEL_NAME})...")
    try:
        # Use 127.0.0.1 to avoid localhost resolution issues on Windows
        llm = OllamaLLM(model=MODEL_NAME, temperature=0, base_url="http://127.0.0.1:11434")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        raise
        
    return db, llm

def run_query_pipeline(db, llm, question):
    """
    Manually runs the Text -> SQL -> Result -> Text pipeline for better control.
    """
    # 1. Get Schema
    schema = db.get_table_info()
    
    t_start_total = time.time()

    # 2. Generate SQL with enhanced context about the schema
    # Use simpler format for SQL-specialized models like sqlcoder
    if "sqlcoder" in MODEL_NAME.lower() or "nsql" in MODEL_NAME.lower():
        sql_prompt = f"""### Task
Generate a SQL query to answer the following question: {question}

### Database Schema
{schema}

### SQL Query
SELECT"""
    else:
        sql_prompt = f"""You are an expert SQL data analyst. 
Given the following database schema, write a SQLite query to answer the user's question.
Return ONLY the SQL query. Do not include markdown formatting like ```sql.

Schema:
{schema}

Database Context:
- bots: Contains bot information (bot_id, name, language, author, created_at)
- matches: Contains match records between two bots (left_bot_id, right_bot_id, winner_bot_id, duration_s)
- rounds: Contains round-level data for each match (each match can have multiple rounds)
- events: Contains detailed game events (actions, positions, states) for each round

Question: {question}
SQL Query:"""
    
    print("Thinking (Generating SQL)...")
    t1_start = time.time()
    try:
        sql_response = llm.invoke(sql_prompt)
    except Exception as e:
        return f"Error communicating with LLM: {e}\nEnsure Ollama is running and the model '{MODEL_NAME}' is available."
    t1_end = time.time()
    
    # Clean SQL
    sql_query = (sql_response or "").strip()
    
    # For SQL-specialized models, prepend SELECT if it's missing
    if "sqlcoder" in MODEL_NAME.lower() or "nsql" in MODEL_NAME.lower():
        if sql_query and not sql_query.upper().startswith("SELECT"):
            sql_query = "SELECT " + sql_query
    
    # Remove markdown code blocks if present
    if "```" in sql_query:
        # Find the content inside the backticks
        parts = sql_query.split("```")
        if len(parts) >= 2:
            sql_query = parts[1]
            if sql_query.lower().startswith("sql"):
                sql_query = sql_query[3:]
    
    sql_query = sql_query.strip()
    
    # Debug: Show what the LLM actually returned
    if not sql_query:
        print(f">> WARNING: LLM returned empty/invalid response")
        print(f">> Raw LLM Response: {repr(sql_response)}")
        return f"Error: LLM returned an empty SQL query. Raw response: {repr(sql_response)[:200]}"
        
    print(f">> Generated SQL: {sql_query}")
    print(f"   (SQL Gen Time: {(t1_end - t1_start)*1000:.4f} ms)")
    
    # 3. Execute SQL
    try:
        print("Executing...")
        t2_start = time.time()
        result = db.run(sql_query)
        t2_end = time.time()
        print(f"   (DB Exec Time: {(t2_end - t2_start)*1000:.4f} ms)")
    except Exception as e:
        return f"Error executing SQL: {e}\nGenerated query was: {sql_query}"
        
    # 4. Generate Natural Answer
    answer_prompt = f"""You are a helpful data assistant.
        Based on the user's question, the SQL query used, and the raw result, write a natural language answer.
        Do not repeat the SQL query. Just give the answer in a clear sentence.
        
        Question: {question}
        SQL Query: {sql_query}
        Raw Result: {result}
        
        Answer (in a natural, conversational sentence):"""
    
    print("Formulating answer...")
    t3_start = time.time()
    final_answer = llm.invoke(answer_prompt)
    t3_end = time.time()
    print(f"   (Answer Gen Time: {(t3_end - t3_start)*1000:.4f} ms)")
    
    t_end_total = time.time()
    print(f"   [Total Time: {(t_end_total - t_start_total)*1000:.4f} ms]")

    return final_answer

def main():
    print("==========================================")
    print("   Sumobot Sample Game Query Interface")
    print("==========================================")
    print(f"Using Database: {DB_FILE}")
    print(f"Using Model:    {MODEL_NAME} (via Ollama)")
    print("------------------------------------------")
    
    try:
        db, llm = get_engine()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Dynamically report table row counts
    table_counts = {}
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cur = conn.cursor()
            for table in ["bots", "matches", "rounds", "events"]:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    table_counts[table] = cur.fetchone()[0]
                except Exception:
                    table_counts[table] = "?"
    except Exception:
        table_counts = {t: "?" for t in ["bots", "matches", "rounds", "events"]}

    print("\nDatabase contains:")
    print(f"  - {table_counts.get('bots', '?')} bots (with names, languages, authors)")
    print(f"  - {table_counts.get('matches', '?')} matches (left vs right bot battles)")
    print(f"  - {table_counts.get('rounds', '?')} rounds (multiple rounds per match)")
    print(f"  - {table_counts.get('events', '?')} events (detailed game actions/positions)")
    print("------------------------------------------")

    print("\nSystem ready. Type 'exit' to quit.")
    print("\nExample questions:")
    print("  - Which bot won the most matches?")
    print("  - What is the average match duration?")
    print("  - Who is the author of Bot_01?")
    print("  - How many matches did Bot_05 participate in?")
    print("  - Show me the top 5 bots by win rate")
    
    while True:
        try:
            question = input("\nAsk a question: ").strip()
            if not question:
                continue
            if question.lower() in ['exit', 'quit', 'q']:
                break
            
            response = run_query_pipeline(db, llm, question)
            
            print(f"\n>> Answer: {response}")
            
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            break
        except Exception as e:
            print(f"\nError processing query: {e}")

if __name__ == "__main__":
    main()
