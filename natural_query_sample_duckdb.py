import os
import time
import duckdb
from langchain_ollama import OllamaLLM

# Configuration
DB_FILE = "sample_game.duckdb"
SQLITE_FILE = "sample_game.sqlite"
MODEL_NAME = "duckdb-nsql:7b"  # Can be changed to other models

def _format_schema(con: duckdb.DuckDBPyConnection) -> str:
    """
    Build a compact schema string for the LLM prompt (DuckDB).
    """
    try:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    except Exception:
        # Fallback via information_schema
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]

    if not tables:
        return "(No tables found.)"

    parts = []
    for t in tables:
        try:
            # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
            cols = con.execute(f"PRAGMA table_info('{t}')").fetchall()
        except Exception as e:
            parts.append(f"Table {t}: <error reading columns: {e}>")
            continue

        col_lines = []
        for _, name, coltype, notnull, dflt, pk in cols:
            extras = []
            if pk:
                extras.append("PK")
            if notnull:
                extras.append("NOT NULL")
            if dflt is not None:
                extras.append(f"DEFAULT {dflt}")
            extra_str = (" " + " ".join(extras)) if extras else ""
            col_lines.append(f"  {name} {coltype}{extra_str}")
        parts.append(f"Table {t}(\n" + ",\n".join(col_lines) + "\n)")
    return "\n\n".join(parts)


def get_engine():
    """
    Initializes the DuckDB connection and the local LLM (Ollama).
    If DuckDB file doesn't exist but SQLite does, import it.
    """
    if not os.path.exists(DB_FILE):
        if os.path.exists(SQLITE_FILE):
            print(f"Creating {DB_FILE} from {SQLITE_FILE}...")
            con = duckdb.connect(DB_FILE)
            # Import all tables from SQLite
            con.execute(f"ATTACH '{SQLITE_FILE}' AS sqlite_db (TYPE SQLITE)")
            
            # Get table names from SQLite
            tables = [r[0] for r in con.execute("SELECT name FROM sqlite_db.sqlite_master WHERE type='table'").fetchall()]
            
            for table in tables:
                print(f"  Importing table: {table}")
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM sqlite_db.{table}")
            
            con.execute("DETACH sqlite_db")
            print(f"Import complete!")
        else:
            raise FileNotFoundError(
                f"Neither {DB_FILE} nor {SQLITE_FILE} found. Please ensure sample_game.sqlite exists."
            )
    else:
        con = duckdb.connect(DB_FILE, read_only=False)

    print(f"Connecting to local LLM via Ollama (model: {MODEL_NAME})...")
    llm = OllamaLLM(model=MODEL_NAME, temperature=0, base_url="http://127.0.0.1:11434")

    return con, llm


def _clean_sql(sql_text: str) -> str:
    sql_query = (sql_text or "").strip()
    if "```" in sql_query:
        parts = sql_query.split("```")
        if len(parts) >= 2:
            sql_query = parts[1].strip()
            if sql_query.lower().startswith("sql"):
                sql_query = sql_query[3:].strip()
    return sql_query.strip().rstrip(";")


def _is_safe_readonly(sql_query: str) -> bool:
    """
    Optional guardrail: allow only read-only queries by default.
    """
    import re
    q = (sql_query or "").strip().lower()
    # Normalize whitespace to handle multiline queries
    q = re.sub(r'\s+', ' ', q)
    # Allow WITH ... SELECT ...
    if q.startswith("with "):
        return True
    return q.startswith("select ") or q.startswith("show ") or q.startswith("describe ") or q.startswith("pragma ")


def run_query_pipeline(con: duckdb.DuckDBPyConnection, llm, question: str) -> str:
    """
    Text -> DuckDB SQL -> Result -> Text
    """
    schema = _format_schema(con)

    t_start_total = time.time()

    # 1) Generate DuckDB SQL
    # Use different prompt format for SQL-specialized models
    if "duckdb-nsql" in MODEL_NAME.lower() or "sqlcoder" in MODEL_NAME.lower():
        sql_prompt = f"""### Task
Generate a DuckDB SQL query to answer the following question: {question}

### Database Schema
{schema}

### SQL Query
SELECT"""
    else:
        sql_prompt = f"""You are an expert SQL data analyst.
Given the following database schema, write a DuckDB SQL query to answer the user's question.
Return ONLY the SQL query. Do not include markdown formatting like ```sql.

Schema:
{schema}

Database Context:
- bots: Contains bot information (bot_id, name, language, author, created_at)
- matches: Contains match records between two bots (left_bot_id, right_bot_id, winner_bot_id, duration_s)
  * To count unique bots in matches, use UNION to combine left_bot_id, right_bot_id, and winner_bot_id
- rounds: Contains round-level data for each match (each match can have multiple rounds)
- events: Contains detailed game events (actions, positions, states) for each round

Important SQL Rules:
- Write clean, efficient queries
- Avoid redundant JOINs or self-joins without proper aliases
- Use UNION to combine columns from the same table when counting distinct values across multiple columns
- Prefer subqueries or CTEs for clarity when needed

Question: {question}
SQL Query:"""

    print("Thinking (Generating SQL)...")
    t1_start = time.time()
    try:
        sql_response = llm.invoke(sql_prompt)
    except Exception as e:
        return f"Error communicating with Ollama: {e}\nEnsure Ollama is running."
    t1_end = time.time()

    sql_query = _clean_sql(sql_response)
    
    # For SQL-specialized models, prepend SELECT if it's missing
    if "duckdb-nsql" in MODEL_NAME.lower() or "sqlcoder" in MODEL_NAME.lower():
        if sql_query and not sql_query.upper().startswith("SELECT"):
            sql_query = "SELECT " + sql_query

    print(f">> Generated SQL: {sql_query}")
    print(f"   (SQL Gen Time: {(t1_end - t1_start) * 1000:.2f} ms)")

    if not sql_query:
        print(f">> WARNING: LLM returned empty/invalid response")
        print(f">> Raw LLM Response: {repr(sql_response)}")
        return f"Error: LLM returned an empty SQL query. Raw response: {repr(sql_response)[:200]}"

    # Optional safety: block non-read-only queries
    if not _is_safe_readonly(sql_query):
        return (
            "Blocked a non-read-only query for safety.\n"
            f"Generated query was: {sql_query}\n"
            "If you want to allow writes/DDL, edit _is_safe_readonly()."
        )

    # 2) Execute against DuckDB
    try:
        print("Executing...")
        t2_start = time.time()

        # Prefer a dataframe result for nicer display (requires pandas)
        result_obj = None
        try:
            result_obj = con.execute(sql_query).fetchdf()
        except Exception:
            result_obj = con.execute(sql_query).fetchall()

        t2_end = time.time()
        print(f"   (DB Exec Time: {(t2_end - t2_start) * 1000:.2f} ms)")
    except Exception as e:
        return f"Error executing DuckDB SQL: {e}"

    # Prepare raw result string for LLM answer
    if hasattr(result_obj, "to_string"):
        # pandas DataFrame
        raw_result = result_obj.head(50).to_string(index=False)
    else:
        raw_result = str(result_obj[:50]) if isinstance(result_obj, list) else str(result_obj)

    # 3) Natural language answer
    # SQL-specialized models can't generate conversational text, use a general model
    if "duckdb-nsql" in MODEL_NAME.lower() or "sqlcoder" in MODEL_NAME.lower():
        print("Using general-purpose model for natural language answer...")
        answer_llm = OllamaLLM(model="gemma3:4b", temperature=0, base_url="http://127.0.0.1:11434")
    else:
        answer_llm = llm
    
    answer_prompt = f"""You are a helpful data assistant.
        Based on the user's question, the SQL query used, and the raw result, write a natural language answer.
        Do not repeat the SQL query. Just give the answer in a clear sentence.

        Question: {question}
        SQL Query: {sql_query}
        Raw Result: {raw_result}

        Answer (in a natural, conversational sentence):"""

    print("Formulating answer...")
    t3_start = time.time()
    final_answer = answer_llm.invoke(answer_prompt)
    t3_end = time.time()
    print(f"   (Answer Gen Time: {(t3_end - t3_start) * 1000:.2f} ms)")

    t_end_total = time.time()
    print(f"   [Total Time: {(t_end_total - t_start_total) * 1000:.2f} ms]")

    return final_answer


def main():
    print("==========================================")
    print("   Sumobot Sample Game Query Interface (DuckDB)")
    print("==========================================")
    print(f"Using Database: {DB_FILE}")
    print(f"Using Model:    {MODEL_NAME} (via Ollama)")
    print("------------------------------------------")
    print("\nDatabase contains:")
    print("  - 30 bots (with names, languages, authors)")
    print("  - 50 matches (left vs right bot battles)")
    print("  - 123 rounds (multiple rounds per match)")
    print("  - 1700 events (detailed game actions/positions)")
    print("------------------------------------------")

    try:
        con, llm = get_engine()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

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
            if question.lower() in ["exit", "quit", "q"]:
                break

            response = run_query_pipeline(con, llm, question)
            print(f"\n>> Answer: {response}")

        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            break
        except Exception as e:
            print(f"\nError processing query: {e}")

    try:
        con.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
