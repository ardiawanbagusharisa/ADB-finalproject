import os
import time
import duckdb
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

# Configuration
DB_FILE = "sumobot.duckdb"   # DuckDB database file produced by setup_database_duckdb.py
MODEL_NAME = "llama3"     # via Ollama


def _format_schema(con: duckdb.DuckDBPyConnection) -> str:
    """
    Build a compact schema string for the LLM prompt (DuckDB).
    """
    try:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    except Exception:
        # Fallback via information_schema (rarely needed)
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
    """
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(
            f"Database {DB_FILE} not found. Please run 'python setup_database_duckdb.py' first."
        )

    con = duckdb.connect(DB_FILE, read_only=False)

    print(f"Connecting to local LLM via Ollama (model: {MODEL_NAME})...")
    llm = OllamaLLM(model=MODEL_NAME, temperature=0)

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
    You can loosen this if you want to support CREATE VIEW, etc.
    """
    q = (sql_query or "").strip().lower()
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
    sql_prompt = f"""You are an expert SQL data analyst.
Given the following database schema, write a DuckDB SQL query to answer the user's question.
Return ONLY the SQL query. Do not include markdown formatting like ```sql.

Schema:
{schema}

Question: {question}
SQL Query:"""

    print("Thinking (Generating SQL)...")
    t1_start = time.time()
    sql_response = llm.invoke(sql_prompt)
    t1_end = time.time()

    sql_query = _clean_sql(sql_response)

    print(f">> Generated SQL: {sql_query}")
    print(f"   (SQL Gen Time: {(t1_end - t1_start) * 1000:.2f} ms)")

    if not sql_query:
        return "Error: LLM returned an empty SQL query."

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
    answer_prompt = f"""You are a helpful data assistant.
        Based on the user's question, the SQL query used, and the raw result, write a natural language answer.
        Do not repeat the SQL query. Just give the answer in a clear sentence.

        Question: {question}
        SQL Query: {sql_query}
        Raw Result: {raw_result}

        Answer (in a natural, conversational sentence):"""

    print("Formulating answer...")
    t3_start = time.time()
    final_answer = llm.invoke(answer_prompt)
    t3_end = time.time()
    print(f"   (Answer Gen Time: {(t3_end - t3_start) * 1000:.2f} ms)")

    t_end_total = time.time()
    print(f"   [Total Time: {(t_end_total - t_start_total) * 1000:.2f} ms]")

    return final_answer


def main():
    print("==========================================")
    print("   Sumobot Natural Query Interface (DuckDB)")
    print("==========================================")
    print(f"Using Database: {DB_FILE}")
    print(f"Using Model:    {MODEL_NAME} (via Ollama)")
    print("------------------------------------------")

    try:
        con, llm = get_engine()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print("\nSystem ready. Type 'exit' to quit.")
    print("Example: 'Who won the most games?'")

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
