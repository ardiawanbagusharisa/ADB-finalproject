import os
import sys
import time
from langchain_community.utilities import SQLDatabase
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

# Configuration
DB_FILE = 'sumobot.db'
MODEL_NAME = "gemma3:4b" #"qwen2.5-coder:7b" #"deepseek-coder:6.7b" #"duckdb-nsql:7b" #"sqlcoder:7b" #"llama3" 
# Removed: #"qwen2.5-coder:3b"  

def get_engine():
    """
    Initializes the Database and LLM.
    """
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Database {DB_FILE} not found. Please run 'python setup_database.py' first.")

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

    # 2. Generate SQL
    sql_prompt = f"""You are an expert SQL data analyst. 
    Given the following database schema, write a SQLite query to answer the user's question.
    Return ONLY the SQL query. Do not include markdown formatting like ```sql.
    
    Schema:
    {schema}
    
    Question: {question}
    SQL Query:"""
    
    print("Thinking (Generating SQL)...")
    t1_start = time.time()
    sql_response = llm.invoke(sql_prompt)
    t1_end = time.time()
    
    # Clean SQL
    sql_query = sql_response.strip()
    # Remove markdown code blocks if present
    if "```" in sql_query:
        # Find the content inside the backticks
        parts = sql_query.split("```")
        if len(parts) >= 2:
            sql_query = parts[1]
            if sql_query.lower().startswith("sql"):
                sql_query = sql_query[3:]
    
    sql_query = sql_query.strip()
        
    print(f">> Generated SQL: {sql_query}")
    print(f"   (SQL Gen Time: {(t1_end - t1_start)*1000:.4f} ms)")
    
    # 3. Execute SQL
    try:
        print("Executing...")
        t2_start = time.time()
        result = db.run(sql_query)
        t2_end = time.time()
        print(f"   (DB Exec Time: {(t2_end - t2_start)*1000:.2f} ms)")
        # print(f">> Raw Result: {result}") # Optional: print raw result for debugging
    except Exception as e:
        return f"Error executing SQL: {e}"
        
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
    print(f"   (Answer Gen Time: {(t3_end - t3_start)*1000:.2f} ms)")
    
    t_end_total = time.time()
    print(f"   [Total Time: {(t_end_total - t_start_total)*1000:.2f} ms]")

    return final_answer

def main():
    print("==========================================")
    print("   Sumobot Natural Query Interface")
    print("==========================================")
    print(f"Using Database: {DB_FILE}")
    print(f"Using Model:    {MODEL_NAME} (via Ollama)")
    print("------------------------------------------")
    
    try:
        db, llm = get_engine()
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
