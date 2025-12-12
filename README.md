# DB Final Project - Natural Query for Game Analytic 

Files:  
1. requirements.txt: Lists all necessary Python libraries (pandas, langchain, ollama, etc.).
2. setup_database.py: A script to read GameRecord_Short.csv and create a structured SQLite database (sumobot.db) with indices for performance.
3. natural_query.py: The main interface script. It connects to a local LLM (via Ollama) and the SQLite database to answer natural language questions.
4. README.md: Comprehensive instructions on how to install dependencies, set up Ollama, prepare the database, and run the project. 
5. .github/copilot-instructions.md: You can use this file as the instructor for your vibe coding. It's not related to our project, but it can help you to vibe code. 

How to run:  
1. Create new py venv and then activate it.
2. Install Dependencies: Run `pip install -r requirements.txt`.
3. Install & Run Ollama: Download Ollama, run `ollama serve` (or just open the desktop app), and pull (choose) a model like `ollama pull llama3`.
4. Setup Database: Run `python setup_database.py` (you can change your database setup here).
5. Start Querying: Run `python natural_query.py` (you can test your natural query here).

Notes:  
- You do not need an API key for Ollama, but you do need to install the Ollama application separately from the Python libraries.
- This process requires some packages from langchain (but you dont need to worry as it's should already be handled by the py script. If not, make sure run this `pip install -U langchain langchain-community langchain-core`.

To stop the query process: `ctrl + c`

I suggest you to use vs code and install these extensions: 
- DB Viewer 
- Rainbow CSV 
- GitHub Copilot 

Some screenshots:  
* Ollama on windows  
<img src = "Figures/ss ollama.png" alt ="figure" width="500">

* DB Viewer  
<img src = "Figures/ss dbviewer.png" alt ="figure" width="500">

* Example of the natural query result  
<img src = "Figures/ss nquery.png" alt ="figure" width="500">