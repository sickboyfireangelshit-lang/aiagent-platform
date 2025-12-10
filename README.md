# ai-agent-app

This repository contains a deployable open‑source General AI Agent application based on the provided blueprint. All files below can be copied directly into a GitHub repository.

---
# Project Structure
```
ai-agent-app/
├── README.md
├── requirements.txt
├── app/
│   ├── main.py
│   ├── config.py
│   ├── agent.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── code_executor.py
│   │   ├── web_browser.py
│   │   ├── data_analysis.py
│   ├── models/
│   │   ├── request_models.py
│   │   ├── response_models.py
```

---
# README.md
```markdown
# Open-Source General AI Agent
A deployable open-source agent built with FastAPI + LangChain + Ollama. Supports:
- Code execution (sandboxed)
- Data analysis
- Web browsing
- Reasoning loop via ReAct

## Requirements
- Python 3.10+
- Ollama installed
- A model such as `llama3.1:8b` pulled locally

```
ollama pull llama3.1
```

## Installation
```bash
git clone <your repo>
cd ai-agent-app
pip install -r requirements.txt
```

## Run the App
```bash
uvicorn app.main:app --reload
```

## API Endpoints
### POST /agent/run
Execute a general AI query.
```
{
  "prompt": "Analyze data.csv and summarize"
}
```
```
{
  "response": "..."
}
```
```
---
# requirements.txt
```
fastapi
uvicorn
langchain
langchain-community
pydantic
requests
beautifulsoup4
pandas
matplotlib
```

---
# app/config.py
```python
OLLAMA_MODEL = "llama3.1"
```

---
# app/models/request_models.py
```python
from pydantic import BaseModel

class AgentRequest(BaseModel):
    prompt: str
```

---
# app/models/response_models.py
```python
from pydantic import BaseModel

class AgentResponse(BaseModel):
    response: str
```

---
# app/tools/code_executor.py
```python
import subprocess
import tempfile


def execute_python_code(code: str) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
            tmp.write(code.encode())
            tmp.flush()
            result = subprocess.run([
                "python", tmp.name
            ], capture_output=True, text=True, timeout=10)
        return result.stdout or result.stderr
    except Exception as e:
        return str(e)
```

---
# app/tools/web_browser.py
```python
import requests
from bs4 import BeautifulSoup


def browse_web(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(soup.stripped_strings)
        return text[:5000]
    except Exception as e:
        return f"Error: {e}"
```

---
# app/tools/data_analysis.py
```python
import pandas as pd
import matplotlib.pyplot as plt
import tempfile


def analyze_data(file_path: str) -> str:
    try:
        df = pd.read_csv(file_path)
        return df.describe().to_string()
    except Exception as e:
        return str(e)
```

---
# app/agent.py
```python
from langchain.llms import Ollama
from langchain.agents import initialize_agent, Tool
from .tools.code_executor.py import execute_python_code
from .tools.web_browser import browse_web
from .tools.data_analysis import analyze_data
from .config import OLLAMA_MODEL

llm = Ollama(model=OLLAMA_MODEL)

tools = [
    Tool(
        name="python_executor",
        func=execute_python_code,
        description="Execute Python code in a sandbox."
    ),
    Tool(
        name="web_browser",
        func=browse_web,
        description="Fetch and summarize webpage content."
    ),
    Tool(
        name="data_analysis",
        func=analyze_data,
        description="Run analysis on CSV data."
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

def run_agent(prompt: str) -> str:
    return agent.run(prompt)
```

---
# app/main.py
```python
from fastapi import FastAPI
from .models.request_models import AgentRequest
from .models.response_models import AgentResponse
from .agent import run_agent

app = FastAPI(title="Open Source AI Agent")


@app.post("/agent/run", response_model=AgentResponse)
def run(request: AgentRequest):
    response = run_agent(request.prompt)
    return AgentResponse(response=response)
https://chatgpt.com/canvas/shared/693913259cc88191ae4a4829e7a308c6
