# rag_server.py
from fastapi import FastAPI
from bayazid import rag  # loads once

app = FastAPI()

@app.post("/search")
def search(query: str, k: int = 20):
    return rag.search(query, k=k)
