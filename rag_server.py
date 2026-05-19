# rag_server.py — Shared FAISS RAG server (port 5080)
# Loads BookRAG directly — does NOT import from bayazid.py
# (bayazid.rag is _RemoteRAG which would call this server → infinite loop)

import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

# Import BookRAG class directly, not the rag instance
from bayazid import BookRAG

# Load FAISS once — all clients share this single instance
_rag = BookRAG()

app = FastAPI(title="RAG Server")


class SearchRequest(BaseModel):
    query: str
    k: int = 10


@app.post("/search")
async def search(req: SearchRequest):
    k       = min(req.k, 20)
    results = await asyncio.to_thread(_rag.search, req.query, k)
    return {"results": results, "count": len(results)}


@app.post("/context")
async def context(req: SearchRequest):
    k   = min(req.k, 20)
    ctx = await asyncio.to_thread(_rag.get_context_for_teaching, req.query, k)
    return {"context": ctx}


@app.get("/report")
async def report():
    return {
        "indexed": _rag.manifest["indexed"],
        "failed":  _rag.manifest["failed"],
        "total":   len(_rag.manifest["indexed"]),
    }


@app.get("/health")
async def health():
    return {
        "status":  "operational",
        "port":    5080,
        "indexed": len(_rag.manifest["indexed"]),
        "ready":   _rag.vectorstore is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5080, reload=False)