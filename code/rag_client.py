# rag_client.py — HTTP client for the shared RAG server (port 5080)
# Drop this import into marin.py, bayazid.py, and arena.py instead of
# loading BookRAG / FAISS directly.
#
# Usage:
#   from rag_client import get_rag_context, search_rag
#
#   context = await get_rag_context("PID controller tuning")
#   results = await search_rag("ESP32 pinout", k=10)

import asyncio
import httpx

RAG_SERVER = "http://127.0.0.1:5080"
_TIMEOUT   = 8.0   # seconds — raise this if your machine is slow


async def get_rag_context(query: str, k: int = 10) -> str:
    """
    Returns a formatted context string ready to inject into a system prompt.
    Returns empty string on any error so the chat still works without RAG.
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                f"{RAG_SERVER}/context",
                json={"query": query, "k": k}
            )
            r.raise_for_status()
            return r.json().get("context", "")
    except Exception as e:
        print(f"⚠️ RAG server error: {e}")
        return ""


async def search_rag(query: str, k: int = 10) -> list:
    """
    Returns raw list of {content, source, page} dicts.
    Returns empty list on any error.
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                f"{RAG_SERVER}/search",
                json={"query": query, "k": k}
            )
            r.raise_for_status()
            return r.json().get("results", [])
    except Exception as e:
        print(f"⚠️ RAG server error: {e}")
        return []


def get_rag_context_sync(query: str, k: int = 10) -> str:
    """
    Synchronous wrapper — use this anywhere you can't await
    (e.g. inside a sync generator like marin.py's response()).
    """
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.post(
                f"{RAG_SERVER}/context",
                json={"query": query, "k": k}
            )
            r.raise_for_status()
            return r.json().get("context", "")
    except Exception as e:
        print(f"⚠️ RAG server error: {e}")
        return ""