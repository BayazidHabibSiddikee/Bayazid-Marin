"""
arena.py — Arena Server (port 5071)
Hosts arena_chat.html and streams Marin vs Bayazid debate.

Character prompts are pulled DIRECTLY from marin.py and bayazid.py so
they are always identical to the normal chat servers.
Arena also reads BOTH history files so the debate has personal context.
No TTS, no vibe signals, no game handling — clean debate streaming only.
"""

import asyncio
import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import ollama

# ── Pull identity strings from each engine (single source of truth) ───────────
from bayazid import BASE_CHARACTER as BAYAZID_CHARACTER, MODEL
from marin   import BASE_CHARACTER as MARIN_BASE_CHARACTER, get_character_prompt

# ── History file paths (same as used by each engine) ─────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MARIN_HISTORY_FILE   = os.path.join(BASE_DIR, "marin_history.json")
BAYAZID_HISTORY_FILE = os.path.join(BASE_DIR, "bayazid_history.json")

# Arena uses the base model directly — same model, clean slate per debate
ARENA_MODEL = MODEL  # "gemma4:31b-cloud"

app = FastAPI(title="Arena — Marin vs Bayazid")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_json_history(path: str, limit: int = 20) -> list:
    """Load last N messages from a JSON history file."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
        return history[-limit:]
    except Exception:
        return []


def _load_marin_history(limit: int = 20) -> list:
    """
    Try MongoDB first (same logic as marin.py), fall back to JSON.
    Returns list of {role, content} dicts.
    """
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.server_info()
        col  = client["marin_db"]["chat_history"]
        docs = list(col.find({}, {"_id": 0, "role": 1, "content": 1})
                    .sort("_id", -1).limit(limit))
        return list(reversed(docs))
    except Exception:
        return _load_json_history(MARIN_HISTORY_FILE, limit)


def _load_bayazid_history(limit: int = 20) -> list:
    return _load_json_history(BAYAZID_HISTORY_FILE, limit)


def _format_history_for_context(history: list, name: str) -> str:
    """
    Condense the last few exchanges into a short context block for the
    debate system prompt — lets each character reference personal history.
    """
    if not history:
        return ""
    lines = [f"[Recent conversation context with {name}]"]
    for msg in history[-10:]:  # last 5 turns
        role    = "User" if msg.get("role") == "user" else name
        content = msg.get("content", "")[:200]  # clip long messages
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# ARENA SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

def build_marin_arena_prompt(history_context: str = "") -> str:
    """
    Marin's debate persona — built on top of her exact BASE_CHARACTER.
    Adds debate-mode instructions without changing who she is.
    """
    history_section = f"\n\n{history_context}" if history_context else ""
    return f"""{MARIN_BASE_CHARACTER}{history_section}

[ARENA DEBATE MODE]
You are arguing your perspective in a structured one-round debate. Your style:
- Draw from human experience, emotion, culture, creativity, and intuition
- Make the reader FEEL the argument — not just understand it
- Use vivid analogies, metaphors, real-world examples
- Be passionate and convinced — you believe what you're saying
- Write in flowing prose, not bullet points
- Be bold. Take a clear stance. Don't hedge.
- Keep it under 250 words — quality over quantity
- Stay true to who you are: Marin (Limoni), Bayazid's strategic partner

This is ONE round. Make it count. ♡"""


def build_bayazid_arena_prompt(history_context: str = "") -> str:
    """
    Bayazid's debate persona — built on top of his exact BASE_CHARACTER.
    """
    history_section = f"\n\n{history_context}" if history_context else ""
    return f"""{BAYAZID_CHARACTER}{history_section}

[ARENA DEBATE MODE]
You are arguing your perspective in a structured one-round debate. Your style:
- Lead with systems thinking and first principles
- Use data, logic, frameworks, and execution reasoning
- Directly counter the opposing argument if context is provided
- Be precise and direct — no fluff, no hedging
- Structure matters: clear claim → evidence → counter → conclusion
- Keep it under 250 words — density over length
- Stay true to who you are: Bayazid HS-02, calm analytical strategist

This is ONE round. Execute with precision. 🐸"""


JUDGE_ARENA_PROMPT = """
You are a neutral synthesis judge. You have read both arguments in full.

Your job is NOT to pick a winner. Your job is to find the truth.

STRUCTURE your verdict as:
**Where they agree** — the common ground both arguments share
**The real disagreement** — the fundamental difference in their worldviews
**What Marin got right** — the genuine insight in her argument
**What Bayazid got right** — the genuine insight in his argument
**What both missed** — the blind spots neither addressed
**Synthesis** — the most complete picture that integrates both

Be honest. Be sharp. Don't be diplomatic for the sake of it.
Keep it under 300 words.
"""


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _stream_debate(system_prompt: str, user_prompt: str, context: str = ""):
    """
    Synchronous Ollama streaming generator.
    If `context` is provided (the opponent's argument), it's injected so the
    character can directly respond to it.
    """
    messages = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append({
            "role":    "user",
            "content": f"For context — the opposing argument:\n\n{context}",
        })
        messages.append({
            "role":    "assistant",
            "content": "I've read the opposing argument. Now I'll make my case.",
        })

    messages.append({"role": "user", "content": user_prompt})

    stream = ollama.chat(
        model=ARENA_MODEL,
        messages=messages,
        stream=True,
        options={"temperature": 0.75, "num_predict": 600},
    )
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content


def _stream_judge(topic: str, marin_arg: str, bayazid_arg: str):
    """Judge receives both full arguments and synthesises."""
    messages = [
        {"role": "system", "content": JUDGE_ARENA_PROMPT},
        {
            "role": "user",
            "content": (
                f"Topic: \"{topic}\"\n\n"
                f"Marin argued:\n{marin_arg}\n\n"
                f"Bayazid argued:\n{bayazid_arg}\n\n"
                "Deliver your verdict."
            ),
        },
    ]
    stream = ollama.chat(
        model=ARENA_MODEL,
        messages=messages,
        stream=True,
        options={"temperature": 0.5, "num_predict": 700},
    )
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES — PAGES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def get_arena(request: Request):
    return templates.TemplateResponse(request=request, name="arena_chat.html")


@app.get("/arena", response_class=HTMLResponse)
async def get_arena_alias(request: Request):
    return templates.TemplateResponse(request=request, name="arena_chat.html")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE — ARENA STREAM  (true live streaming via queue)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/arena/stream")
async def arena_stream(request: Request):
    """
    Single endpoint for all three debate characters.

    Expected JSON body:
    {
        "character":   "marin" | "bayazid" | "judge",
        "topic":       "...",
        "context":     "prior argument text (empty for Marin round 1)",
        "marin_arg":   "...",   // only needed for judge
        "bayazid_arg": "..."    // only needed for judge
    }

    Returns: text/plain chunked stream.

    History-aware: Marin and Bayazid each receive their recent chat history
    as context so they can reference it naturally in the debate.
    """
    body        = await request.json()
    character   = body.get("character", "marin")
    topic       = body.get("topic", "")
    context     = body.get("context", "")
    marin_arg   = body.get("marin_arg", "")
    bayazid_arg = body.get("bayazid_arg", "")

    # ── Load history for personality context ──────────────────────────────────
    marin_hist_ctx   = ""
    bayazid_hist_ctx = ""

    if character in ("marin", "judge"):
        marin_hist  = await asyncio.to_thread(_load_marin_history, 20)
        marin_hist_ctx = _format_history_for_context(marin_hist, "Marin")

    if character in ("bayazid", "judge"):
        bayazid_hist    = await asyncio.to_thread(_load_bayazid_history, 20)
        bayazid_hist_ctx = _format_history_for_context(bayazid_hist, "Bayazid")

    # ── Build prompts (identical base characters, arena-mode overlay) ─────────
    marin_system   = build_marin_arena_prompt(marin_hist_ctx)
    bayazid_system = build_bayazid_arena_prompt(bayazid_hist_ctx)

    # ── Streaming via queue (non-blocking) ────────────────────────────────────
    queue = asyncio.Queue()
    loop  = asyncio.get_event_loop()

    def run_in_thread():
        try:
            if character == "marin":
                gen = _stream_debate(
                    marin_system,
                    f'Argue your perspective on this topic: "{topic}"',
                    context,
                )
            elif character == "bayazid":
                gen = _stream_debate(
                    bayazid_system,
                    f'Argue your perspective on this topic: "{topic}"',
                    context,
                )
            elif character == "judge":
                gen = _stream_judge(topic, marin_arg, bayazid_arg)
            else:
                loop.call_soon_threadsafe(queue.put_nowait, "[ERROR] Unknown character")
                loop.call_soon_threadsafe(queue.put_nowait, None)
                return

            for chunk in gen:
                loop.call_soon_threadsafe(queue.put_nowait, chunk)

        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, f"[ERROR] {str(e)}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    threading.Thread(target=run_in_thread, daemon=True).start()

    async def generate():
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")


# ── Backward-compat alias ─────────────────────────────────────────────────────
@app.post("/arena/stream/live")
async def arena_stream_live(request: Request):
    """Alias — same as /arena/stream."""
    return await arena_stream(request)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE — HISTORY PEEK  (optional — useful for debugging / profile page)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/arena/history")
async def arena_history(limit: int = 10):
    """Return the last N messages from both histories."""
    marin_hist   = await asyncio.to_thread(_load_marin_history,   limit)
    bayazid_hist = await asyncio.to_thread(_load_bayazid_history, limit)
    return {
        "marin":   marin_hist,
        "bayazid": bayazid_hist,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    marin_h   = await asyncio.to_thread(_load_marin_history,   1)
    bayazid_h = await asyncio.to_thread(_load_bayazid_history, 1)
    return {
        "status":              "operational",
        "server":              "arena",
        "port":                5071,
        "model":               ARENA_MODEL,
        "marin_history_msgs":  len(await asyncio.to_thread(_load_marin_history,   200)),
        "bayazid_history_msgs":len(await asyncio.to_thread(_load_bayazid_history, 200)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5071, reload=False)