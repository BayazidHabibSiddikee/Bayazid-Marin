# VPA — Virtual Partner Architecture
## Full Project Context & Instruction Document

---

## 1. WHAT THIS IS

A locally-running, multi-AI web system with two distinct AI personalities served from **two separate FastAPI servers** on different ports. Both run on **Ollama** (local LLM inference — no cloud, no API keys, no subscriptions). A third interface called **Arena** lets both AIs debate a topic with a judge synthesizing the conclusion.

The system is built for one user: **Bayazid**, a self-directed engineering student in Bogura, Bangladesh, focused on embedded systems, IoT, ML, computer vision, and robotics.

---

## 2. THE TWO AIs

### 🌸 Marin Kitagawa
**Server:** `app.py` → `http://localhost:5069`
**Classifier:** `marin_fier.py`

**Personality:**
- Warm, expressive, emotionally intelligent, creatively broad
- Responds with flair — literary, human, non-robotic
- Mood-aware: detects user vibe (lovely, angry, sad, excited, playful) and adjusts tone
- Writes in a way that avoids plagiarism detection — natural, varied, broad perspective
- Handles: general conversation, creative writing, emotional support, broad knowledge, vision analysis, structured learning cards (concept/code/lab-report format)
- Has a VTuber-style 2D avatar in VSeeFace with 5 facial expressions (joy, fun, neutral, sorrow, angry) driven via OSC signals (`expression.py`)
- Built-in games: Tic Tac Toe (avatar reacts to board state)

**Design theme:** Cherry blossom — dark crimson night (`#1a0a10`), sakura pink (`#ffb7c5`), deep rose (`#e91e8c`), `Playfair Display` + `Zen Kaku Gothic New` fonts

**Key files:** `marin.py`, `app.py`, `marin_fier.py`, `expression.py`

---

### ⚔ Bayazid HS-02
**Server:** `main.py` → `http://localhost:5070`
**Classifier:** `classifier.py`

**Personality:**
- Calm, analytical, execution-focused, systems thinker
- Direct and honest — no fake flattery, no hedging
- Responds concisely unless depth is requested, then goes deep
- Expertise: embedded systems, IoT, ESP32, Arduino, Python, C/C++, Linux, FastAPI, ML, computer vision, robotics
- Handles: technical chat, structured teaching (3 depths), MCQ quiz generation, code review, error diagnosis, week-by-week study plans, focus session tracking
- Aware of active focus sessions — injects timer context into AI responses

**Design theme:** Sword King — deep black (`#0a0a0f`), royal purple (`#8b5cf6`), electric violet (`#a78bfa`), `Cinzel` + `Rajdhani` + `Share Tech Mono` fonts, scan-line overlay, noise grain texture

**Key files:** `bayazid.py`, `main.py`, `classifier.py`

---

## 3. ARCHITECTURE DECISIONS

### Two Separate Servers (Not One)
The user runs both AIs **simultaneously** — Marin answers something, Bayazid refines it, or vice versa. They are used as two independent tools in the same workflow, not alternatives. Separate servers mean:
- Independent restarts — crashing one doesn't kill the other
- No route conflicts or shared state collisions
- Each can be updated/modified independently
- Ollama queues requests regardless — this is a hardware constraint, not a server constraint

### Shared Knowledge Base
Both AIs share the **same** `doc/` folder and `faiss_db/` vector store. There is one RAG loader (`rag/loader.py`) that builds and maintains the FAISS index. Both `marin.py` and `bayazid.py` import from it:
```python
from rag.loader import rag_db
```
This means: add a PDF to `doc/`, restart either server, both AIs gain the knowledge. No duplication, one source of truth.

### Single Model, Three Characters
The system runs **one Ollama model** (e.g. `llama3.2:3b`). All three characters — Marin, Bayazid, and the Arena Judge — are just different system prompts on the same model. No extra downloads required.

---

## 4. FILE STRUCTURE

```
vpa/
├── app.py                  ← Marin's FastAPI server (port 5069)
├── main.py                 ← Bayazid's FastAPI server (port 5070)
│
├── marin.py                ← Marin AI engine (streaming, memory, vibe, structured cards)
├── bayazid.py              ← Bayazid AI engine (streaming, memory, teach, quiz, review, plan)
│
├── marin_fier.py           ← Marin's intent + vibe classifier (regex, 0ms)
├── classifier.py           ← Bayazid's intent classifier (regex, 0ms)
│
├── expression.py           ← VSeeFace OSC — controls Marin's 2D avatar expressions
├── config.py               ← Shared constants (models, ports, paths, app list)
│
├── rag/
│   └── loader.py           ← Shared FAISS RAG loader (both AIs import this)
│
├── games/
│   └── tiktaktoe.py        ← Marin's Tic Tac Toe game engine
│
├── doc/                    ← Drop PDF textbooks here — shared by both AIs
├── faiss_db/               ← FAISS vector index (auto-built, shared)
│   ├── index.faiss
│   ├── index.pkl
│   └── manifest.json       ← Tracks which PDFs are indexed
│
├── templates/
│   ├── index.html          ← Shared landing page (links to both chats + Arena)
│   ├── marin_chat.html     ← Marin's chat interface (cherry blossom theme)
│   ├── bayazid_chat.html   ← Bayazid's chat interface (sword king theme)
│   ├── arena_chat.html     ← Arena — Marin vs Bayazid debate UI
│   └── profile.html        ← Shared profile — both timers, stats, projects
│
├── static/
│   ├── uploads/            ← Runtime image uploads (gitignored)
│   ├── generated/          ← AI-generated images (gitignored)
│   └── screenshots/        ← README screenshots (committed to git)
│
├── bayazid_history.json    ← Bayazid's persistent conversation memory
├── marin_history.json      ← Marin's persistent conversation memory
│
└── requirements.txt
```

---

## 5. THE PAGES

### `index.html` — Landing / Hub
Links to all three interfaces. Shows live stats (both timers, active sessions). Has the animated purple grid background from Bayazid's theme as base, with cherry blossom orbs on the Marin section. Navigation buttons go to `marin_chat.html`, `bayazid_chat.html`, and `arena_chat.html`.

### `marin_chat.html` — Marin's Chat
Cherry blossom theme. Features: streaming chat, image upload (vision), vibe badge (updates based on user mood), structured learning cards (concept/code/lab-report format rendered as rich cards), markdown + KaTeX math, highlight.js code syntax, copy button on code blocks. Connects to `http://localhost:5069/message`.

### `bayazid_chat.html` — Bayazid's Chat
Sword King theme. Features: streaming chat, sidebar with mode switching (Chat / Teach / Quiz / Code Review / Study Plan), mode tabs, depth toggle (Quick / Standard / Deep), quiz config panel with topic + difficulty + question count, quiz rendered as interactive MCQ buttons with immediate answer feedback and explanations, focus timer strip with live elapsed clock, image upload, markdown + KaTeX math. Connects to `http://localhost:5070/message`.

### `arena_chat.html` — The Arena
Split-screen debate interface. Left column is Marin (cherry theme), right column is Bayazid (sword theme). Bottom panel is the Judge (gold theme, slides up after debate). One topic input at the top. Fight flow: Marin argues → Bayazid argues (receives Marin's full argument as context) → Judge synthesizes. All three stream live. Single round, then judgment. Both columns share the same topic echo at the start. Connects to `/arena/stream` on whichever server hosts it. Status pill shows IDLE → ⚔ FIGHTING → ✓ DONE. Reset button clears everything.

### `profile.html` — Operator Profile
Shows: identity card (Bayazid's focus areas, active projects), live session card (appears when a timer is running), stats grid (sessions today, focus time, total sessions), skill progress bars (embedded systems, IoT, ML, CV, Python, robotics), active projects cards (CNC plotter, ESP32 car, face recognition, surveillance robot), session log.

---

## 6. THE ARENA — HOW IT WORKS

The Arena uses the same single Ollama model with three different system prompts:

**Marin's prompt in Arena:**
> You are Marin Kitagawa arguing your perspective on a topic. Be expressive, emotionally intelligent, draw from human experience, creativity, and broad cultural knowledge. Argue with warmth and conviction. One round, make it count.

**Bayazid's prompt in Arena:**
> You are Bayazid HS-02 arguing your perspective on a topic. Be analytical, systems-focused, data-driven, precise. Counter the opposing argument with logic and execution thinking. One round, no fluff.

**Judge's prompt in Arena:**
> You are a neutral synthesis judge. You have read both arguments in full. Identify where they agree, where they fundamentally differ, what each side missed, and deliver a balanced conclusion that integrates both perspectives. Do not pick a winner — find the truth in both.

**Backend endpoint:** `POST /arena/stream`
```json
{
  "character": "marin" | "bayazid" | "judge",
  "role": "argue" | "conclude",
  "topic": "the topic string",
  "context": "prior argument(s) passed as context"
}
```
Returns `text/plain` stream. Marin streams first. When done, Bayazid gets `Marin's argument` in the context field so it can actually counter her. When both finish, the judge gets both full arguments in context and concludes.

---

## 7. CLASSIFIERS

### `marin_fier.py`
Detects:
- **Intent:** image_gen, play_tiktaktoe, play_connect4, play_wordgame, chat
- **Vibe:** lovely, flirty, angry, sad, excited, playful, neutral

Vibe is injected into Marin's system prompt so she adjusts her emotional tone automatically.

### `classifier.py`
Detects:
- **Intent:** chat, teach, quiz, study_plan, code_review, debug, code_gen, timer, vision, productivity
- **Sub-intent:** (e.g. for teach: quick/standard/deep | for quiz: easy/medium/hard | for timer: start/stop/status/stats)
- **Urgency:** normal, high (for errors/deadlines)

Intent routes the message to the correct handler function in `bayazid.py` before the model runs.

---

## 8. MEMORY SYSTEM

Both AIs have persistent memory saved as JSON files:
- `marin_history.json` — Marin keeps last N messages
- `bayazid_history.json` — Bayazid keeps last 50 messages

Memory is loaded on server start and appended on every exchange. The context window passed to Ollama is always the last 12 messages (configurable). Both can be cleared from the UI (`/memory/clear` endpoint).

---

## 9. FOCUS TIMER (Bayazid only)

Bayazid has a `StudyTimer` class that tracks named focus sessions with start time, end time, duration, and task name. Stats: sessions today, total today in formatted time, all-time session count.

The timer strip is always visible in `bayazid_chat.html`. When a session is active, Bayazid's AI responses are aware of it — the current task and elapsed time are injected into the system prompt so Bayazid can reference what you're working on without being told.

Commands: `/timer start [task]`, `/timer stop`, `/timer status`, `/timer stats`

---

## 10. RAG — BOOK KNOWLEDGE

The shared RAG loader (`rag/loader.py`) works like this:
1. On server start, it scans `doc/` for PDFs
2. Any PDF not already in `faiss_db/manifest.json` gets chunked and embedded
3. New chunks are added to the existing FAISS index (incremental — doesn't re-embed everything)
4. The index is saved to `faiss_db/` and the manifest updated
5. Image-only or corrupt PDFs are skipped and logged
6. The resulting `rag_db` singleton is imported by both `marin.py` and `bayazid.py`

Embedding model: `all-MiniLM-L6-v2` (HuggingFace, runs locally)

---

## 11. LATEX / MATH RENDERING

Both chat interfaces use **KaTeX** for math rendering. The `renderContent()` / `renderMarkdown()` function:
1. Stashes `$$...$$` and `$...$` blocks before markdown processing
2. Runs markdown through `marked.js`
3. Restores and renders math blocks with KaTeX
4. Never corrupts math inside code blocks (those are excluded)

The landing page (`index.html`) also runs `renderMathInElement()` on page load for any static math in the HTML.

---

## 12. EXPRESSION SYSTEM (Marin only)

`expression.py` sends OSC messages to VSeeFace (running on `127.0.0.1:39539`). The 5 supported expressions are: `fun`, `joy`, `neutral`, `sorrow`, `angry`.

Before setting a new expression, all others are reset to `0.0` (prevents mixing). The target expression is set to `1.0`. This is tied to the vibe detected by `marin_fier.py` — the vibe drives both the AI's tone and the avatar's face.

This file runs independently as a standalone script, not imported by the server.

---

## 13. TECH STACK

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| AI Inference | Ollama (local, offline) |
| Primary model | `llama3.2:3b` (or any installed Ollama model) |
| Fast model | `qwen2.5:0.5b` (quiz generation) |
| Vector store | FAISS + `all-MiniLM-L6-v2` |
| Frontend | Vanilla HTML / CSS / JS — no framework |
| Markdown | `marked.js` |
| Math | KaTeX 0.16.9 |
| Code highlighting | `highlight.js` |
| Avatar | VSeeFace (external) + `python-osc` |
| Marin fonts | Playfair Display, Zen Kaku Gothic New |
| Bayazid fonts | Cinzel, Rajdhani, Share Tech Mono |

---

## 14. WHAT IS NOT NEEDED

| File | Status | Reason |
|---|---|---|
| `expression.py` | Optional, standalone | Only needed if VSeeFace avatar is running |
| `games/` folder | Marin-side only | Bayazid has no games |
| `static/generated/` | Marin-side only | Image gen not in Bayazid |
| `tools/` package | Check imports | Only include if `marin.py` or `marin_fier.py` imports from it |
| `EMAILS` in config | Remove | Not relevant to study partner use case |

---

## 15. HOW TO RUN

```bash
# Terminal 1 — Marin
cd vpa
python app.py          # starts on :5069

# Terminal 2 — Bayazid
cd vpa
python main.py         # starts on :5070

# Optional — Marin avatar expressions
python expression.py   # only if VSeeFace is open
```

Then open:
- `http://localhost:5069` — Marin's world
- `http://localhost:5070` — Bayazid's command center
- `http://localhost:5070/arena` — The Arena (hosted on Bayazid's server)

---

*"Execution over illusion. Systems over chaos. Growth over dependency." 🐸*
*"Always ♡" 🌸*