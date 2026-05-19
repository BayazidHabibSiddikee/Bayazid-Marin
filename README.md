# VPA — Virtual Partner Architecture

A locally-running, multi-AI web system with three distinct interfaces served from three separate FastAPI servers. All AI inference runs on **Ollama** (local, offline, no cloud/API keys).

Built for Bayazid — a self-directed engineering student at RUET, Rajshahi, Bangladesh, focused on embedded systems, IoT, ML, computer vision, and robotics.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      VPA Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Marin Chat      │  │ Bayazid Chat   │  │ The Arena      │  │
│  │ (Port 5069)     │  │ (Port 5070)    │  │ (Port 5071)    │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│           ▼                    ▼                    ▼           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ app.py          │  │ main.py         │  │ arena.py       │  │
│  │ (FastAPI)       │  │ (FastAPI)       │  │ (FastAPI)      │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│           ▼                    ▼                    ▼           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ marin.py        │  │ bayazid.py      │  │ Same engines   │  │
│  │ AI Engine       │  │ AI Engine       │  │ via imports    │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                ▼                                  │
│                   ┌────────────────────────┐                      │
│                   │    RAG Server          │                      │
│                   │    (Port 5080)         │                      │
│                   └───────────┬────────────┘                      │
│                               ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                   OLLAMA (Local LLM)                      │   │
│  │         gemma4:31b-cloud  |  qwen2.5:0.5b                 │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Three AI Interfaces

### 1. 🌸 Marin Kitagawa — Port 5069

**Server**: `python app.py` → `http://localhost:5069`

| Feature | Description |
|---------|-------------|
| Personality | Warm, expressive, emotionally intelligent, creative |
| Vibe System | Detects user mood (lovely, flirty, angry, sad, excited, playful) and adjusts tone |
| Learning Cards | Structured output: concept/code/lab-report format |
| Vision | Image upload analysis |
| Games | Tic Tac Toe, Connect 4, Word Game |
| YouTube | Transcript fetching and analysis |
| Avatar | VSeeFace integration via OSC (5 expressions) |
| Design | Cherry blossom theme — sakura pink, deep rose |

**Key Files**: `app.py`, `marin.py`, `marin_fier.py`, `expression.py`, `games/`

---

### 2. ⚔️ Bayazid HS-02 — Port 5070

**Server**: `python main.py` → `http://localhost:5070`

| Feature | Description |
|---------|-------------|
| Personality | Calm, analytical, execution-focused, systems thinker |
| Teach Mode | Structured explanations with 3 depth levels (Quick/Standard/Deep) |
| Quiz Engine | AI-generated MCQ with instant feedback |
| Code Review | Structured bug → efficiency → style → improved version |
| Error Diagnosis | Root cause analysis and exact fixes |
| Study Plans | Week-by-week learning plans with projects |
| Focus Timer | Named session tracking with stats |
| Vision | Circuit diagrams, schematics, code screenshots |
| Design | Sword King theme — deep black, royal purple, electric violet |

**Key Files**: `main.py`, `bayazid.py`, `classifier.py`

**Slash Commands**:
- `/timer start [task]` — Begin focus session
- `/timer stop` — End session
- `/timer status` — Check current session
- `/timer stats` — View productivity stats

---

### 3. 🏟️ The Arena — Port 5071

**Server**: `python arena.py` → `http://localhost:5071`

A debate interface where both AIs argue a topic with a neutral judge synthesizing conclusions.

| Flow | Description |
|------|-------------|
| Step 1 | Marin argues first |
| Step 2 | Bayazid argues (receives Marin's full argument as context) |
| Step 3 | Judge synthesizes (receives both arguments) |

**Character Prompts**: Pulled directly from `marin.py` and `bayazid.py` (single source of truth)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Pull Required Models

```bash
# Primary model — for all AI responses
ollama pull gemma4:31b-cloud

# Fast model — for quizzes, games, quick tasks
ollama pull qwen2.5:0.5b
```

### 3. Run Servers

Use the provided shell scripts for easy startup:

```bash
# Make scripts executable (if not already)
chmod +x *.sh

# Option 1: Run all servers at once (recommended)
./run_all.sh

# Option 2: Run individually in separate terminals
./run_marin.sh      # Port 5069 - Marin Chat
./run_bayazid.sh    # Port 5070 - Bayazid Chat
./run_arena.sh      # Port 5071 - Arena Debate
./rag_server.sh    # Port 5080 - RAG Server
```

Or run directly with Python:

```bash
# Terminal 1 — RAG Server (optional, for book knowledge)
python rag_server.py

# Terminal 2 — Marin
python app.py

# Terminal 3 — Bayazid
python main.py

# Terminal 4 — Arena (optional)
python arena.py
```

> **Note**: All `.sh` scripts must be executable. Run `chmod +x *.sh` if needed.

### 4. Access Interfaces

| Interface | URL |
|-----------|-----|
| Marin Chat | http://localhost:5069 |
| Bayazid Chat | http://localhost:5070 |
| Arena | http://localhost:5071 |
| Profile | http://localhost:5070/profile |

---

## Project Structure

```
vpa/
├── app.py                  # Marin FastAPI server (port 5069)
├── main.py                 # Bayazid FastAPI server (port 5070)
├── arena.py                # Arena debate server (port 5071)
│
├── marin.py                # Marin AI engine
├── bayazid.py              # Bayazid AI engine
│
├── marin_fier.py           # Marin's intent + vibe classifier
├── classifier.py          # Bayazid's intent classifier
│
├── expression.py           # VSeeFace OSC avatar control
│
├── rag_server.py          # RAG server for book knowledge (port 5080)
│
├── *.sh                    # Shell scripts for easy startup
│   ├── run_all.sh         # Start all 4 servers
│   ├── run_marin.sh       # Start Marin only
│   ├── run_bayazid.sh     # Start Bayazid only
│   ├── run_arena.sh       # Start Arena only
│   └── rag_server.sh      # Start RAG server only
│
├── games/                  # Game engines
│   ├── tiktaktoe.py
│   ├── connect4_ai.py
│   ├── connect4_2p.py
│   └── wordgame.py
│
├── tools/                  # Utility tools
│   ├── email_tool.py
│   ├── bangla.py
│   ├── translate.py
│   ├── timer.py
│   └── alarm.py
│
├── templates/              # HTML templates
│   ├── index.html         # Landing page
│   ├── marin_chat.html    # Marin chat UI
│   ├── bayazid_chat.html # Bayazid chat UI
│   ├── arena_chat.html    # Arena debate UI
│   └── profile.html       # User profile & stats
│
├── static/                 # Static files
│   └── uploads/           # Uploaded images
│
├── doc/                    # PDF textbooks for RAG
├── faiss_db/              # FAISS vector index
│
├── marin_history.json     # Marin conversation memory
├── bayazid_history.json   # Bayazid conversation memory
├── timer_sessions.json    # Focus timer data
├── vibe_state.json        # Current mood state
│
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## RAG (Book Knowledge)

Drop PDF textbooks into the `doc/` folder. Both AIs can answer questions using your book library.

### How It Works

1. Server start → scans `doc/` for PDFs
2. New PDFs → chunked → embedded → added to FAISS index
3. Query → FAISS search → top-k chunks → injected into prompt
4. Result → AI response with citations

### Index Management

```python
# View index report
from bayazid import rag
print(rag.get_index_report())

# Re-index failed files
rag.reindex_failed()
```

---

## Configuration

### Changing Models

**Marin** (`marin.py` line 41):
```python
MODEL = "llama3.2:3b"  # Change here
```

**Bayazid** (`bayazid.py` line 21):
```python
MODEL = "llama3.2:3b"  # Change here
```

**Arena** (inherits from bayazid.py)

### Available Models

```bash
ollama list                    # List installed
ollama pull mistral            # Pull new model
ollama pull deepseek-coder    # Another option
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + Uvicorn |
| AI Inference | Ollama (local, offline) |
| Primary Model | gemma4:31b-cloud |
| Fast Model | qwen2.5:0.5b |
| Vector Store | FAISS + all-MiniLM-L6-v2 |
| Frontend | Vanilla HTML/CSS/JS |
| Math Rendering | KaTeX |
| Code Highlighting | highlight.js |
| Markdown | marked.js |
| Avatar | VSeeFace + python-osc |

---

## Optional Features

### VSeeFace Avatar

Run `python expression.py` to enable 2D avatar expressions via OSC.

```bash
# Terminal
python expression.py
```

### YouTube Analysis (Marin)

Paste a YouTube URL and Marin will fetch and analyze the transcript.

---

## Shell Scripts

The project includes several shell scripts for easy server management:

| Script | Description | Port |
|--------|-------------|------|
| `run_all.sh` | Start all 4 servers at once | All |
| `run_marin.sh` | Start Marin server only | 5069 |
| `run_bayazid.sh` | Start Bayazid server only | 5070 |
| `run_arena.sh` | Start Arena server only | 5071 |
| `rag_server.sh` | Start RAG server only | 5080 |

### Usage Examples

```bash
# Run all servers
./run_all.sh

# Output:
# → RAG Server (port 5080)
# → Marin (port 5069)
# → Bayazid (port 5070)
# → Arena (port 5071)

# Run individual servers
./run_marin.sh      # In terminal 1
./run_bayazid.sh    # In terminal 2
./run_arena.sh      # In terminal 3
```

### Stopping Servers

```bash
# Stop all Python/uvicorn processes
pkill -f "uvicorn\|rag_server"

# Or stop specific port
kill $(lsof -t -i:5069)
```

---

## Troubleshooting

### Ollama Not Running

```bash
# Check status
ollama list

# Start service
ollama serve
```

### Port Already in Use

```bash
# Find process
lsof -i :5069

# Kill it
kill -9 <PID>
```

### RAG Not Working

```bash
# Check doc folder
ls doc/

# Re-index
python -c "from bayazid import rag; rag.reindex_failed()"
```

---

## License

Built for educational purposes. Customize freely.

---

*"Execution over illusion. Systems over chaos. Growth over dependency."* 🐸

*"Always ♡"* 🌸