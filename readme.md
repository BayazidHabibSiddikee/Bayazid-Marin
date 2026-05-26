# BAYAZID HS-02
### Cognitive Warfare System — Productivity-Focused AI Study Partner

> *"Execution over illusion. Systems over chaos. Growth over dependency."* 🐸

A locally-running AI assistant built for engineers and self-directed learners.
Teaches, quizzes, reviews code, tracks focus sessions, and renders LaTeX math — all served from your own machine via a FastAPI backend and a **Black & Purple "Sword King"** web interface.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [Usage Guide](#usage-guide)
  - [Chat Mode](#chat-mode)
  - [Teach Mode](#teach-mode)
  - [Quiz Mode](#quiz-mode)
  - [Code Review](#code-review)
  - [Study Plan](#study-plan)
  - [Focus Timer](#focus-timer)
  - [Vision (Image Upload)](#vision-image-upload)
- [Slash Commands](#slash-commands)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Customization](#customization)

---

## Features

| Feature | Description |
|---|---|
| **Deep Chat** | Direct, no-fluff technical conversation with persistent memory (last 50 messages saved to disk) |
| **Teach Mode** | Structured explanations: concept → mechanism → working code → next step. Three depths: Quick / Standard / Deep |
| **Quiz Engine** | AI-generated MCQ quizzes on any topic. Easy / Medium / Hard. Instant answer feedback with explanations |
| **Code Review** | Structured review: bugs → efficiency → style → improved version |
| **Error Diagnosis** | Paste any traceback — get root cause and exact fix |
| **Study Plans** | Week-by-week learning plans tied to hands-on project tasks |
| **Focus Timer** | Named session tracker with live elapsed time, daily totals, and stats |
| **Vision Input** | Upload circuit diagrams, schematics, code screenshots — multimodal analysis |
| **LaTeX Rendering** | Full KaTeX support for math formulas in chat and on the landing page |
| **Intent Classifier** | Zero-overhead regex classifier routes messages to the right handler automatically |

---

## Project Structure

```
bayazid-hs02/
├── bayazid.py          # Core AI engine — all intelligence lives here
├── main.py             # FastAPI server — routes, endpoints, streaming
├── classifier.py       # Regex intent classifier — zero RAM, instant routing
├── config.py           # Shared constants — model, port, paths
│
├── templates/
│   ├── index.html      # Landing page / dashboard (KaTeX + stats)
│   ├── chat.html       # Main chat interface (all modes, timer, quiz UI)
│   └── profile.html    # Operator profile — focus stats, project tracker
│
└── static/
    └── uploads/        # Auto-created — stores uploaded images
```

---

## Requirements

### System
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally

### Ollama Models

Pull the models before starting:

```bash
# Primary model — used for all AI responses
ollama pull llama3.2:3b

# Fast model — used for quiz generation and quick tasks
ollama pull qwen2.5:0.5b
```

You can swap to any model. See [Configuration](#configuration).

### Python Packages

```bash
pip install fastapi uvicorn python-multipart ollama
```

Optional — for RAG (teaching from your own PDF book library):

```bash
pip install langchain langchain-community langchain-huggingface \
            langchain-text-splitters faiss-cpu pypdf sentence-transformers
```

---

## Installation

```bash
# 1. Clone or download the project
git clone <your-repo-url>
cd bayazid-hs02

# 2. Install dependencies
pip install fastapi uvicorn python-multipart ollama

# 3. Pull Ollama models
ollama pull llama3.2:3b
ollama pull qwen2.5:0.5b

# 4. (Optional) Create a doc/ folder and drop your PDFs in for RAG
mkdir doc
# copy your PDF books into doc/
```

---

## Running the Server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 5069 --reload
```

Then open your browser:

| Page | URL |
|---|---|
| Dashboard | http://localhost:5069 |
| Chat | http://localhost:5069/chat |
| Profile | http://localhost:5069/profile |

---

### Configuration

You can now manage models, API keys, and server settings in `settings.json`. The application will automatically load these settings on startup.

### Workspace Structure

- `storage/`: Contains all persistent state (chat history, vibe state, timer sessions, and FAISS database).
- `settings.json`: Your primary configuration file.
- `doc/`: Place your PDFs and documents here for RAG indexing.
- `code/`: Place your source code files here for RAG indexing.

```
What is the difference between I2C and SPI?
How do I set up UART on ESP32?
Explain the convolution operation in CNNs
```

### Teach Mode

Click **TEACH** in the sidebar or mode tabs, then describe what you want to learn. Bayazid structures the response as: concept → mechanism → working code example → next step.

Three depth levels (toggle with the **DEPTH** button):

| Depth | Description |
|---|---|
| `QUICK` | 3-minute explanation, core concept + one example only |
| `STANDARD` | Full explanation with working code and next step |
| `DEEP` | Internals, edge cases, common mistakes, project-level example |

```
Explain PWM control for servo motors
teach deep — how does PID control work
What is RTOS and how does it compare to bare-metal?
```

### Quiz Mode

Click **QUIZ** in the sidebar. A configuration panel appears:

1. Enter a topic (e.g. `I2C Protocol`, `Python decorators`, `CNN architecture`)
2. Choose difficulty: **Easy / Medium / Hard**
3. Choose number of questions: **3 / 5 / 10**
4. Click **LAUNCH ⚡**

The quiz renders as interactive buttons. Click an option — correct answers highlight green, wrong ones red, with an explanation shown immediately. Once answered, all options lock.

```
Topics: ESP32 GPIO, Kalman Filters, Binary Trees, UART Protocol,
        NumPy, PID Controllers, Linked Lists, Linux file permissions
```

### Code Review

Click **CODE REVIEW** in the sidebar, then paste your code (with or without triple-backtick fences). Bayazid returns a structured review:

- **Bugs** — correctness issues that must be fixed
- **Efficiency** — performance or memory concerns
- **Style** — readability and best practices
- **Improved Version** — rewritten code if changes are significant

```
review my code:
```python
def read_sensor():
    data = []
    while True:
        data.append(i2c.read())
```
```

### Study Plan

Click **STUDY PLAN** in the sidebar, then describe what you want to learn and optionally your goal:

```
Study plan for computer vision — goal: build a real-time face recognition system
Study plan for RTOS and FreeRTOS, 3 weeks
How to learn control systems from scratch
```

The plan is structured in phases (Foundation → Core Skills → Projects → Advanced) with specific daily tasks, mini-projects, and resource recommendations.

### Focus Timer

The timer strip runs across the top of the chat interface at all times.

**Starting a session:**
- Click **▶ START** in the timer strip, or
- Type `/timer start [task name]` in chat

**Stopping:**
- Click **⏹ STOP** (appears while session is active), or
- Type `/timer stop` in chat

The AI is aware of your active session — if you're mid-session, Bayazid factors your current task and elapsed time into responses.

**Checking stats:**
- `/timer status` — current session elapsed time
- `/timer stats` — today's total focus time and session count

The profile page at `/profile` shows a live session card and daily stats.

### Vision (Image Upload)

Click **📎 ATTACH** in the input area and select an image. Supported inputs:

- Circuit diagrams and schematics
- Code screenshots
- PCB layouts
- Error screenshots
- Whiteboard diagrams

Then describe what you want analyzed in the text input and send.

---

## Slash Commands

These work directly in the chat input in any mode:

| Command | Action |
|---|---|
| `/timer start [task]` | Begin a named focus session |
| `/timer stop` | End the current session |
| `/timer status` | Show elapsed time for current session |
| `/timer stats` | Show today's total focus time and session count |

Natural language also works — the classifier understands:

```
start a timer for ESP32 project
focus on control systems
how long have I been studying?
stop the session
```

---

## API Reference

All endpoints served at `http://localhost:5069`.

### Pages

| Method | Path | Description |
|---|---|---|
| GET | `/` | Landing page / dashboard |
| GET | `/chat` | Main chat interface |
| GET | `/profile` | Operator profile and stats |

### Chat & AI

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/message` | `message` (str), `image` (file, optional), `study_context` (JSON str, optional) | `text/plain` stream |
| POST | `/quiz/generate` | `topic` (str), `difficulty` (str), `num_questions` (int) | JSON quiz object |

### Timer

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/timer/start` | `task` (str) | JSON |
| POST | `/timer/stop` | — | JSON |
| POST | `/timer/status` | — | JSON |
| POST | `/timer/stats` | — | JSON |
| GET | `/timer/stats` | — | JSON stats object |

### Memory

| Method | Path | Response |
|---|---|---|
| POST | `/memory/clear` | `{"ok": true}` |
| GET | `/memory/status` | `{"message_count": N, "messages": [...]}` |

### Misc

| Method | Path | Response |
|---|---|---|
| POST | `/upload` | `{"ok": true, "path": "/static/uploads/..."}` |
| GET | `/health` | `{"status": "operational", "codename": "BAYAZID HS-02"}` |

---

## Configuration

All configuration lives in `config.py`:

```python
# Switch models here
DEFAULT_MODEL = "gemma4:31b-cloud"   # Main model for chat, teach, plans, review
FAST_MODEL    = "qwen2.5:0.5b"  # Fast model for quiz generation

# Server
HOST = "0.0.0.0"
PORT = 5069

# Session
MEMORY_MAX_MESSAGES   = 12   # Messages kept in active context window
POMODORO_WORK_MINUTES = 25   # Future pomodoro integration
```

To change the primary model everywhere in `bayazid.py`:

```python
MODEL = "llama3.2:3b"  # Line 20 — change this one line
```

Any Ollama-compatible model works: `mistral`, `deepseek-coder`, `phi3`, `gemma2`, etc.

---

## How It Works

```
User Input
    │
    ▼
classifier.py          — Regex intent detection (0ms, no model needed)
    │                    Intents: chat, teach, quiz, code_review,
    │                             debug, study_plan, timer, vision
    ▼
main.py (FastAPI)      — Routes to the correct handler based on intent
    │
    ▼
bayazid.py             — AI engine
    ├── main()              General chat with memory + timer context injection
    ├── teach_topic()       Structured teaching with 3 depth levels
    ├── generate_quiz()     JSON quiz generation with field normalization
    ├── create_study_plan() Phased study plans streamed as markdown
    ├── review_code()       Structured code review
    ├── explain_error()     Error diagnosis with exact fix
    ├── StudyTimer          Session tracking, daily totals, stats
    └── ConversationMemory  Persistent history (JSON file, last 50 messages)
    │
    ▼
Ollama (local)         — LLM inference, fully offline
    │
    ▼
StreamingResponse      — Chunks streamed to browser as they generate
    │
    ▼
chat.html              — renderMarkdown() + KaTeX renders the response live
```

---

## Customization

### Changing the User Profile

Edit `USER_CONTEXT` in `bayazid.py` to match your actual situation:

```python
USER_CONTEXT = """
User: Your Name
Location: Your City
Focus Areas: Your subjects
Active Projects: Your current builds
Preferences: Your communication style
"""
```

### Adding RAG (Teaching from Your Books)

1. Install RAG dependencies (see [Requirements](#requirements))
2. Create a `doc/` directory in the project root
3. Drop any PDF textbooks into `doc/`
4. Restart the server — Bayazid will index them automatically into a FAISS vector store
5. On subsequent starts, only new books are indexed (incremental update)

The knowledge base is stored in `faiss_db/`. Corrupted or image-only PDFs are skipped and logged in `faiss_db/manifest.json`.



## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| AI Inference | Ollama (local, offline) |
| Frontend | Vanilla HTML/CSS/JS — no framework |
| Math Rendering | KaTeX 0.16.9 |
| Fonts | Cinzel, Rajdhani, Share Tech Mono (Google Fonts) |
| Vector Store (optional) | FAISS + HuggingFace all-MiniLM-L6-v2 |
| Memory Persistence | JSON file (`bayazid_history.json`) |

---

*Built for: Bayazid — Rajshahi, Bangladesh*
*Codename: HS-02 | Status: OPERATIONAL* 🐸
