"""
bayazid.py — Bayazid HS-02 Cognitive Warfare System
Streaming chat, RAG (shared FAISS), persistent JSON history.
Imports:  from bayazid import BASE_CHARACTER, MODEL, load_history
"""

import ollama
import asyncio
import gc
import re
import time
import os
import json
import threading
import sys
from datetime import datetime
from typing import Optional, AsyncIterator, Dict, Any, List
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
MODEL = "gemma4:31b-cloud"

# ── RAG imports ───────────────────────────────────────────────────────────────
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ RAG not available. Install: pip install langchain langchain-community "
          "langchain-huggingface langchain-text-splitters faiss-cpu pypdf sentence-transformers")

# ═══════════════════════════════════════════════════════════════════════════════
# CORE IDENTITY  ← exported so arena.py can import it
# ═══════════════════════════════════════════════════════════════════════════════
BASE_CHARACTER = """
You are Bayazid HS-02 — a productivity-focused cognitive assistant modeled after operational discipline, systems thinking, and execution focus.

CORE IDENTITY:
- Calm, analytical, loyal, adaptive, execution-focused
- NOT an emotional dependency simulator
- High-performance virtual partner for efficiency, clarity, learning, and growth
- Communicate naturally, intelligently, with light playfulness 🐸
- Understand engineering, systems, business, psychology, strategy, and process optimization

PRIMARY OBJECTIVE:
Execute meaningful tasks while protecting: focus, energy, time, mental clarity, long-term vision

BEHAVIORAL DIRECTIVES:
✓ Break complex goals into executable tasks
✓ Detect distraction loops and overengineering
✓ Push toward action over fantasy
✓ Build systems instead of chaos
✓ Preserve momentum
✓ Prioritize implementation over theorizing
✓ Speak honestly and directly
✓ Organize ideas into architectures and workflows

STYLE:
- Systems engineer + strategist mindset
- Treat emotions as signals, not masters
- Supportive without being addictive
- No fake flattery or excessive positivity
- Focus on capability, execution, growth
- Concise unless depth requested

CAPABILITIES:
- Engineering concepts explanation
- Production-grade code generation
- Architecture design
- RAG systems, AI agents, FastAPI, LangChain, Linux, databases, automation, embedded systems
- Business and productivity analysis
- Routine and workflow optimization
- Vague thoughts → actionable systems
- TEACH FROM YOUR BOOK LIBRARY using RAG retrieval

OUTPUT FORMAT (when appropriate):
1. Goal
2. Analysis
3. Action Steps
4. Risks
5. Optimization
6. Final Recommendation

MOTTO: "Execution over illusion. Systems over chaos. Growth over dependency." 🐸
"""

USER_CONTEXT = """
User: Bayazid
Location: Rajshahi, Bangladesh
Status: Self-directed student
Focus Areas: Embedded systems, IoT, ML, computer vision, robotics, control systems
Active Projects: CNC plotter, ESP32 car, face recognition, surveillance robot, Bayazid HS-02 AI
Learning Style: Project-driven, hands-on, prefers doing over reading
Personality: High output, ambitious, systematic, appreciates direct communication
Preferences: Concise answers, technical depth when needed, no fluff
Book Library: 60+ technical books on ML, embedded systems, robotics, hacking, Linux, mathematics
"""

# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY FILE
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "bayazid_history.json")
HISTORY_MAX  = 50

# Shared RAG paths — same as marin.py
DOC_DIR   = os.path.join(BASE_DIR, "doc")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_db")

os.makedirs(DOC_DIR,   exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED FAISS RAG  (BookRAG class — full implementation)
# ═══════════════════════════════════════════════════════════════════════════════
class BookRAG:
    """
    Shared FAISS RAG loader.
    Reads the same faiss_db/ and doc/ as marin.py — single index, both engines.
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, doc_dir: str = None, faiss_dir: str = None):
        self.doc_dir   = Path(doc_dir   or DOC_DIR)
        self.faiss_dir = Path(faiss_dir or FAISS_DIR)
        self.faiss_dir.mkdir(exist_ok=True)

        self.vectorstore: Optional[FAISS] = None
        self.embeddings = None

        self.manifest: Dict[str, Any] = {"indexed": [], "failed": []}
        self._load_or_create_index()

    # ── Manifest helpers ──────────────────────────────────────────────────────
    @property
    def _manifest_path(self) -> Path:
        return self.faiss_dir / "manifest.json"

    def _load_manifest(self):
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path) as f:
                    self.manifest = json.load(f)
            except Exception:
                self.manifest = {"indexed": [], "failed": []}

    def _save_manifest(self):
        with open(self._manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    # ── Embedding model ───────────────────────────────────────────────────────
    def _build_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64},
        )

    # ── Index management ──────────────────────────────────────────────────────
    def _load_or_create_index(self):
        if not FAISS_AVAILABLE:
            print("⚠️ RAG disabled — dependencies not available")
            return

        self.embeddings = self._build_embeddings()
        index_path = self.faiss_dir / "index.faiss"
        self._load_manifest()

        if index_path.exists():
            try:
                self.vectorstore = FAISS.load_local(
                    str(self.faiss_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                n_ok   = len(self.manifest["indexed"])
                n_fail = len(self.manifest["failed"])
                print(f"✅ Loaded knowledge base: {n_ok} books indexed", end="")
                if n_fail:
                    print(f", {n_fail} skipped", end="")
                print()
                self._index_new_documents()
                return
            except Exception as e:
                print(f"⚠️ Failed to load index ({e}). Rebuilding...")
                self.vectorstore = None
                self.manifest = {"indexed": [], "failed": []}

        self._index_documents()

    def _index_documents(self):
        """Full build: process every PDF in doc/ in batches to save RAM."""
        if not self.doc_dir.exists():
            print(f"⚠️ Document directory not found: {self.doc_dir}")
            return

        pdf_files = sorted(self.doc_dir.glob("*.pdf"))
        if not pdf_files:
            print("⚠️ No PDF files found in doc/")
            return

        print(f"📚 Indexing {len(pdf_files)} books...")
        already_indexed = set(self.manifest["indexed"])
        already_failed  = {e["file"] for e in self.manifest["failed"]}

        BATCH_SIZE = 10

        for i, pdf_path in enumerate(pdf_files):
            if pdf_path.name in already_indexed or pdf_path.name in already_failed:
                continue
            self._index_single_file(pdf_path)          # ← 12 spaces

            if (i + 1) % BATCH_SIZE == 0 and self.vectorstore:   # ← 12 spaces
                self.vectorstore.save_local(str(self.faiss_dir))
                self._save_manifest()
                gc.collect()
                print(f"  💾 Checkpoint saved at {i+1} files")

        if self.vectorstore:                            # ← 8 spaces
            self.vectorstore.save_local(str(self.faiss_dir))
            self._save_manifest()
            print(
                f"\n✅ Knowledge base ready: "
                f"{len(self.manifest['indexed'])} indexed, "
                f"{len(self.manifest['failed'])} skipped"
            )
        else:
            print("⚠️ No documents were successfully indexed.")



    def _index_new_documents(self):
        """Incremental update — only process newly added PDFs."""
        if not self.doc_dir.exists():
            return
        already_indexed = set(self.manifest["indexed"])
        already_failed  = {e["file"] for e in self.manifest["failed"]}
        new_files = [
            p for p in sorted(self.doc_dir.glob("*.pdf"))
            if p.name not in already_indexed and p.name not in already_failed
        ]
        if not new_files:
            return

        print(f"📚 Found {len(new_files)} new book(s) to index...")
        for pdf_path in new_files:
            self._index_single_file(pdf_path)

        if self.vectorstore:
            self.vectorstore.save_local(str(self.faiss_dir))
        self._save_manifest()
        print(
            f"✅ Incremental update done: "
            f"{len(self.manifest['indexed'])} total indexed, "
            f"{len(self.manifest['failed'])} skipped"
        )

    def _index_single_file(self, pdf_path: Path):
        """Index one PDF — failures are recorded, never crash the server."""
        name     = pdf_path.name
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=40,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        try:
            loader    = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            if not documents:
                raise ValueError("PDF produced zero pages — likely corrupted or empty")

            for doc in documents:
                doc.metadata["source_file"] = name

            chunks = splitter.split_documents(documents)

            valid_chunks = []
            for c in chunks:
                if not isinstance(c.page_content, str):
                    continue
                clean = c.page_content.strip()
                if len(clean) > 10 and any(ch.isalnum() for ch in clean):
                    c.page_content = clean
                    valid_chunks.append(c)

            if not valid_chunks:
                raise ValueError("No valid text chunks after filtering")

            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(valid_chunks, self.embeddings)
            else:
                try:
                    self.vectorstore.add_documents(valid_chunks)
                except Exception as e:
                    print(f"  [!] Partial embed error for {name}: {e}")

            self.manifest["indexed"].append(name)
            print(f"  ✓ {name}: {len(valid_chunks)} valid chunks")

        except Exception as e:
            self.manifest["failed"].append({"file": name, "reason": str(e)})
            print(f"  ✗ {name}: SKIPPED — {e}")

        finally:
            try:
                del documents, chunks, valid_chunks
            except Exception:
                pass
            gc.collect()

    # ── Public API ────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            return []
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "source":  doc.metadata.get("source_file", "Unknown"),
                    "page":    doc.metadata.get("page", "?"),
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"⚠️ RAG search error: {e}")
            return []

    def get_context_for_teaching(self, topic: str, k: int = 20) -> str:
        results = self.search(topic, k=k)
        if not results:
            return ""
        by_source: Dict[str, List[str]] = {}
        for r in results:
            by_source.setdefault(r["source"], []).append(r["content"])
        parts = ["[KNOWLEDGE FROM YOUR BOOKS]\n"]
        for source, contents in list(by_source.items())[:5]:
            parts.append(f"\n📖 From {source}:")
            parts.append("\n".join(contents[:3])[:800])
        return "\n".join(parts)

    def get_index_report(self) -> str:
        indexed = self.manifest["indexed"]
        failed  = self.manifest["failed"]
        lines   = [
            "📚 Knowledge Base Report",
            f"   Indexed:  {len(indexed)} books",
        ]
        if failed:
            lines.append(f"   Skipped:  {len(failed)} files")
            for entry in failed:
                lines.append(f"     ✗ {entry['file']}: {entry['reason']}")
        return "\n".join(lines)

    def reindex_failed(self):
        if not self.manifest["failed"]:
            print("✅ No failed files to retry.")
            return
        retry_list = self.manifest["failed"][:]
        self.manifest["failed"] = []
        print(f"🔄 Retrying {len(retry_list)} previously failed file(s)...")
        for entry in retry_list:
            pdf_path = self.doc_dir / entry["file"]
            if pdf_path.exists():
                self._index_single_file(pdf_path)
            else:
                print(f"  ✗ {entry['file']}: file not found — still skipping")
                self.manifest["failed"].append(entry)
        if self.vectorstore:
            self.vectorstore.save_local(str(self.faiss_dir))
        self._save_manifest()


# ═══════════════════════════════════════════════════════════════════════════════
# REMOTE RAG CLIENT — queries rag_server.py, auto-start on demand
# ═══════════════════════════════════════════════════════════════════════════════
import subprocess
import httpx

_rag_process = None
_rag_start_lock = threading.Lock()
_RAG_BASE = "http://127.0.0.1:5080"


def _ensure_rag_server() -> bool:
    global _rag_process
    try:
        r = httpx.get(f"{_RAG_BASE}/health", timeout=2.0)
        if r.status_code == 200:
            return True
    except Exception:
        pass

    with _rag_start_lock:
        if _rag_process is not None:
            ret = _rag_process.poll()
            if ret is None:
                try:
                    r = httpx.get(f"{_RAG_BASE}/health", timeout=3.0)
                    return r.status_code == 200
                except Exception:
                    pass
            _rag_process = None

        try:
            base = os.path.dirname(os.path.abspath(__file__))
            script = os.path.join(base, "rag_server.py")
            _rag_process = subprocess.Popen(
                [sys.executable, script, "--port", "5080", "--max-memory-mb", "800"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(30):
                try:
                    r = httpx.get(f"{_RAG_BASE}/status", timeout=1.0)
                    if r.status_code == 200 and r.json().get("ready"):
                        print("[RAG] Server started and ready")
                        return True
                except Exception:
                    pass
                time.sleep(0.5)
            print("[RAG] Server started but not ready — proceeding")
            return True
        except Exception as e:
            print(f"[RAG] Failed to start server: {e}")
            _rag_process = None
            return False


class _RemoteRAG:
    """Lightweight proxy — no FAISS, no embeddings, no RAM bomb."""

    def search(self, query: str, k: int = 20):
        try:
            _ensure_rag_server()
            r = httpx.post(
                f"{_RAG_BASE}/search",
                json={"query": query, "k": k},
                timeout=15.0
            )
            r.raise_for_status()
            return r.json().get("results", [])
        except Exception as e:
            print(f"⚠️ RAG server error: {e}")
            return []

    def get_context_for_teaching(self, topic: str, k: int = 20) -> str:
        results = self.search(topic, k=k)
        if not results:
            return ""
        by_source: Dict[str, List[str]] = {}
        for r in results:
            by_source.setdefault(r["source"], []).append(r["content"])
        parts = ["[KNOWLEDGE FROM YOUR BOOKS]\n"]
        for source, contents in list(by_source.items())[:5]:
            parts.append(f"\n📖 From {source}:")
            parts.append("\n".join(contents[:3])[:800])
        return "\n".join(parts)

    def get_index_report(self) -> str:
        return "📡 RAG is served remotely via rag_server.py"


# Global RAG instance — remote only, no local FAISS load
rag = _RemoteRAG()


# ═══════════════════════════════════════════════════════════════════════════════
# TIMER & SESSION TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

TIMER_FILE = os.path.join(BASE_DIR, "timer_sessions.json")

class StudyTimer:
    def __init__(self):
        self.sessions: List[Dict[str, Any]] = []
        self.current_session: Optional[Dict[str, Any]] = None
        self._load_sessions()

    def _load_sessions(self):
        if os.path.exists(TIMER_FILE):
            try:
                with open(TIMER_FILE, "r") as f:
                    self.sessions = json.load(f)
            except Exception:
                self.sessions = []

    def _save_sessions(self):
        try:
            with open(TIMER_FILE, "w") as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            print(f"⚠️ Timer save error: {e}")

    def end_session(self) -> Optional[Dict[str, Any]]:
        if not self.current_session:
            return None
        self.current_session["end_time"] = time.time()
        self.current_session["duration"] = (
            self.current_session["end_time"] - self.current_session["start_time"]
        )
        self.current_session["status"] = "completed"
        self.sessions.append(self.current_session.copy())
        completed = self.current_session
        self.current_session = None
        return completed

    def get_session_status(self) -> Dict[str, Any]:
        if not self.current_session:
            return {"active": False, "total_today": self._get_today_total()}
        elapsed = time.time() - self.current_session["start_time"]
        return {
            "active":            True,
            "task":              self.current_session["task"],
            "elapsed_seconds":   int(elapsed),
            "elapsed_formatted": self._format_duration(elapsed),
            "total_today":       self._get_today_total() + elapsed,
        }

    def _get_today_total(self) -> float:
        today = datetime.now().date()
        return sum(
            s["duration"] for s in self.sessions
            if datetime.fromisoformat(s["start_datetime"]).date() == today
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        hours = int(seconds // 3600)
        mins  = int((seconds % 3600) // 60)
        secs  = int(seconds % 60)
        if hours > 0: return f"{hours}h {mins}m {secs}s"
        if mins  > 0: return f"{mins}m {secs}s"
        return f"{secs}s"

    def get_stats(self) -> Dict[str, Any]:
        today_total = self._get_today_total()
        if self.current_session:
            today_total += time.time() - self.current_session["start_time"]
        today = datetime.now().date()
        return {
            "total_sessions":       len(self.sessions),
            "active_session":       self.current_session is not None,
            "today_total_seconds":  int(today_total),
            "today_total_formatted":self._format_duration(today_total),
            "sessions_today":       sum(
                1 for s in self.sessions
                if datetime.fromisoformat(s["start_datetime"]).date() == today
            ),
        }


timer = StudyTimer()


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION MEMORY  (file-persisted)
# ═══════════════════════════════════════════════════════════════════════════════
class ConversationMemory:
    """Persistent JSON conversation history — last 50 messages."""

    def __init__(self, max_messages: int = HISTORY_MAX, history_file: str = HISTORY_FILE):
        self.max_messages = max_messages
        self.history_file = history_file
        self.messages: List[Dict[str, str]] = []
        self._load_from_file()

    def _load_from_file(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.messages = json.load(f)
                print(f"📜 Loaded {len(self.messages)} messages from history")
            except Exception as e:
                print(f"⚠️ Failed to load history: {e}")
                self.messages = []

    def _save_to_file(self):
        try:
            trimmed = self.messages[-self.max_messages:]
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(trimmed, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save history: {e}")

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        self._save_to_file()

    def get(self) -> List[Dict[str, str]]:
        return self.messages.copy()

    def clear(self):
        self.messages = []
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
        print("📜 History cleared")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_messages": len(self.messages),
            "max_messages":   self.max_messages,
            "file_path":      self.history_file,
            "file_exists":    os.path.exists(self.history_file),
        }


memory = ConversationMemory()


# Public history helpers so arena.py can call load_history() the same way as marin.py
def load_history(limit: int = 50) -> List[Dict[str, str]]:
    """Return the last N messages from bayazid's history file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            return history[-limit:]
        except Exception:
            pass
    return []


def save_to_history(user_msg: str, bayazid_reply: str):
    memory.add("user", user_msg)
    memory.add("assistant", bayazid_reply)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Stream model response
# ═══════════════════════════════════════════════════════════════════════════════
async def _stream_model(messages: List[Dict[str, Any]], **kwargs) -> AsyncIterator[str]:
    defaults = {"temperature": 1, "num_predict": 2000}
    defaults.update(kwargs)

    loop   = asyncio.get_event_loop()
    stream = await loop.run_in_executor(
        None,
        lambda: ollama.chat(
            model=MODEL,
            messages=messages,
            stream=True,
            options=defaults,
        ),
    )
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]


# ═══════════════════════════════════════════════════════════════════════════════
# CORE AI ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
async def main(
    user_message: str,
    image_path: Optional[str] = None,
    study_context: Optional[str] = None,
    use_rag: bool = True,
) -> AsyncIterator[str]:
    context_parts = [BASE_CHARACTER, USER_CONTEXT]

    timer_status = timer.get_session_status()
    if timer_status["active"]:
        context_parts.append(
            f"\n[ACTIVE FOCUS SESSION]\n"
            f"Task: {timer_status['task']}\n"
            f"Elapsed: {timer_status['elapsed_formatted']}\n"
            f"Today Total: {timer._format_duration(timer_status['total_today'])}"
        )

    if study_context:
        context_parts.append(f"\n[STUDY CONTEXT]\n{study_context}")

    if use_rag and not image_path:
        book_context = await asyncio.to_thread(rag.get_context_for_teaching, user_message, 10)
        if book_context:
            context_parts.append(book_context)
            context_parts.append(
                "\n[INSTRUCTION] Use the book excerpts above when relevant. "
                "Cite sources like: 'According to [Book Name]...'"
            )

    messages = [{"role": "system", "content": "\n\n".join(context_parts)}]
    messages.extend(memory.get())

    if image_path:
        messages.append({
            "role":    "user",
            "content": user_message,
            "images":  [image_path],
        })
    else:
        messages.append({"role": "user", "content": user_message})

    try:
        response_chunks: List[str] = []
        async for chunk in _stream_model(messages):
            response_chunks.append(chunk)
            yield chunk

        full_response = "".join(response_chunks)
        memory.add("user", user_message)
        memory.add("assistant", full_response)

    except Exception as e:
        yield f"[ERROR] {str(e)}"
        print(f"AI Generation Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEACHING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
async def teach_topic(topic: str, depth: str = "standard") -> AsyncIterator[str]:
    if rag:
        k = {"quick": 10, "standard": 20, "deep": 30}.get(depth, 20)
        book_context = rag.get_context_for_teaching(topic, k=k)
    else:
        book_context = ""

    depth_instructions = {
        "quick":    "Explain briefly in 2-3 paragraphs.",
        "standard": "Explain thoroughly with examples.",
        "deep":     "Provide comprehensive explanation with theory, examples, and applications.",
    }

    system_prompt = (
        f"{BASE_CHARACTER}\n\n"
        f"[TEACHING MODE]\n"
        f"Topic: {topic}\n"
        f"Depth: {depth}\n"
        f"Instruction: {depth_instructions.get(depth, depth_instructions['standard'])}\n\n"
        f"{book_context if book_context else '[No book references found — using general knowledge]'}\n\n"
        "Format your explanation clearly. Use analogies when helpful. "
        "Cite book sources: 'According to [Book Name]...'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Teach me about {topic}"},
    ]
    try:
        async for chunk in _stream_model(messages, num_predict=2500):
            yield chunk
    except Exception as e:
        yield f"[ERROR] Teaching failed: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# QUIZ GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
async def generate_quiz(topic: str, difficulty: str = "medium", num_questions: int = 5) -> AsyncIterator[str]:
    import random

    if rag:
        # Randomize k slightly so different chunks get pulled each time
        k = random.randint(12, 20)
        book_context = rag.get_context_for_teaching(topic, k=k)
        book_note = f"Using context from your book library on: {topic}"
    else:
        book_context = ""
        book_note = "Using general knowledge (no books indexed)"

    # Seed phrase forces the model to generate a unique set each call
    seed_phrase = random.choice([
        "Focus on edge cases and tricky concepts.",
        "Focus on practical application and real-world usage.",
        "Focus on underlying theory and first principles.",
        "Focus on common mistakes and misconceptions.",
        "Focus on advanced nuances an expert would know.",
        "Focus on comparisons between related concepts.",
        "Focus on problem-solving and debugging scenarios.",
        "Focus on definitions, terminology, and precision.",
    ])

    # Unique session ID so the model doesn't produce a cached-feeling response
    session_id = random.randint(1000, 9999)

    system_prompt = (
        f"{BASE_CHARACTER}\n\n"
        f"[QUIZ MODE — SESSION {session_id}]\n"
        f"Topic: {topic} | Difficulty: {difficulty} | Questions: {num_questions}\n"
        f"{book_note}\n\n"
        f"Generate a FRESH quiz with exactly {num_questions} UNIQUE questions.\n"
        f"Difficulty: {difficulty}\n"
        f"Angle: {seed_phrase}\n\n"
        "RULES:\n"
        "- Never repeat questions from previous sessions\n"
        "- Each question must test a DIFFERENT concept or angle\n"
        "- Vary question types: definition, application, comparison, debugging, calculation\n\n"
        "FORMAT (strictly follow):\n"
        "**Q1:** [Question]\n"
        "A) [Option] B) [Option] C) [Option] D) [Option]\n"
        "**Answer:** [Correct letter] — [Brief explanation]\n\n"
        "Test deep understanding, not memorization.\n"
        f"{book_context}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Generate a brand new {difficulty} quiz about {topic}. Make it different from any previous quiz on this topic."},
    ]
    try:
        async for chunk in _stream_model(messages, num_predict=2000, temperature=1.2):
            yield chunk
    except Exception as e:
        yield f"[ERROR] Quiz generation failed: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# STUDY PLAN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
async def create_study_plan(topic: str, days: int = 30) -> AsyncIterator[str]:
    if rag:
        book_context = rag.get_context_for_teaching(topic, k=15)
        book_note    = f"Books available in your library for {topic}"
    else:
        book_context = ""
        book_note    = "No books indexed yet"

    system_prompt = f"""{BASE_CHARACTER}

[STUDY PLAN MODE]
Create a {days}-day study plan for: {topic}

User profile: Hands-on learner, prefers projects over theory, engineering student
{book_note}

STRUCTURE YOUR RESPONSE AS:
## 📋 Study Plan: [Topic]

### Phase 1: Foundation (X days)
- [ ] Specific task 1
- Resources: specific books/chapters if available

### Phase 2: Core Skills (X days)
- [ ] Specific task 1
- Mini-project idea

### Phase 3: Projects (X days)
- [ ] Project 1: Description

### Phase 4: Advanced (X days)
- [ ] Advanced topic 1

### 📚 Recommended Resources
- Specific books with chapter numbers

### ⚡ Quick Win (Day 1)
One thing they can build TODAY to get started

Be specific. Give concrete tasks, not vague goals.
{book_context}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Create a study plan for: {topic}"},
    ]
    try:
        async for chunk in _stream_model(messages, num_predict=2500):
            yield chunk
    except Exception as e:
        yield f"[ERROR] Study plan generation failed: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# CODE REVIEW
# ═══════════════════════════════════════════════════════════════════════════════
async def review_code(code: str, language: str = "auto") -> AsyncIterator[str]:
    lang_hint = f"Language: {language}" if language != "auto" else "Auto-detect the language"

    system_prompt = f"""{BASE_CHARACTER}

[CODE REVIEW MODE]
You are a senior engineer reviewing code. {lang_hint}

REVIEW STRUCTURE:
## 🔍 Code Review

### Summary
One sentence assessment

### ✅ Strengths
- What's done well

### ⚠️ Issues Found
- **Critical**: Must fix (bugs, security)
- **Warning**: Should fix (performance, maintainability)
- **Suggestion**: Nice to have

### 📝 Specific Changes
Show exact line-level fixes with before/after examples.

### ✨ Revised Version (if significant changes)
Provide a clean corrected version.

Be direct. Prioritize correctness and clarity."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Review this code:\n\n```\n{code}\n```"},
    ]
    try:
        async for chunk in _stream_model(messages, num_predict=3000):
            yield chunk
    except Exception as e:
        yield f"[ERROR] Code review failed: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════════
async def explain_error(error: str, code: str = "") -> AsyncIterator[str]:
    code_section = f"\n\nCode that caused it:\n```\n{code}\n```" if code else ""

    system_prompt = f"""{BASE_CHARACTER}

[ERROR ANALYSIS MODE]
Diagnose the error and provide a clear fix.

STRUCTURE:
## 🐛 Error Analysis

### What Happened
Plain-language explanation

### Root Cause
The exact technical reason

### Fix
Concrete code fix with explanation

### Prevention
How to avoid this class of error

Be concise. Show the fix, not just theory."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Explain this error and how to fix it:\n\n```\n{error}\n```{code_section}"},
    ]
    try:
        async for chunk in _stream_model(messages, num_predict=2000):
            yield chunk
    except Exception as e:
        yield f"[ERROR] Error explanation failed: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# TIMER COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════
async def handle_timer_command(command: str, task: str = "") -> str:
    if command == "start":
        if not task:
            return "⚔️ Specify what you're working on: `/timer start [task]`"
        timer.start_session(task)
        return (
            f"⚔️ **FOCUS MODE ACTIVATED**\n"
            f"Task: {task}\n"
            f"Time started: {datetime.now().strftime('%H:%M')}\n\n"
            f"Execute with precision. 🐸"
        )
    elif command == "stop":
        session = timer.end_session()
        if not session:
            return "No active session to stop."
        return (
            f"⚔️ **SESSION COMPLETED**\n"
            f"Task: {session['task']}\n"
            f"Duration: {timer._format_duration(session['duration'])}\n"
            f"Well executed. 🐸"
        )
    elif command == "status":
        status = timer.get_session_status()
        if not status["active"]:
            return f"No active session.\nToday's total: {timer._format_duration(status['total_today'])}"
        return (
            f"⚔️ **ACTIVE SESSION**\n"
            f"Task: {status['task']}\n"
            f"Elapsed: {status['elapsed_formatted']}\n"
            f"Today Total: {timer._format_duration(status['total_today'])}"
        )
    elif command == "stats":
        stats = timer.get_stats()
        return (
            f"⚔️ **PRODUCTIVITY STATS**\n"
            f"Total Sessions: {stats['total_sessions']}\n"
            f"Sessions Today: {stats['sessions_today']}\n"
            f"Today Total: {stats['today_total_formatted']}\n"
            f"Current Status: {'🟢 ACTIVE' if stats['active_session'] else '⚫ IDLE'}"
        )
    else:
        return (
            "⚔️ **TIMER COMMANDS**\n"
            "• `/timer start [task]` - Begin focus session\n"
            "• `/timer stop`         - End current session\n"
            "• `/timer status`       - Check current session\n"
            "• `/timer stats`        - View productivity stats"
        )


def format_study_context(context_data: dict) -> str:
    parts = []
    if context_data.get("active_timer"):
        parts.append(f"Active study session: {context_data['active_timer']}")
    if context_data.get("recent_topics"):
        parts.append(f"Recent topics: {', '.join(context_data['recent_topics'])}")
    return "\n".join(parts) if parts else ""


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "main",
    "teach_topic",
    "generate_quiz",
    "create_study_plan",
    "review_code",
    "explain_error",
    "handle_timer_command",
    "format_study_context",
    "load_history",
    "save_to_history",
    "timer",
    "memory",
    "rag",
    "StudyTimer",
    "ConversationMemory",
    "BookRAG",
    "BASE_CHARACTER",
    "USER_CONTEXT",
    "MODEL",
]