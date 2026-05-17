"""
BAYAZID HS-02 — Cognitive Warfare System
A productivity-focused AI assistant with RAG-powered learning from your books
"""

import ollama
import asyncio
import gc
import re
import time
import os
import json
from datetime import datetime
from typing import Optional, AsyncIterator, Dict, Any, List
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION — Change this one line to switch models everywhere
# ═══════════════════════════════════════════════════════════════════════════
MODEL = "gemma4:31b-cloud"

# RAG imports
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ RAG not available. Install: pip install langchain langchain-community langchain-huggingface langchain-text-splitters faiss-cpu pypdf sentence-transformers")

# ═══════════════════════════════════════════════════════════════════════════
# CORE IDENTITY
# ═══════════════════════════════════════════════════════════════════════════

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
- **TEACH FROM YOUR BOOK LIBRARY** using RAG retrieval

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

# ═══════════════════════════════════════════════════════════════════════════
# HISTORY FILE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "bayazid_history.json")
HISTORY_MAX = 50  # Keep last 50 messages (25 turns)

# ═══════════════════════════════════════════════════════════════════════════
# RAG SYSTEM (Knowledge Base from Books)
# ═══════════════════════════════════════════════════════════════════════════

class BookRAG:
    """
    RAG system for teaching from your book library.

    Design decisions:
    - HuggingFace all-MiniLM-L6-v2: offline, fast, no Ollama dependency
    - Strict chunk filtering prevents TextEncodeInput crashes
    - One book at a time + gc.collect() keeps RAM flat across 60+ PDFs
    - manifest.json tracks indexed + failed files; corrupted PDFs are
      skipped permanently until you call reindex_failed()
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, doc_dir: str = "doc", faiss_dir: str = "faiss_db"):
        self.doc_dir   = Path(doc_dir)
        self.faiss_dir = Path(faiss_dir)
        self.faiss_dir.mkdir(exist_ok=True)

        self.vectorstore: Optional[FAISS] = None
        self.embeddings  = None

        self.manifest: Dict[str, Any] = {
            "indexed": [],   # filenames successfully indexed
            "failed":  []    # [{"file": ..., "reason": ...}, ...]
        }

        self._load_or_create_index()

    # ── Manifest helpers ──────────────────────────────────────────────────

    @property
    def _manifest_path(self) -> Path:
        return self.faiss_dir / "manifest.json"

    def _load_manifest(self):
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, "r") as f:
                    self.manifest = json.load(f)
            except Exception:
                self.manifest = {"indexed": [], "failed": []}

    def _save_manifest(self):
        with open(self._manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    # ── Embedding model ───────────────────────────────────────────────────

    def _build_embeddings(self) -> "HuggingFaceEmbeddings":
        """Load the HuggingFace embedding model into RAM (once)."""
        return HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64}
        )

    # ── Index management ──────────────────────────────────────────────────

    def _load_or_create_index(self):
        """Load existing FAISS index, or build from scratch."""
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
                    allow_dangerous_deserialization=True  # safe: local files only
                )
                n_ok   = len(self.manifest["indexed"])
                n_fail = len(self.manifest["failed"])
                print(f"✅ Loaded knowledge base: {n_ok} books indexed", end="")
                if n_fail:
                    print(f", {n_fail} skipped (corrupted/unreadable)", end="")
                print()
                self._index_new_documents()
                return
            except Exception as e:
                print(f"⚠️ Failed to load existing index ({e}). Rebuilding...")
                self.vectorstore = None
                self.manifest = {"indexed": [], "failed": []}

        self._index_documents()

    def _index_documents(self):
        """Full build: process every PDF in doc/, one at a time."""
        if not self.doc_dir.exists():
            print(f"⚠️ Document directory not found: {self.doc_dir}")
            return

        pdf_files = sorted(self.doc_dir.glob("*.pdf"))
        if not pdf_files:
            print("⚠️ No PDF files found in doc/")
            return

        print(f"📚 Indexing {len(pdf_files)} books (one at a time to protect RAM)...")

        already_indexed = set(self.manifest["indexed"])
        already_failed  = {e["file"] for e in self.manifest["failed"]}

        for pdf_path in pdf_files:
            if pdf_path.name in already_indexed or pdf_path.name in already_failed:
                continue
            self._index_single_file(pdf_path)

        if self.vectorstore:
            self.vectorstore.save_local(str(self.faiss_dir))
            self._save_manifest()
            print(f"\n✅ Knowledge base ready: "
                  f"{len(self.manifest['indexed'])} indexed, "
                  f"{len(self.manifest['failed'])} skipped")
        else:
            print("⚠️ No documents were successfully indexed.")

    def _index_new_documents(self):
        """Incremental update — only process PDFs added since last run."""
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
        print(f"✅ Incremental update done: "
              f"{len(self.manifest['indexed'])} total indexed, "
              f"{len(self.manifest['failed'])} skipped")

    def _index_single_file(self, pdf_path: Path):
        """
        Index one PDF.  Any failure → recorded in manifest, never retried
        automatically.  RAM freed with gc.collect() after each book.
        """
        name = pdf_path.name
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=40,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        try:
            # ── Load ──────────────────────────────────────────────────────
            loader    = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            if not documents:
                raise ValueError("PDF produced zero pages — likely corrupted or empty")

            # Tag source
            for doc in documents:
                doc.metadata["source_file"] = name

            # ── Split ─────────────────────────────────────────────────────
            chunks = splitter.split_documents(documents)

            # ── ULTRA-STRICT FILTER (prevents TextEncodeInput crash) ───────
            # .strip() first, then check length AND alphanumeric content
            valid_chunks = []
            for c in chunks:
                if not isinstance(c.page_content, str):
                    continue
                clean = c.page_content.strip()
                if len(clean) > 10 and any(ch.isalnum() for ch in clean):
                    c.page_content = clean   # store the stripped version
                    valid_chunks.append(c)

            if not valid_chunks:
                raise ValueError(
                    "No valid text chunks after filtering — "
                    "scanned image PDF or all pages empty"
                )

            # ── Add to vectorstore ─────────────────────────────────────────
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(valid_chunks, self.embeddings)
            else:
                try:
                    self.vectorstore.add_documents(valid_chunks)
                except Exception as e:
                    # A bad chunk slipped through — log but don't lose the book
                    print(f"  [!] Partial embed error for {name}: {e}")

            self.manifest["indexed"].append(name)
            print(f"  ✓ {name}: {len(valid_chunks)} valid chunks")

        except Exception as e:
            self.manifest["failed"].append({"file": name, "reason": str(e)})
            print(f"  ✗ {name}: SKIPPED — {e}")

        finally:
            # Always release memory, even if indexing failed
            try:
                del documents, chunks, valid_chunks
            except Exception:
                pass
            gc.collect()

    # ── Public API ────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """Semantic search across all indexed books."""
        if not self.vectorstore:
            return []
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "source":  doc.metadata.get("source_file", "Unknown"),
                    "page":    doc.metadata.get("page", "?")
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"⚠️ RAG search error: {e}")
            return []

    def get_context_for_teaching(self, topic: str, k: int = 20) -> str:
        """Retrieve and format relevant book excerpts for a topic."""
        results = self.search(topic, k=k)
        if not results:
            return ""

        by_source: Dict[str, List[str]] = {}
        for r in results:
            by_source.setdefault(r["source"], []).append(r["content"])

        context_parts = ["[KNOWLEDGE FROM YOUR BOOKS]\n"]
        for source, contents in list(by_source.items())[:5]:
            context_parts.append(f"\n📖 From {source}:")
            context_parts.append("\n".join(contents[:3])[:800])

        return "\n".join(context_parts)

    def get_index_report(self) -> str:
        """Human-readable report of what's in the knowledge base."""
        indexed = self.manifest["indexed"]
        failed  = self.manifest["failed"]
        lines   = [
            "📚 Knowledge Base Report",
            f"   Indexed:  {len(indexed)} books"
        ]
        if failed:
            lines.append(f"   Skipped:  {len(failed)} files (corrupted/unreadable)")
            for entry in failed:
                lines.append(f"     ✗ {entry['file']}: {entry['reason']}")
        return "\n".join(lines)

    def reindex_failed(self):
        """
        Retry previously failed files — useful after replacing a corrupted
        PDF with a clean copy.  Call manually: rag.reindex_failed()
        """
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


# Global RAG instance
rag = BookRAG() if FAISS_AVAILABLE else None

# ═══════════════════════════════════════════════════════════════════════════
# TIMER & SESSION TRACKING
# ═══════════════════════════════════════════════════════════════════════════

class StudyTimer:
    """Track focused work sessions and productivity"""

    def __init__(self):
        self.sessions: List[Dict[str, Any]] = []
        self.current_session: Optional[Dict[str, Any]] = None

    def start_session(self, task: str) -> Dict[str, Any]:
        if self.current_session:
            self.end_session()
        self.current_session = {
            "task": task,
            "start_time": time.time(),
            "start_datetime": datetime.now().isoformat(),
            "end_time": None,
            "duration": 0,
            "status": "active"
        }
        return self.current_session

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
            "active": True,
            "task": self.current_session["task"],
            "elapsed_seconds": int(elapsed),
            "elapsed_formatted": self._format_duration(elapsed),
            "total_today": self._get_today_total() + elapsed
        }

    def _get_today_total(self) -> float:
        today = datetime.now().date()
        return sum(
            s["duration"] for s in self.sessions
            if datetime.fromisoformat(s["start_datetime"]).date() == today
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        hours   = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs    = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def get_stats(self) -> Dict[str, Any]:
        today_total = self._get_today_total()
        if self.current_session:
            today_total += time.time() - self.current_session["start_time"]
        today = datetime.now().date()
        return {
            "total_sessions": len(self.sessions),
            "active_session": self.current_session is not None,
            "today_total_seconds": int(today_total),
            "today_total_formatted": self._format_duration(today_total),
            "sessions_today": sum(
                1 for s in self.sessions
                if datetime.fromisoformat(s["start_datetime"]).date() == today
            )
        }


timer = StudyTimer()

# ═══════════════════════════════════════════════════════════════════════════
# CONVERSATION MEMORY WITH FILE PERSISTENCE (Last 50 Messages)
# ═══════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """
    Conversation history with file persistence.
    Keeps the last 50 messages (25 conversation turns) in a JSON file.
    """

    def __init__(self, max_messages: int = HISTORY_MAX, history_file: str = HISTORY_FILE):
        self.max_messages = max_messages
        self.history_file = history_file
        self.messages: List[Dict[str, str]] = []
        self._load_from_file()

    def _load_from_file(self):
        """Load history from JSON file on startup."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.messages = json.load(f)
                print(f"📜 Loaded {len(self.messages)} messages from history")
            except Exception as e:
                print(f"⚠️ Failed to load history: {e}")
                self.messages = []
        else:
            self.messages = []

    def _save_to_file(self):
        """Persist history to JSON file."""
        try:
            trimmed = self.messages[-self.max_messages:]
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(trimmed, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save history: {e}")

    def add(self, role: str, content: str):
        """Add a message and auto-save."""
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        self._save_to_file()

    def get(self) -> List[Dict[str, str]]:
        """Get copy of current messages."""
        return self.messages.copy()

    def clear(self):
        """Clear history from memory and file."""
        self.messages = []
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
        print("📜 History cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get history statistics."""
        return {
            "total_messages": len(self.messages),
            "max_messages": self.max_messages,
            "file_path": self.history_file,
            "file_exists": os.path.exists(self.history_file)
        }


memory = ConversationMemory()

# ═══════════════════════════════════════════════════════════════════════════
# HELPER: Stream model response
# ═══════════════════════════════════════════════════════════════════════════

async def _stream_model(messages: List[Dict[str, Any]], **kwargs) -> AsyncIterator[str]:
    """
    Internal helper to stream a response from the configured MODEL.
    All model calls go through this single point.

    ollama.chat with stream=True returns a synchronous generator.
    asyncio.to_thread cannot wrap a generator, so we:
      1. Run the blocking call in a thread to get the generator object.
      2. Iterate it on the main thread (yields instantly once TCP is open).
    For true async iteration, use ollama.AsyncClient instead.
    """
    defaults = {"temperature": 1, "num_predict": 2000}
    defaults.update(kwargs)

    loop   = asyncio.get_event_loop()
    stream = await loop.run_in_executor(
        None,
        lambda: ollama.chat(
            model=MODEL,
            messages=messages,
            stream=True,
            options=defaults
        )
    )

    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]

# ═══════════════════════════════════════════════════════════════════════════
# CORE AI ENGINE
# ═══════════════════════════════════════════════════════════════════════════

async def main(
    user_message: str,
    image_path: Optional[str] = None,
    study_context: Optional[str] = None,
    use_rag: bool = True
) -> AsyncIterator[str]:
    """
    Main AI response generator with RAG support.

    Args:
        user_message:  User's input message
        image_path:    Optional path to image for vision tasks
        study_context: Optional study session context
        use_rag:       Whether to use RAG for knowledge retrieval

    Yields:
        Response chunks as they're generated
    """
    context_parts = [BASE_CHARACTER, USER_CONTEXT]

    # Timer context
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

    # RAG: inject book knowledge (skip for image queries)
    if use_rag and rag and not image_path:
        book_context = rag.get_context_for_teaching(user_message, k=20)
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
            "role": "user",
            "content": user_message,
            "images": [image_path]
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

# ═══════════════════════════════════════════════════════════════════════════
# TEACHING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

async def teach_topic(topic: str, depth: str = "standard") -> AsyncIterator[str]:
    """Teach a topic using RAG from books"""
    if rag:
        k = {"quick": 10, "standard": 20, "deep": 30}.get(depth, 20)
        book_context = rag.get_context_for_teaching(topic, k=k)
    else:
        book_context = ""

    depth_instructions = {
        "quick":    "Explain briefly in 2-3 paragraphs.",
        "standard": "Explain thoroughly with examples.",
        "deep":     "Provide comprehensive explanation with theory, examples, and applications."
    }

    system_prompt = (
        f"{BASE_CHARACTER}\n\n"
        f"[TEACHING MODE]\n"
        f"Topic: {topic}\n"
        f"Depth: {depth}\n"
        f"Instruction: {depth_instructions.get(depth, depth_instructions['standard'])}\n\n"
        f"{book_context if book_context else '[No book references found — using general knowledge]'}\n\n"
        "Format your explanation clearly. Use analogies when helpful. "
        "If using book excerpts, cite them: \"According to [Book Name]...\""
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Teach me about {topic}"}
    ]

    try:
        async for chunk in _stream_model(messages):
            yield chunk
    except Exception as e:
        yield f"[ERROR] {str(e)}"

# ═══════════════════════════════════════════════════════════════════════════
# QUIZ GENERATION
# ═══════════════════════════════════════════════════════════════════════════

async def generate_quiz(topic: str, difficulty: str = "medium", num: int = 5) -> dict:
    """
    Generate quiz questions on a topic.

    Args:
        topic: The subject to quiz on
        difficulty: easy, medium, or hard
        num: Number of questions (1-10)

    Returns:
        dict with 'questions' list, 'topic' string, and 'difficulty' string
    """
    # Clamp / validate values
    difficulty = difficulty if difficulty in ["easy", "medium", "hard"] else "medium"
    num = max(1, min(10, num))

    # Get RAG context if available
    book_context = ""
    if rag:
        book_context = rag.get_context_for_teaching(topic, k=15)

    system_prompt = f"""{BASE_CHARACTER}

[QUIZ GENERATION MODE]
You are a quiz generator. Output ONLY raw JSON — no markdown, no code fences, no explanation text before or after.

Topic: {topic}
Difficulty: {difficulty}
Number of questions: {num}

{book_context}

CRITICAL: The EXACT field names below are mandatory. Do not rename them.
Output ONLY this JSON object and nothing else:
{{
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "questions": [
    {{
      "id": 1,
      "question": "Write the full question here",
      "options": ["First full option text", "Second full option text", "Third full option text", "Fourth full option text"],
      "correct": "Exact text of the correct option (must match one of the options exactly)",
      "explanation": "One sentence explaining why the correct answer is right"
    }}
  ]
}}

Rules:
- "question" must be a non-empty string
- "options" must be a list of 4 strings, plain text (no A. B. C. D. prefixes)
- "correct" must be the FULL TEXT of one of the options, not a letter or index
- Generate exactly {num} questions"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate {num} {difficulty} questions about {topic}"}
    ]

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ollama.chat(
                model=MODEL,
                messages=messages,
                stream=False,
                options={"temperature": 0.7, "num_predict": 3000}
            )
        )

        content = response["message"]["content"].strip()

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        elif "```" in content:
            json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

        result = json.loads(content)

        # Validate top-level structure
        if "questions" not in result:
            result["questions"] = []
        if "topic" not in result:
            result["topic"] = topic
        if "difficulty" not in result:
            result["difficulty"] = difficulty

        # ── Normalize each question to a guaranteed field set ──────────────
        # LLMs often ignore exact field names in prompts. Map every known
        # variation to the canonical names the frontend expects:
        #   question, options, correct, explanation
        normalized = []
        for i, q in enumerate(result["questions"]):
            # Question text
            qtext = (
                q.get("question") or
                q.get("q") or
                q.get("text") or
                q.get("prompt") or
                f"Question {i + 1}"
            )
            # Options list
            opts = (
                q.get("options") or
                q.get("choices") or
                q.get("answers") or
                []
            )
            # Correct answer
            correct = (
                q.get("correct") or
                q.get("answer") or
                q.get("correct_answer") or
                q.get("correctAnswer") or
                ""
            )
            # Explanation
            explanation = (
                q.get("explanation") or
                q.get("reason") or
                q.get("rationale") or
                ""
            )
            # If correct is an index (0-3) or letter (A-D), resolve to full option text
            if correct and opts:
                if correct in ("A", "B", "C", "D"):
                    idx = ord(correct) - ord("A")
                    if 0 <= idx < len(opts):
                        correct = opts[idx]
                elif correct.isdigit():
                    idx = int(correct)
                    if 0 <= idx < len(opts):
                        correct = opts[idx]

            normalized.append({
                "id": q.get("id", i + 1),
                "question": qtext,
                "options": opts,
                "correct": correct,
                "explanation": explanation
            })

        result["questions"] = normalized
        print(f"Quiz generated: {len(normalized)} questions for '{topic}'")
        return result

    except json.JSONDecodeError as e:
        print(f"Quiz JSON parse error: {e}\nRaw content: {content[:300]}")
        return {"topic": topic, "difficulty": difficulty, "questions": []}
    except Exception as e:
        print(f"Quiz generation error: {e}")
        return {"topic": topic, "difficulty": difficulty, "questions": []}

# ═══════════════════════════════════════════════════════════════════════════
# STUDY PLAN GENERATION
# ═══════════════════════════════════════════════════════════════════════════

async def create_study_plan(topic: str) -> AsyncIterator[str]:
    """
    Generate a structured study plan for a topic.

    Args:
        topic: What the user wants to learn

    Yields:
        Study plan content as streamed chunks
    """
    book_context = ""
    if rag:
        book_context = rag.get_context_for_teaching(topic, k=15)
        book_note = "\n[BOOKS AVAILABLE] Use the book excerpts above to recommend specific chapters/readings."
    else:
        book_note = ""

    system_prompt = f"""{BASE_CHARACTER}

[STUDY PLAN MODE]
Create a detailed, actionable study plan for learning: {topic}

User profile: Hands-on learner, prefers projects over theory, engineering student
{book_note}

STRUCTURE YOUR RESPONSE AS:
## 📋 Study Plan: [Topic]

### Phase 1: Foundation (X days)
- [ ] Specific task 1
- [ ] Specific task 2
- Resources: specific books/chapters if available

### Phase 2: Core Skills (X days)
- [ ] Specific task 1
- [ ] Specific task 2
- Mini-project idea

### Phase 3: Projects (X days)
- [ ] Project 1: Description
- [ ] Project 2: Description

### Phase 4: Advanced (X days)
- [ ] Advanced topic 1
- [ ] Advanced topic 2

### 📚 Recommended Resources
- Specific books with chapter numbers
- Online resources
- Practice platforms

### ⚡ Quick Win (Day 1)
One thing they can build TODAY to get started

Be specific. Give concrete tasks, not vague goals. Include time estimates.
{book_context}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Create a study plan for: {topic}"}
    ]

    try:
        async for chunk in _stream_model(messages, num_predict=2500):
            yield chunk
    except Exception as e:
        yield f"[ERROR] Study plan generation failed: {str(e)}"

# ═══════════════════════════════════════════════════════════════════════════
# CODE REVIEW
# ═══════════════════════════════════════════════════════════════════════════

async def review_code(code: str, language: str = "auto") -> AsyncIterator[str]:
    """
    Review code and provide structured feedback.

    Args:
        code: The code to review
        language: Programming language (auto-detect if "auto")

    Yields:
        Code review content as streamed chunks
    """
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
- **Suggestion**: Nice to have (style, best practices)

### 📝 Specific Changes
Show exact line-level fixes with before/after examples where useful.

### ✨ Revised Version (if changes are significant)
Provide a clean corrected version of the code.

Be direct. Prioritize correctness and clarity."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Review this code:\n\n```\n{code}\n```"}
    ]

    try:
        async for chunk in _stream_model(messages, num_predict=3000):
            yield chunk
    except Exception as e:
        yield f"[ERROR] Code review failed: {str(e)}"

# ═══════════════════════════════════════════════════════════════════════════
# ERROR EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════

async def explain_error(error: str, code: str = "") -> AsyncIterator[str]:
    """
    Analyze and explain an error message with fix suggestions.

    Args:
        error: The error message or traceback
        code: Optional code that produced the error

    Yields:
        Error explanation as streamed chunks
    """
    code_section = f"\n\nCode that caused it:\n```\n{code}\n```" if code else ""

    system_prompt = f"""{BASE_CHARACTER}

[ERROR ANALYSIS MODE]
Diagnose the error and provide a clear fix.

STRUCTURE:
## 🐛 Error Analysis

### What Happened
Plain-language explanation of the error

### Root Cause
The exact technical reason

### Fix
Concrete code fix with explanation

### Prevention
How to avoid this class of error in the future

Be concise. Show the fix, not just theory."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Explain this error and how to fix it:\n\n```\n{error}\n```{code_section}"}
    ]

    try:
        async for chunk in _stream_model(messages, num_predict=2000):
            yield chunk
    except Exception as e:
        yield f"[ERROR] Error explanation failed: {str(e)}"

# ═══════════════════════════════════════════════════════════════════════════
# TIMER COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "main",
    "teach_topic",
    "generate_quiz",
    "create_study_plan",
    "review_code",
    "explain_error",
    "handle_timer_command",
    "format_study_context",
    "timer",
    "memory",
    "rag",
    "StudyTimer",
    "ConversationMemory",
    "BookRAG"
]