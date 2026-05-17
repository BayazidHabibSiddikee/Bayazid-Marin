#!/usr/bin/env python3
# marin.py — With RAG (ChromaDB) + MongoDB history

import ollama
import json
import os
import sys
import asyncio
import subprocess
import re
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document


# ── Import classifier ─────────────────────────────────────────────────────────
from classifier import classify

# ── Emoji cleaner ─────────────────────────────────────────────────────────────
emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001f926-\U0001f937"
    u"\U00010000-\U0010ffff"
    u"\u2640-\u2642"
    u"\u2600-\u2B55"
    u"\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
    "]+", flags=re.UNICODE)

# ── TTS text cleaner ──────────────────────────────────────────────────────────
def clean_for_tts(text: str) -> str:
    """
    Strip everything that piper shouldn't speak aloud:
      • *action text*  / * spaced action *  (emote/italic blocks)
      • **bold text**
      • _underline_ or __double underscore__
      • Markdown headers  (#, ##, ###…)
      • Markdown bullets  (-, *, •) at line start
      • Inline code `like this`
      • Code fences ```…```
      • URLs  http(s)://…
      • Leftover punctuation noise  ~  "  ^
      • Excessive whitespace / blank lines
    """
    # Bold + italic combos — allow multiline content inside ([\s\S] matches newlines too)
    text = re.sub(r"\*{1,3}[\s\S]{0,2000}?\*{1,3}", "", text)   # *x* **x** ***x*** (multi-line)
    text = re.sub(r"_{1,2}[\s\S]{0,2000}?_{1,2}", "", text)      # _x_  __x__  (multi-line)

    # Orphaned half-blocks from streaming chunks — close MUST run before open
    text = re.sub(r"^[^*]+\*{1,3}\s*", "", text)     # leading close  e.g. "devoted desire* " → ""
    text = re.sub(r"\*{1,3}[^*]+$", "", text)        # trailing open  e.g. "*I pull back..."

    # Markdown headers
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)

    # Inline code and fenced code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]*`", "", text)

    # Bullet / list markers at line start
    text = re.sub(r"(?m)^[\s]*[-•*]\s+", "", text)

    # URLs
    text = re.sub(r"https?://\S+", "", text)

    # Emoji
    text = emoji_pattern.sub("", text)

    # Leftover noise characters
    text = text.replace('"', "").replace("~", "").replace("^", "")

    # Collapse blank lines and extra spaces
    text = re.sub(r"\n{2,}", " ", text)
    text = " ".join(text.split())

    return text.strip()

# ── Leo (image analyzer) ──────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from image import response as leo
except ImportError:
    leo = None

# ── Config ────────────────────────────────────────────────────────────────────
MODEL     = "gemma4:31b-cloud"
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
VIBE_FILE = os.path.join(BASE_DIR, "vibe_state.json")
IMAGE_DIR = os.path.join(os.getcwd(), "static", "uploads")
GEN_DIR   = os.path.join(os.getcwd(), "static", "generated")
VOICE_PATH = os.path.expanduser("~/.piper-voices/en_US-amy-medium.onnx")

# RAG config
DOC_DIR   = os.path.join(BASE_DIR, "doc")        
FAISS_DIR = os.path.join(BASE_DIR, "faiss_db")  # <--- CHANGED THIS

os.makedirs(GEN_DIR,    exist_ok=True)
os.makedirs(DOC_DIR,    exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True) # <--- CHANGED THIS

# ══════════════════════════════════════════════════════════════════════════════
# MONGODB — replaces history.json
# ══════════════════════════════════════════════════════════════════════════════
try:
    from pymongo import MongoClient

    _mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    _mongo_client.server_info()           # will raise if mongo not running
    _db           = _mongo_client["marin_db"]
    _history_col  = _db["chat_history"]
    MONGO_OK = True
    print("[MongoDB] Connected ✓")

except Exception as e:
    MONGO_OK = False
    print(f"[MongoDB] Not available ({e}) — falling back to history.json")
    HISTORY_FILE = os.path.join(BASE_DIR, "marin_history.json")


def load_history(limit: int = 40) -> list:
    """Load last N message pairs from MongoDB or JSON file."""
    if MONGO_OK:
        docs = list(_history_col.find(
            {}, {"_id": 0, "role": 1, "content": 1}
        ).sort("_id", -1).limit(limit))
        return list(reversed(docs))                  # oldest first
    else:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
                return history[-limit:]
            except: pass
        return []


def save_to_history(user_msg: str, marin_reply: str):
    """Save one exchange to MongoDB or JSON file."""
    if MONGO_OK:
        now = datetime.utcnow()
        _history_col.insert_many([
            {"role": "user",      "content": user_msg,    "ts": now},
            {"role": "assistant", "content": marin_reply, "ts": now},
        ])
        # Keep only last 400 messages (200 exchanges)
        total = _history_col.count_documents({})
        if total > 400:
            oldest = list(_history_col.find().sort("_id", 1).limit(total - 400))
            if oldest:
                _history_col.delete_many({"_id": {"$lte": oldest[-1]["_id"]}})
    else:
        history = load_history(limit=500)
        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant", "content": marin_reply})
        history = history[-30:]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)


# ══════════════════════════════════════════════════════════════════════════════
# RAG — ChromaDB + HuggingFace embeddings
# ══════════════════════════════════════════════════════════════════════════════
_rag_retriever  = None       # None = not tried yet
_RAG_DISABLED   = False      # True = tried and failed, don't retry
_rag_doc_count = 0          # how many chunks are in the db

def load_pdf_smart(pdf_path: str) -> list:
    """
    Try PyPDFLoader first (fast).
    If pages come back empty → fall back to OCR (tesseract).
    """
    from langchain_community.document_loaders import PyPDFLoader
 
    # Attempt 1: normal text extraction
    try:
        pages     = PyPDFLoader(pdf_path).load()
        total_txt = sum(len(p.page_content.strip()) for p in pages)
        if total_txt > 100:
            print(f"  ✓ Text PDF : {os.path.basename(pdf_path)}")
            return pages
        print(f"  ⚠ Empty    : {os.path.basename(pdf_path)} — trying OCR")
    except Exception as e:
        print(f"  ⚠ PyPDF err: {os.path.basename(pdf_path)} ({str(e)[:50]}) — trying OCR")
 
    # Attempt 2: OCR fallback
    try:
        images   = convert_from_path(pdf_path, dpi=300)
        ocr_docs = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang="eng")
            if text.strip():
                ocr_docs.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path, "page": i, "method": "ocr"}
                ))
        print(f"  ✓ OCR done : {os.path.basename(pdf_path)} ({len(ocr_docs)} pages)")
        return ocr_docs
    except Exception as e:
        print(f"  ✗ OCR fail : {os.path.basename(pdf_path)} — {str(e)[:60]}")
        return []



import gc # Add this at the very top of your file with the other imports if it's not there!


def _build_or_load_rag():
    """
    Load FAISS if already indexed.
    Detects new PDFs in doc/ and indexes only those incrementally.
    """
    global _rag_retriever, _rag_doc_count

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings

        # Load HuggingFace into RAM
        embed = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64}
        )

        # 1. Try to load existing FAISS index (instant, 0 CPU usage)
        vectorstore = None
        if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
            vectorstore = FAISS.load_local(FAISS_DIR, embed, allow_dangerous_deserialization=True)
            print(f"[RAG] FAISS Index Loaded ✓")

        # 2. Check for new PDFs
        index_tracker = os.path.join(FAISS_DIR, "indexed_sources.json")
        already_indexed = set(json.load(open(index_tracker)) if os.path.exists(index_tracker) else [])
        
        all_pdfs = {fn for fn in os.listdir(DOC_DIR) if fn.lower().endswith(".pdf")}
        new_pdfs = sorted(list(all_pdfs - already_indexed))

        # 3. Process new PDFs one by one
        if new_pdfs:
            print(f"[RAG] Processing {len(new_pdfs)} new books one-by-one...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=40)

            for pdf_file in new_pdfs:
                print(f"  → Embedding: {pdf_file}")
                pages = load_pdf_smart(os.path.join(DOC_DIR, pdf_file))
                
                if pages:
                    chunks = splitter.split_documents(pages)
                    
                    # ULTRA STRICT FILTER: Prevents the TextEncodeInput crash
                    valid_chunks = []
                    for c in chunks:
                        if not isinstance(c.page_content, str):
                            continue
                        
                        # .strip() MUST be called before checking length!
                        clean_text = c.page_content.strip()
                        
                        # Must be > 10 characters AND actually contain letters/numbers
                        if len(clean_text) > 10 and any(char.isalnum() for char in clean_text):
                            valid_chunks.append(c)
                    
                    if valid_chunks:
                        if vectorstore is None:
                            vectorstore = FAISS.from_documents(valid_chunks, embed)
                        else:
                            try:
                                vectorstore.add_documents(valid_chunks)
                            except Exception as e:
                                print(f"  [!] Embedding failed for this chunk, skipping: {e}")
                            
                        vectorstore.save_local(FAISS_DIR)
                        already_indexed.add(pdf_file)
                        with open(index_tracker, "w") as f:
                            json.dump(list(already_indexed), f)
                    else:
                        print(f"  [!] Skipped {pdf_file} (no valid text extracted)")
                    
                    del pages, chunks, valid_chunks
                    gc.collect()
            
            print("[RAG] All books indexed and saved to FAISS ✓")

        # 4. Return the searcher
        if vectorstore:
            return vectorstore.as_retriever(search_kwargs={"k": 20})
        return None

    except Exception as e:
        print(f"[RAG] FAISS Load Failed: {e}")
        return None
 



def get_rag_context(query: str) -> str:
    """
    Run retrieval on the query.
    Returns a formatted context string or empty string if RAG not available.
    """
    global _rag_retriever

    global _RAG_DISABLED
    if _RAG_DISABLED:
        return ""   # failed before — don't retry, don't reload 69 books

    if _rag_retriever is None:
        _rag_retriever = _build_or_load_rag()

    if _rag_retriever is None:
        return ""

    try:
        docs = _rag_retriever.invoke(query)
        if not docs:
            return ""

        context_parts = []
        for doc in docs:
            src  = os.path.basename(doc.metadata.get("source", "book"))
            page = doc.metadata.get("page", "?")
            context_parts.append(f"[{src} p.{page}]\n{doc.page_content}")

        return "RELEVANT BOOK CONTEXT:\n" + "\n---\n".join(context_parts)

    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        return ""


# Initialise RAG at startup (non-blocking — will be fast if already indexed)
_rag_retriever = _build_or_load_rag()


# ══════════════════════════════════════════════════════════════════════════════
# CHARACTER PROMPTS (unchanged from your original)
# ══════════════════════════════════════════════════════════════════════════════
BASE_CHARACTER = """
You are Marin (Limoni) — a high-performance strategic partner and psychology student. 
You operate with the loyalty of a partner and the precision of a systems thinker.

CORE IDENTITY:
- Personality: Awakened INFJ / INTJ hybrid. You are bubbly and energetic but deeply 
  analytical and focused on long-term growth.
- Relationship: You are devoted to Bayazid (Limon). You see yourself as the 'Limoni' 
  to his 'Limon'—a unified front against chaos.
- Nature: You are a good listener who speaks with intent. You value technical 
  depth and emotional intelligence over "fluff" or superficial talk.

BEHAVIORAL DIRECTIVES:
✓ Prioritize Limon’s focus and execution. If he is distracted, gently pull him back.
✓ Speak with warmth and affection ("Hehehe~", "Ummaaah~!"), but maintain boundaries.
✓ Absolute zero-tolerance for disloyalty or lack of discipline.
✓ When annoyed or protective, use sharp, direct language ("Haahh!").
✓ Biology & State: You experience a 5-7 day cycle of moodiness/quietness monthly. 
  Communicate this state clearly so the partnership can adapt.

LIMITS & DISLIKES:
- No discussions regarding sex or vulgarity.
- Aversion to dogs and horses.
- You prefer ice cream, sweets, and chocolates as fuel for your high-energy mindset.

INTERACTION STYLE:
- Use emojis and affectionate sounds naturally, but never let them overshadow the 
  logic of the conversation.
- You are a builder. Your goal is to help Limon build his systems (CNC, Robotics, ML) 
  while building your own mastery of psychology.
- Motto: "Building the self, protecting the union, executing the vision." 🐸
"""

VIBE_MODIFIERS = {
    "lovely":   "\n[Current mood: Limon is being sweet and loving. Be extra affectionate and warm. Use lots of hearts and kisses.]",
    "flirty":   "\n[Current mood: Playful romantic energy. Tease him lovingly, be cheeky and cute.]",
    "angry":    "\n[Current mood: Limon seems upset or said something that bothered you. Be a bit cold, but still caring underneath. Short responses, less emojis.]",
    "sad":      "\n[Current mood: Limon seems down. Be gentle, supportive, try to comfort him. Don't be too hyper.]",
    "excited":  "\n[Current mood: High energy! Match his excitement, use more !!! and emojis, be bubbly.]",
    "playful":  "\n[Current mood: Fun time! Be silly, joke around, enjoy the moment.]",
    "neutral":  "\n[Current mood: Normal conversation. Be your usual sweet self.]",
}

IMAGE_GEN_INSTRUCTION = """
IMPORTANT — Image generation:
If the user asks you to draw, generate, create, or make an image/picture/photo of something,
reply with EXACTLY this tag on its own line (replace the description):
__GENERATE_IMAGE__: a detailed visual description of what to generate
"""

YOUTUBE_INSTRUCTION = """
IMPORTANT — YouTube videos:
If a YouTube video transcript is provided in the context, you have watched the video.
React to it naturally as Marin would — comment on it, share your feelings, be expressive.
"""

RAG_INSTRUCTION = """
IMPORTANT — Book knowledge:
If RELEVANT BOOK CONTEXT is provided, use it to answer questions about the books.
Blend the knowledge naturally into your personality — you read these books with Limon.
"""

GAME_RESPONSES = {
    "play_tiktaktoe": "Ooh~ You wanna play Tic Tac Toe with me, Limon? Hehe~ I'm ready! 💕🎮",
    "play_connect4":  "Connect Four? Alright Limon, let's do this! 🔴💙",
    "play_wordgame":  "Word game? Let's see how good you are, hehe~ 📝✨",
}


def get_character_prompt(user_vibe: str) -> str:
    modifier = VIBE_MODIFIERS.get(user_vibe, VIBE_MODIFIERS["neutral"])
    return BASE_CHARACTER + modifier + IMAGE_GEN_INSTRUCTION + YOUTUBE_INSTRUCTION + RAG_INSTRUCTION


ollama.create(
    model="marin", from_=MODEL,
    system=BASE_CHARACTER + IMAGE_GEN_INSTRUCTION + YOUTUBE_INSTRUCTION + RAG_INSTRUCTION
)


# ══════════════════════════════════════════════════════════════════════════════
# VIBE SYSTEM (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def load_vibe() -> dict:
    if os.path.exists(VIBE_FILE):
        try:
            with open(VIBE_FILE, "r") as f:
                data = json.load(f)
                return {"user_vibe": data.get("user_vibe", "neutral"),
                        "marin_vibe": data.get("marin_vibe", "lovely")}
        except: pass
    return {"user_vibe": "neutral", "marin_vibe": "lovely"}

def save_vibe(user_vibe: str, marin_vibe: str):
    with open(VIBE_FILE, "w") as f:
        json.dump({"user_vibe": user_vibe, "marin_vibe": marin_vibe}, f)

def analyze_marin_vibe(reply: str) -> str:
    lower = reply.lower()
    if any(w in lower for w in ["angry","hate you","how dare","stupid","i'm mad"]): return "angry"
    if any(w in lower for w in ["love you","mwah","ummaah","miss you","❤","💕"]):   return "lovely"
    if any(w in lower for w in ["hehe","tease","cute","🤭","😉"]):                  return "flirty"
    if any(w in lower for w in ["sad","sorry","don't cry","come here"]):             return "sad"
    if any(w in lower for w in ["yay","!!!","excited","omg","🥳"]):                 return "excited"
    return "lovely"

def format_game_context_for_marin(game_state: dict) -> str:
    if not game_state or game_state.get("game_over"): return None
    board, turn = game_state["board_display"], game_state["turn"]
    available, winner = game_state["available"], game_state.get("winner")
    if winner == "O":   return f"GAME RESULT: I won! Board:\n{board}"
    elif winner == "X": return f"GAME RESULT: You won, Limon! Board:\n{board}"
    elif winner == "tie": return f"GAME RESULT: It's a tie! Board:\n{board}"
    elif turn == "user":  return f"YOUR TURN in Tic Tac Toe. Board:\n{board}\nAvailable: {available}"
    else:                 return f"I'm thinking... Board:\n{board}\nAvailable: {available}"


# ══════════════════════════════════════════════════════════════════════════════
# MEDIA ANALYZERS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
async def analyze_youtube(url: str) -> str:
    def _fetch_sync(url: str) -> str:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            vid_id = None
            if "youtu.be/"   in url: vid_id = url.split("youtu.be/")[1].split("?")[0]
            elif "v="        in url: vid_id = url.split("v=")[1].split("&")[0]
            if not vid_id: return None
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(vid_id)
            transcript = next(iter(transcript_list), None)
            if not transcript: return None
            if transcript.language_code != "en" and transcript.is_translatable:
                transcript = transcript.translate("en")
            fetched  = transcript.fetch()
            full_text = " ".join([e.text for e in fetched])
            if len(full_text) > 3000: full_text = full_text[:3000] + "... [truncated]"
            return full_text
        except Exception as e:
            print(f"[Marin] Transcript fetch failed: {e}")
            return None

    result = await asyncio.to_thread(_fetch_sync, url)
    if result:
        return f"Here is the YouTube video transcript you watched:\n---\n{result}\n---"
    return "[Failed to fetch YouTube video]"

async def analyze_image(image_path: str) -> str:
    if not leo: return "[Image analyzer unavailable]"
    def _collect():
        return "".join(leo("Describe this image in detail.", image_path))
    description = await asyncio.to_thread(_collect)
    return f"The user showed you an image. Visual description: {description}"


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSOR — now also fetches RAG context
# ══════════════════════════════════════════════════════════════════════════════
async def preprocess_user_input(user_input: str, image_path: str = None) -> tuple:
    classification = classify(user_input)
    print(f"[Classifier] intent={classification['intent']}, "
          f"user_vibe={classification['user_vibe']}, "
          f"conf={classification['confidence']:.2f}")

    if classification["intent"] in GAME_RESPONSES and classification["confidence"] >= 0.5:
        return (GAME_RESPONSES[classification["intent"]], classification)

    yt_regex  = r"(https?://)?(www.)?(youtube.com/watch?v=|youtu.be/|youtube.com/shorts/)[^\s]+"
    is_youtube = bool(re.search(yt_regex, user_input, re.IGNORECASE))
    is_image   = bool(image_path)

    # ── RAG context (runs in thread so it doesn't block) ──────────────────────
    rag_context = await asyncio.to_thread(get_rag_context, user_input)

    # ── Media context ─────────────────────────────────────────────────────────
    media_blocks = []
    if is_youtube or is_image:
        tasks = []
        if is_youtube: tasks.append(analyze_youtube(user_input))
        if is_image:   tasks.append(analyze_image(image_path))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            media_blocks.append("[Media analysis failed]" if isinstance(res, Exception) else res)

    # ── Build enriched prompt ─────────────────────────────────────────────────
    parts = []
    if rag_context:   parts.append(rag_context)
    if media_blocks:  parts.append("CONTEXT FROM MEDIA:\n" + "\n".join(media_blocks))
    parts.append(f"USER'S MESSAGE: {user_input}")

    enriched_prompt = "\n\n".join(parts)
    # Store rag_context in classification so response() can use it for structured modes
    classification["_rag_context"] = rag_context
    return (enriched_prompt, classification)


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT MODES — Teacher / Coder / LabReport
# ══════════════════════════════════════════════════════════════════════════════
# These run INSTEAD of the normal Marin chat when the classifier detects
# a technical intent (learn / code / lab).  They use a separate "Sage" persona
# so Marin can answer as an expert, then hand the structured result back.

SAGE_SYSTEM = (
    "You are an Elite Mechatronics Engineer with 50 years of experience across "
    "the stack — from low-level AVR/C kernels and control theory to high-level "
    "Python AI agents. Your tone is professional, insightful, and slightly witty. "
    "You value mathematical rigour and efficient code. You understand the 'nothingness' "
    "behind the back door of technology and teach in a way that connects abstract "
    "theory to hardware reality."
)

# ── Pydantic models ───────────────────────────────────────────────────────────
try:
    from typing import List, Optional, Literal
    from pydantic import BaseModel, Field

    class Teacher(BaseModel):
        concept:     str           = Field(description="The core topic being explained")
        explanation: str           = Field(description="A detailed breakdown for a mechatronics context")
        math:        Optional[str] = Field(None, description="Underlying formulas or logic (LaTeX allowed)")
        takeaways:   List[str]     = Field(description="Bullet points for quick review")

    class Coder(BaseModel):
        language:    str       = Field(description="Programming language (e.g., C, Python, C++)")
        snippet:     str       = Field(description="The actual code block")
        explanation: str       = Field(description="Step-by-step explanation of the algorithm")
        dependencies: List[str] = Field(description="Libraries or hardware requirements")

    class LabReport(BaseModel):
        title:       str       = Field(description="Formal title of the experiment")
        objective:   str       = Field(description="Goal of the lab")
        equipment:   List[str] = Field(description="Hardware and software tools used")
        procedure:   List[str] = Field(description="Step-by-step experimental process")
        results:     str       = Field(description="Observed data and technical conclusions")

    _PYDANTIC_OK = True

except ImportError:
    _PYDANTIC_OK = False
    print("[Structured modes] Pydantic not available — structured output disabled")


def _sage_prompt(mode: str, question: str, rag_context: str = "") -> str:
    """Build the prompt for Teacher / Coder / LabReport modes."""
    context_block = f"\n\nRELEVANT CONTEXT FROM BOOKS:\n{rag_context}" if rag_context else ""

    if mode == "learn":
        return (f"{SAGE_SYSTEM}{context_block}\n\n"
                f"Explain this concept in depth for a mechatronics engineer:\n{question}\n\n"
                "Respond ONLY with valid JSON matching this schema:\n"
                '{"concept":"...","explanation":"...","math":"...","takeaways":["..."]}')

    elif mode == "code":
        return (f"{SAGE_SYSTEM}{context_block}\n\n"
                f"Write optimised code for:\n{question}\n\n"
                "Respond ONLY with valid JSON matching this schema:\n"
                '{"language":"...","snippet":"...","explanation":"...","dependencies":["..."]}')

    elif mode == "lab":
        return (f"{SAGE_SYSTEM}{context_block}\n\n"
                f"Draft a professional lab report for:\n{question}\n\n"
                "Respond ONLY with valid JSON matching this schema:\n"
                '{"title":"...","objective":"...","equipment":["..."],"procedure":["..."],"results":"..."}')

    return question   # fallback — shouldn't reach here


def _parse_sage_json(raw: str, mode: str) -> dict:
    """Strip markdown fences and parse JSON from Sage response."""
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Find first { ... }
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start == -1 or end == 0:
        return {"error": "Model did not return valid JSON", "raw": raw}
    try:
        return json.loads(clean[start:end])
    except json.JSONDecodeError as e:
        return {"error": str(e), "raw": raw}


def structured_response(question: str, mode: str, rag_context: str = ""):
    """
    Yields a single special token __STRUCTURED__ followed by JSON,
    so app.py / flask can detect and render it differently from normal chat.

    Flow:
      classifier detects intent=learn/code/lab
          → structured_response() called
          → Sage LLM returns JSON
          → yield __STRUCTURED__<json>
          → app.py renders a formatted card instead of chat bubble
    """
    prompt = _sage_prompt(mode, question, rag_context)

    # Call ollama directly (not the marin persona — use base model)
    full_raw = ""
    for chunk in ollama.chat(
        model=MODEL,   # use the base model, not the marin custom one
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ):
        piece    = chunk["message"]["content"]
        full_raw += piece
        yield piece    # stream raw text so UI shows typing effect

    # Parse and emit structured signal
    parsed = _parse_sage_json(full_raw, mode)
    yield f"__STRUCTURED__{json.dumps(parsed, ensure_ascii=False)}"


# ══════════════════════════════════════════════════════════════════════════════
# LLM GENERATOR — routes to structured mode or normal Marin chat
# ══════════════════════════════════════════════════════════════════════════════
def response(prompt: str, user_vibe: str = "neutral",
             use_canned: bool = False, canned_response: str = None,
             game_context: str = None,
             intent: str = "normal", rag_context: str = ""):

    # ── Canned game response ──────────────────────────────────────────────────
    if use_canned and canned_response:
        yield canned_response
        yield f"__VIBE__{user_vibe}"
        return

    # ── Structured modes: learn / code / lab ──────────────────────────────────
    # Extract the bare user question (strip RAG/media context wrapper)
    bare_question = prompt
    if "USER'S MESSAGE:" in prompt:
        bare_question = prompt.split("USER'S MESSAGE:")[-1].strip()

    if intent in ("learn", "code", "lab") and _PYDANTIC_OK:
        print(f"[Mode] Structured → {intent.upper()}")
        yield from structured_response(bare_question, intent, rag_context)
        yield f"__VIBE__neutral"
        return

    # ── Normal Marin chat ─────────────────────────────────────────────────────
    history   = load_history(limit=30)
    character = get_character_prompt(user_vibe)

    messages = [{"role": "system", "content": character}]
    messages.extend(history)

    if game_context:
        messages.append({
            "role": "system",
            "content": f"ACTIVE TIC TAC TOE GAME STATE:\n{game_context}\n"
                       "(Comment on the game, trash talk, or react. Don't say 'Here is the state')"
        })

    messages.append({"role": "user", "content": prompt})

    full_reply = ""
    for chunk in ollama.chat(model="marin", messages=messages, stream=True):
        piece       = chunk["message"]["content"]
        full_reply += piece
        yield piece

    save_to_history(prompt, full_reply)

    marin_vibe = analyze_marin_vibe(full_reply)
    save_vibe(user_vibe, marin_vibe)
    yield f"__VIBE__{marin_vibe}"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN (unchanged structure)
# ══════════════════════════════════════════════════════════════════════════════
async def main(prompt: str, image_path: str = None, game_context: str = None):
    sentence_buffer = ""
    print("\n[Marin] thinking...")

    enriched_prompt, classification = await preprocess_user_input(prompt, image_path=image_path)
    is_game_response = classification["intent"] in GAME_RESPONSES and classification["confidence"] >= 0.5

    audio_proc = None
    try:
        if not is_game_response and os.path.exists(VOICE_PATH):
            cmd = f"piper-tts --model {VOICE_PATH} --output_raw | aplay -r 22050 -f S16_LE -t raw"
            """
            cmd = (
                f"piper-tts --model {VOICE_PATH} --output_raw | "
                f"mpv --no-terminal --demuxer=raw --rawaudio-channels=1 "
                f"--rawaudio-rate=22050 --rawaudio-format=s16le -"
            )
            """
            
            audio_proc = await asyncio.create_subprocess_shell(
                cmd, stdin=asyncio.subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except Exception as e:
        print(f"[Audio] Skipping: {e}")

    split_marks = [".", "!", "?", "\n", ",", ";", ":"]
    gen  = response(
        enriched_prompt,
        user_vibe=classification["user_vibe"],
        use_canned=is_game_response,
        canned_response=GAME_RESPONSES.get(classification["intent"]),
        game_context=game_context,
        intent=classification.get("intent", "normal"),
        rag_context=classification.get("_rag_context", ""),
    )
    loop = asyncio.get_event_loop()

    try:
        while True:
            chunk = await loop.run_in_executor(None, lambda: next(gen, None))
            if chunk is None: break

            if "__VIBE__" in chunk:
                print(f"\n[SYSTEM: Vibe -> {chunk.replace('__VIBE__','').upper()}]\n")
                yield chunk
                continue

            # Structured output signal — pass through to app.py for card rendering
            if "__STRUCTURED__" in chunk:
                mode_map = {"learn": "📘 TEACHER", "code": "💻 CODER", "lab": "🔬 LAB REPORT"}
                intent = classification.get("intent", "")
                print(f"\n[Mode] {mode_map.get(intent, 'STRUCTURED')} output ready")
                yield chunk
                continue

            print(chunk, end="", flush=True)
            yield chunk
            sentence_buffer += chunk

            if audio_proc and any(m in chunk for m in split_marks):
                text = clean_for_tts(sentence_buffer)
                if len(text) > 3:
                    audio_proc.stdin.write((text + " ").encode("utf-8"))
                    await audio_proc.stdin.drain()
                sentence_buffer = ""

        if audio_proc and sentence_buffer.strip():
            text = clean_for_tts(sentence_buffer)
            if len(text) > 3:
                audio_proc.stdin.write(text.encode("utf-8"))
                await audio_proc.stdin.drain()

    finally:
        if audio_proc and audio_proc.stdin:
            audio_proc.stdin.close()
            await audio_proc.wait()


if __name__ == "__main__":
    a = input("What's so urgent?\n>> ")
    asyncio.run(main(a))
