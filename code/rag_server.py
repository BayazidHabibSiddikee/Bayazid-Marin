# rag_server.py — Shared RAG server (port 5080)
#
# Supports two knowledge bases:
#   doc/   → books, documents  (PDF, DOCX, TXT, MD)
#   code/  → your source files (PY, C, CPP, H, MD)
#
# Both indexed into ONE FAISS index — source_type metadata lets you filter.
# File upload endpoints let Marin/Bayazid frontends accept files directly.
#
# pip install docx2txt   (for .docx support)

import asyncio
import gc
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ RAG dependencies not available")

try:
    import docx2txt
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️ docx2txt not installed — .docx skipped. Run: pip install docx2txt")


# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DOC_DIR   = Path(BASE_DIR) / "doc"
CODE_DIR  = Path(BASE_DIR) / "code"
FAISS_DIR = Path(BASE_DIR) / "faiss_db"

DOC_DIR.mkdir(exist_ok=True)
CODE_DIR.mkdir(exist_ok=True)
FAISS_DIR.mkdir(exist_ok=True)

DOC_EXTENSIONS  = {".pdf", ".docx", ".txt", ".md"}
CODE_EXTENSIONS = {".py", ".c", ".cpp", ".h", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════
class KnowledgeBase:
    """
    Unified FAISS index over doc/ and code/.
    Chunk metadata: source_file, source_type (doc|code), language, page.
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MANIFEST_PATH   = FAISS_DIR / "manifest.json"

    # Prose — larger chunks, less overlap
    DOC_CHUNK_SIZE  = 450
    DOC_OVERLAP     = 40
    # Code — smaller chunks, more overlap (keep functions readable)
    CODE_CHUNK_SIZE = 300
    CODE_OVERLAP    = 50

    def __init__(self):
        self.vectorstore: Optional[FAISS] = None
        self.embeddings  = None
        self.manifest: Dict[str, Any] = {"indexed": [], "failed": []}
        self._boot()

    # ── Startup ───────────────────────────────────────────────────────────────
    def _boot(self):
        if not FAISS_AVAILABLE:
            print("⚠️ FAISS not available — RAG disabled")
            return

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64},
        )
        self._load_manifest()

        index_path = FAISS_DIR / "index.faiss"
        if index_path.exists():
            try:
                self.vectorstore = FAISS.load_local(
                    str(FAISS_DIR), self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                n = len(self.manifest["indexed"])
                print(f"✅ Knowledge base loaded: {n} files indexed")
            except Exception as e:
                print(f"⚠️ Failed to load index ({e}) — rebuilding")
                self.vectorstore = None
                self.manifest    = {"indexed": [], "failed": []}

        self._index_new_files()

    # ── Manifest ──────────────────────────────────────────────────────────────
    def _load_manifest(self):
        if self.MANIFEST_PATH.exists():
            try:
                with open(self.MANIFEST_PATH) as f:
                    self.manifest = json.load(f)
                self.manifest.setdefault("indexed", [])
                self.manifest.setdefault("failed",  [])
            except Exception:
                self.manifest = {"indexed": [], "failed": []}

    def _save_manifest(self):
        with open(self.MANIFEST_PATH, "w") as f:
            json.dump(self.manifest, f, indent=2)

    # ── File discovery ────────────────────────────────────────────────────────
    def _all_files(self) -> List[Path]:
        files = []
        for ext in DOC_EXTENSIONS:
            files.extend(DOC_DIR.glob(f"*{ext}"))
        for ext in CODE_EXTENSIONS:
            files.extend(CODE_DIR.glob(f"*{ext}"))
        return sorted(set(files))

    def _index_new_files(self):
        already_indexed = set(self.manifest["indexed"])
        already_failed  = {e["file"] for e in self.manifest["failed"]}
        new_files = [
            f for f in self._all_files()
            if f.name not in already_indexed and f.name not in already_failed
        ]
        if not new_files:
            return
        print(f"📚 Indexing {len(new_files)} new file(s)...")
        for path in new_files:
            self._index_single_file(path)
        if self.vectorstore:
            self.vectorstore.save_local(str(FAISS_DIR))
        self._save_manifest()
        print(f"✅ Done: {len(self.manifest['indexed'])} total indexed")

    # ── Loaders ───────────────────────────────────────────────────────────────
    def _load_file(self, path: Path) -> List[Document]:
        ext         = path.suffix.lower()
        name        = path.name
        source_type = "code" if path.parent.resolve() == CODE_DIR.resolve() else "doc"

        if ext == ".pdf":
            docs = PyPDFLoader(str(path)).load()
            for d in docs:
                d.metadata.update({"source_file": name, "source_type": "doc", "language": "text"})
            return docs

        if ext == ".docx":
            if not DOCX_AVAILABLE:
                raise ImportError("docx2txt not installed — run: pip install docx2txt")
            text = docx2txt.process(str(path))
            return [Document(page_content=text,
                             metadata={"source_file": name, "source_type": "doc",
                                       "language": "text", "page": 0})]

        if ext == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            return [Document(page_content=text,
                             metadata={"source_file": name, "source_type": source_type,
                                       "language": "text", "page": 0})]

        if ext == ".md":
            text = path.read_text(encoding="utf-8", errors="ignore")
            return [Document(page_content=text,
                             metadata={"source_file": name, "source_type": source_type,
                                       "language": "markdown", "page": 0})]

        if ext == ".py":
            text = path.read_text(encoding="utf-8", errors="ignore")
            return [Document(page_content=text,
                             metadata={"source_file": name, "source_type": "code",
                                       "language": "python", "page": 0})]

        if ext in {".c", ".cpp", ".h"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            lang = {"c": "c", ".cpp": "cpp", ".h": "c"}.get(ext, "c")
            return [Document(page_content=text,
                             metadata={"source_file": name, "source_type": "code",
                                       "language": lang, "page": 0})]

        raise ValueError(f"Unsupported extension: {ext}")

    def _get_splitter(self, path: Path) -> RecursiveCharacterTextSplitter:
        if path.suffix.lower() in {".py", ".c", ".cpp", ".h"}:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.CODE_CHUNK_SIZE,
                chunk_overlap=self.CODE_OVERLAP,
                separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""],
            )
        return RecursiveCharacterTextSplitter(
            chunk_size=self.DOC_CHUNK_SIZE,
            chunk_overlap=self.DOC_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Core indexer ──────────────────────────────────────────────────────────
    def _index_single_file(self, path: Path):
        name = path.name
        try:
            documents = self._load_file(path)
            if not documents:
                raise ValueError("File produced zero content")

            splitter = self._get_splitter(path)
            chunks   = splitter.split_documents(documents)

            valid = []
            for c in chunks:
                if not isinstance(c.page_content, str):
                    continue
                clean = c.page_content.strip()
                if len(clean) > 10 and any(ch.isalnum() for ch in clean):
                    c.page_content = clean
                    valid.append(c)

            if not valid:
                raise ValueError("No valid chunks after filtering")

            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(valid, self.embeddings)
            else:
                try:
                    self.vectorstore.add_documents(valid)
                except Exception as e:
                    print(f"  [!] Partial embed error for {name}: {e}")

            self.manifest["indexed"].append(name)
            src = path.parent.name
            print(f"  ✓ [{src}] {name}: {len(valid)} chunks")

        except Exception as e:
            self.manifest["failed"].append({"file": name, "reason": str(e)})
            print(f"  ✗ {name}: SKIPPED — {e}")

        finally:
            try:
                del documents, chunks, valid
            except Exception:
                pass
            gc.collect()

    # ── Public API ────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 10,
               source_type: str = None) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            return []
        try:
            fetch_k = k * 3 if source_type else k
            docs    = self.vectorstore.similarity_search(query, k=fetch_k)
            results = []
            for doc in docs:
                meta = doc.metadata
                if source_type and meta.get("source_type") != source_type:
                    continue
                results.append({
                    "content":     doc.page_content,
                    "source":      meta.get("source_file", "Unknown"),
                    "source_type": meta.get("source_type", "doc"),
                    "language":    meta.get("language",    "text"),
                    "page":        meta.get("page",        0),
                })
                if len(results) >= k:
                    break
            return results
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return []

    def get_context(self, query: str, k: int = 10,
                    source_type: str = None) -> str:
        results = self.search(query, k=k, source_type=source_type)
        if not results:
            return ""

        by_source: Dict[str, List[Dict]] = {}
        for r in results:
            by_source.setdefault(r["source"], []).append(r)

        parts = ["[KNOWLEDGE FROM YOUR BOOKS & CODE]\n"]
        for source, chunks in list(by_source.items())[:5]:
            stype = chunks[0]["source_type"]
            lang  = chunks[0]["language"]
            icon  = "💻" if stype == "code" else "📖"
            parts.append(f"\n{icon} From {source}:")
            for chunk in chunks[:3]:
                if stype == "code":
                    parts.append(f"```{lang}\n{chunk['content'][:600]}\n```")
                else:
                    parts.append(chunk["content"][:600])
        return "\n".join(parts)

    def add_file(self, path: Path) -> Dict[str, Any]:
        name = path.name
        # Allow re-indexing updated files
        if name in self.manifest["indexed"]:
            self.manifest["indexed"].remove(name)
        # Clear from failed too
        self.manifest["failed"] = [e for e in self.manifest["failed"] if e["file"] != name]

        self._index_single_file(path)
        if self.vectorstore:
            self.vectorstore.save_local(str(FAISS_DIR))
        self._save_manifest()

        success = name in self.manifest["indexed"]
        return {
            "ok":      success,
            "message": f"Indexed {name}" if success else f"Failed: see /report",
        }

    def get_report(self) -> Dict[str, Any]:
        return {
            "total":   len(self.manifest["indexed"]),
            "indexed": self.manifest["indexed"],
            "failed":  self.manifest["failed"],
        }


# Global instance
kb = KnowledgeBase()


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="RAG Server", version="2.0")


class SearchRequest(BaseModel):
    query:       str
    k:           int = 10
    source_type: str = None  # "doc" | "code" | None = search everything


# ── Search ────────────────────────────────────────────────────────────────────

@app.post("/search")
async def search(req: SearchRequest):
    results = await asyncio.to_thread(kb.search, req.query, min(req.k, 20), req.source_type)
    return {"results": results, "count": len(results)}


@app.post("/context")
async def context(req: SearchRequest):
    ctx = await asyncio.to_thread(kb.get_context, req.query, min(req.k, 20), req.source_type)
    return {"context": ctx}


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/upload/doc")
async def upload_doc(file: UploadFile = File(...)):
    """Upload PDF, DOCX, TXT, or MD into doc/ and index immediately."""
    ext = Path(file.filename).suffix.lower()
    if ext not in DOC_EXTENSIONS:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: {DOC_EXTENSIONS}")
    dest = DOC_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = await asyncio.to_thread(kb.add_file, dest)
    return {"filename": file.filename, **result}


@app.post("/upload/code")
async def upload_code(file: UploadFile = File(...)):
    """Upload PY, C, CPP, H, or MD into code/ and index immediately."""
    ext = Path(file.filename).suffix.lower()
    if ext not in CODE_EXTENSIONS:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: {CODE_EXTENSIONS}")
    dest = CODE_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = await asyncio.to_thread(kb.add_file, dest)
    return {"filename": file.filename, **result}


@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload image into static/uploads/ for vision tasks. Not RAG-indexed."""
    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: {IMAGE_EXTENSIONS}")
    upload_dir = Path(BASE_DIR) / "static" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "filename": file.filename, "url": f"/static/uploads/{file.filename}"}


# ── Info ──────────────────────────────────────────────────────────────────────

@app.get("/report")
async def report():
    return kb.get_report()


@app.get("/health")
async def health():
    return {
        "status":   "operational",
        "port":     5080,
        "total":    len(kb.manifest["indexed"]),
        "ready":    kb.vectorstore is not None,
        "doc_dir":  str(DOC_DIR),
        "code_dir": str(CODE_DIR),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5080, reload=False)