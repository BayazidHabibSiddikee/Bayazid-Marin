import os
import re
import json
import asyncio
import subprocess
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import httpx

# ── RAG configuration ──────────────────────────────────────────────────────────
RAG_URL = "http://127.0.0.1:5080"

async def get_rag_context(query: str, enabled: bool = True) -> str:
    if not enabled:
        return ""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{RAG_URL}/context",
                json={"query": query, "k": 10},
                timeout=10.0
            )
            if r.status_code == 200:
                return r.json().get("context", "")
    except Exception as e:
        print(f"[RAG] Context fetch error: {e}")
    return ""

# ── Media Analysis (YouTube / Image) ─────────────────────────────────────────

async def analyze_youtube(url: str) -> str:
    def _fetch(url: str) -> str:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            vid_id = None
            if "youtu.be/"   in url: vid_id = url.split("youtu.be/")[1].split("?")[0]
            elif "v="        in url: vid_id = url.split("v=")[1].split("&")[0]
            if not vid_id: return None
            ytt_api = YouTubeTranscriptApi()
            tlist   = ytt_api.list(vid_id)
            t       = next(iter(tlist), None)
            if not t: return None
            if t.language_code != "en" and t.is_translatable:
                t = t.translate("en")
            full = " ".join(e.text for e in t.fetch())
            if len(full) > 3000: full = full[:3000] + "... [truncated]"
            return full
        except Exception as e:
            print(f"[AgentLogic] YouTube fetch failed: {e}")
            return None

    result = await asyncio.to_thread(_fetch, url)
    if result:
        return f"YouTube video transcript:\n---\n{result}\n---"
    return "[Failed to fetch YouTube transcript]"

async def analyze_image(image_path: str) -> str:
    try:
        from image import response as leo
        if not leo: return "[Image analyzer unavailable]"
        def _collect():
            return "".join(leo("Describe this image in detail.", image_path))
        description = await asyncio.to_thread(_collect)
        return f"Image analysis: {description}"
    except ImportError:
        return "[Image analyzer module not found]"

# ── Tool Execution ──────────────────────────────────────────────────────────

def execute_text_commands(text: str, base_dir: str):
    """
    Scan text for shell commands and execute them.
    (Similar to marin.py's _exec_text_commands)
    """
    from marin import _TEXT_CMD_PAT, _strip_md_trail, _convert_heredocs
    from marin_fier import is_cmd_allowed, _cmd_log
    
    body = re.sub(r'```(?:\w*\n)?([\s\S]*?)```', r'\1', text)
    body = re.sub(r'[^\x20-\x7E\n]', '', body)
    body = re.sub(r'`([^`\n]+)`', r'\1', body)
    body = _convert_heredocs(body)

    raw_cmds = []
    for m in _TEXT_CMD_PAT.finditer(body):
        cmd = _strip_md_trail(m.group(1))
        if cmd:
            raw_cmds.append(cmd)

    if not raw_cmds: return

    def _run():
        for cmd in raw_cmds:
            allowed, reason = is_cmd_allowed(cmd)
            if not allowed: continue
            
            try:
                r = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=30,
                    cwd=base_dir,
                    env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")},
                )
                out = f"[EXIT {r.returncode}] {(r.stdout or r.stderr or '(done)').strip()[:500]}"
                print(f"[Agent] Ran: {cmd[:80]} -> {out[:100]}")
                
                # Update cmd log if available
                if _cmd_log is not None:
                    ts = datetime.now().strftime("%H:%M:%S")
                    _cmd_log.append({"cmd": cmd, "allowed": True, "output": out[:200], "ts": ts})
                    if len(_cmd_log) > 100: _cmd_log.pop(0)
            except Exception as e:
                print(f"[Agent] Command failed: {cmd[:80]} — {e}")

    threading.Thread(target=_run, daemon=True).start()

# ── Unified Preprocessor ─────────────────────────────────────────────────────

async def preprocess_input(user_input: str, image_path: str = None, rag_enabled: bool = False, agent_name: str = "marin") -> Dict[str, Any]:
    from marin_fier import classify, execute_tool
    
    classification = classify(user_input, agent_name=agent_name)
    intent = classification.get("intent", "chat")
    params = classification.get("params", {})
    
    tool_outputs = []
    if intent not in ("chat", "normal", "learn", "code", "lab"):
        try:
            out = execute_tool(intent, params, agent_name=agent_name)
            if out: tool_outputs.append(f"[TOOL: {intent}]\n{out}")
        except Exception as e:
            print(f"[AgentLogic] Tool execution failed: {e}")

    yt_regex = r"(https?://)?(www.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[^\s]+"
    is_youtube = bool(re.search(yt_regex, user_input, re.IGNORECASE))
    
    rag_context = ""
    if rag_enabled:
        rag_context = await get_rag_context(user_input)

    media_blocks = []
    if is_youtube or image_path:
        tasks = []
        if is_youtube: tasks.append(analyze_youtube(user_input))
        if image_path:   tasks.append(analyze_image(image_path))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            media_blocks.append("[Media analysis failed]" if isinstance(res, Exception) else res)

    parts = []
    if rag_context:   parts.append(rag_context)
    if media_blocks:  parts.append("CONTEXT FROM MEDIA:\n" + "\n".join(media_blocks))
    if tool_outputs:  parts.append("TOOL EXECUTION RESULTS:\n" + "\n\n".join(tool_outputs))
    parts.append(f"USER'S MESSAGE: {user_input}")

    enriched_prompt = "\n\n".join(parts)
    
    return {
        "enriched_prompt": enriched_prompt,
        "classification": classification,
        "rag_context": rag_context,
        "tool_outputs": tool_outputs
    }
