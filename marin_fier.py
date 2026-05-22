#!/usr/bin/env python3
"""
marin_fier.py — Two-stage intent classifier with LangChain StructuredTool

Stage 1 — Regex:  Zero-latency, handles obvious patterns + typos via fuzzy match.
Stage 2 — LLM:   ChatOllama(qwen2.5:0.5b).bind_tools(TOOLS) with Pydantic schemas.
                  qwen outputs a structured tool_call with validated, typed params.
                  num_predict=40 — enough for one tool call JSON, nothing more.

Flow:
  user text
    → regex pre-filter (instant)
    → if no match: qwen with bound tools → StructuredTool call
    → params validated by Pydantic
    → return {intent, params, user_vibe, confidence, _tool_ack}
"""

import re
import json
import os
import signal
import sys
import subprocess
from pathlib import Path
from difflib import get_close_matches
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama

BASE_DIR = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS — typed, validated params for every tool
# ══════════════════════════════════════════════════════════════════════════════

class AlarmInput(BaseModel):
    time: str = Field(description="Clock time to alarm, e.g. '7:30 AM', '23:00'")

class TimerInput(BaseModel):
    duration: str = Field(description="Duration, e.g. '30 minutes', '1 hour 15 minutes'")

class CryptoInput(BaseModel):
    coin: str = Field(
        default="bitcoin",
        description="Coin id: bitcoin, ethereum, solana, dogecoin, bnb, cardano, ripple, etc."
    )

class StockInput(BaseModel):
    company: str = Field(
        description="Company name (1-3 words) or ticker, e.g. 'Apple', 'Tesla', 'AAPL'"
    )

class Connect4Input(BaseModel):
    mode: str = Field(
        default="computer",
        description="'computer' for AI opponent, 'two' for two-player mode"
    )

class CmdInput(BaseModel):
    command: str = Field(description="Shell command to run, e.g. 'ls -la', 'git status', 'python3 --version'")

class MathPlotInput(BaseModel):
    expression: str = Field(
        description=(
            "Full math expression to plot. Can be:\n"
            "- A preset name: 'circle', 'heart', 'rose', 'spiral', 'butterfly', 'lissajous', "
            "'cardioid', 'astroid', 'lemniscate', 'sine', 'parabola'\n"
            "- Parametric: x=r*cos(t), y=r*sin(t)\n"
            "- Equation: 'y = x^2', 'x = y^2 + 5'\n"
            "- Natural language: 'draw a heart', 'plot butterfly'\n"
            "- Explicit: '-x r*cos(t) -y r*sin(t)'"
        )
    )

class RunSequenceInput(BaseModel):
    commands: str = Field(
        description=(
            "One or more commands to run sequentially. "
            "Each command is auto-killed after a delay (4s math, 20s stock, 10s crypto, 40s butterfly).\n\n"
            "Acceptable formats:\n"
            '- Comma-separated presets: "heart, butterfly, spiral"\n'
            '- Space-separated presets: "heart butterfly spiral"\n'
            '- Semi-colon separated shell commands: "ls -la; python3 maths/mathplot.py heart"\n'
            '- JSON array: \'[{"cmd":"python3 maths/mathplot.py heart","delay":3}]\'\n\n'
            "This tool is ideal when the user asks for MULTIPLE graphs or operations at once."
        )
    )

class NoInput(BaseModel):
    pass   # tools that need no parameters


# ── COMMAND ALLOWLIST ─────────────────────────────────────────────────────────
_CMD_ALLOW = re.compile(
    r'^(ls|cat|pwd|echo|whoami|date|uptime|df|du|free|uname|hostname|'
    r'ps|git|python3?|pip3?|mkdir|touch|cp|mv|ln|rmdir|chmod|chown|'
    r'find|grep|rg|head|tail|wc|sort|uniq|cut|awk|sed|tr|'
    r'curl|wget|make|gcc|g\+\+|avr-gcc|avrdude|cargo|rustc|'
    r'tree|which|type|env|printenv|lsusb|lsblk|ip|ping|'
    r'ollama|uvicorn|node|npm|npx|yarn|pnpm|'
    r'systemctl|journalctl|ffmpeg|ffprobe|yt-dlp|youtube-dl|'
    r'cd|nano|vim|nvim|emacs|code|less|more|'
    r'docker|podman|kubectl|docker-compose|'
    r'ssh|scp|rsync|diff|stat|realpath|readlink|basename|dirname|'
    r'time|timeout|htop|top|btop|atop|sensors|lscpu|lspci|lsmod|dmesg|'
    r'bash|zsh|sh|fish|screen|tmux|'
    r'nohup|watch|sleep|id|who|groups|users|'
    r'firefox|chromium|google-chrome|xdg-open|mpv|vlc|'
    r'notify-send|file|bc|expr|tee|yes|xargs|'
    r'convert|magick|tesseract|jq|yq|'
    r'nl|od|xxd|hexdump|expand|unexpand|fmt|pr|fold|'
    r'locale|timedatectl|localectl)',
    re.IGNORECASE
)
_CMD_BLOCK = re.compile(
    r'(rm\s+-rf\s*/|sudo\s+rm|dd\s+if=|mkfs|shutdown|reboot|passwd|'
    r'useradd|userdel|wget.+\|\s*bash|curl.+\|\s*sh|>\s*/dev/sd|nc\s+-e|bash\s+-i)',
    re.IGNORECASE
)

_cmd_log: list[dict] = []   # in-memory log (last 100), served at /cmd/log

# ── KNOWN SCRIPTS — alias → full command; auto-resolved when mentioned in chat ──
_KNOWN_SCRIPTS: dict[str, str] = {
    "execute_limoni": "python3 unique/execute_limoni.py",
    "limoni":         "python3 unique/execute_limoni.py",
    "unique":         "python3 unique/execute_limoni.py",
    "marin":          "python3 marin.py",
    "bayazid":        "python3 bayazid.py",
    "arena":          "python3 arena.py",
    "rag_server":     "python3 rag_server.py",
    "run_marin":      "bash run_marin.sh",
    "run_bayazid":    "bash run_bayazid.sh",
    "run_arena":      "bash run_arena.sh",
    "run_all":        "bash run_all.sh",
    "mathplot":       "python3 maths/mathplot.py",
    "stock":          "python3 tools/stock.py",
    "crypto":         "python3 tools/crypto.py",
}
_KNOWN_SCRIPT_KEYS: set[str] = set(_KNOWN_SCRIPTS.keys())


def is_cmd_allowed(cmd: str) -> tuple[bool, str]:
    first = cmd.strip().split()[0].lstrip('./') if cmd.strip() else ""
    if _CMD_BLOCK.search(cmd):
        return False, "dangerous pattern blocked"
    if not _CMD_ALLOW.match(first):
        return False, f"'{first}' not in allowlist"
    return True, "ok"


def tool_run_command(command: str) -> str:
    import datetime
    allowed, reason = is_cmd_allowed(command)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    entry = {"cmd": command, "allowed": allowed, "output": "", "ts": ts}
    if not allowed:
        entry["output"] = reason
        _cmd_log.append(entry)
        if len(_cmd_log) > 100: _cmd_log.pop(0)
        return f"Can't run `{command}` — {reason}."
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15)
        out = (r.stdout or r.stderr or "(no output)").strip()
        if len(out) > 1500: out = out[:1500] + "\n...[truncated]"
        entry["output"] = out
        _cmd_log.append(entry)
        if len(_cmd_log) > 100: _cmd_log.pop(0)
        return out
    except subprocess.TimeoutExpired:
        entry["output"] = "timed out (15s)"
        _cmd_log.append(entry)
        return "Command timed out after 15 seconds."
    except Exception as e:
        entry["output"] = str(e)
        _cmd_log.append(entry)
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL LAUNCHERS — each function fires the subprocess and returns a context str
# These are the actual callables passed to StructuredTool
# ══════════════════════════════════════════════════════════════════════════════

def _popen(script: str, args: list[str] = [], timeout: float | None = None):
    import threading
    path = BASE_DIR / script
    if not path.exists():
        return f"Script not found: {script}"
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")
    try:
        proc = subprocess.Popen(
            [sys.executable, str(path)] + args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(BASE_DIR),
            env=env,
        )
    except Exception as e:
        return f"Failed to launch {script}: {e}"

    if timeout:
        def _kill():
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                import time
                time.sleep(0.4)
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                print(f"[tool:{script}] killed after {timeout}s")
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"[tool:{script}] kill error: {e}")
        threading.Timer(timeout, _kill).start()

    return None   # None = launched OK


def tool_set_alarm(time: str) -> str:
    err = _popen("tools/alarm.py", [time])
    if err: return err
    return (f"Alarm set for {time}. "
            f"The alarm script is running in the background and will beep when it fires.")

def tool_set_timer(duration: str) -> str:
    err = _popen("tools/timer.py", [duration])
    if err: return err
    return f"Countdown timer started for {duration}. It's ticking in the background."

def tool_get_crypto_price(coin: str = "bitcoin") -> str:
    coin = coin.lower().strip()
    err = _popen("tools/crypto.py", ["--coin", coin], timeout=10)
    if err: return err
    return (f"Live {coin.title()} price tracker window is now open. "
            f"Shows USD price refreshing every second via CoinGecko.")

def tool_get_stock_info(company: str) -> str:
    # Clamp to 3 words max
    company = " ".join(company.split()[:3]).strip()
    # If it's a known ticker (ALL-CAPS, ≤5 chars) pass --ticker to bypass Yahoo search
    if company.upper() in _COMMON_UPPER or (company.isupper() and 1 <= len(company) <= 5):
        err = _popen("tools/stock.py", ["--ticker", company.upper()])
    else:
        err = _popen("tools/stock.py", ["--company", company])
    if err: return err
    return (f"Stock info window opened for {company}. "
            f"Fetching current market price and 30-day chart via Yahoo Finance.")

def tool_open_news() -> str:
    err = _popen("tools/news.py")
    if err: return err
    return "BBC News / Al Jazeera is loading in your default browser."

def tool_send_email() -> str:
    err = _popen("tools/email_tool.py")
    if err: return err
    return "Email composer launched. It will ask for recipient, subject, and body interactively."

def tool_play_tictactoe() -> str:
    err = _popen("tools/tictactoe.py")
    if err: return err
    return "Tic Tac Toe game window is opening."

def tool_play_connect4(mode: str = "computer") -> str:
    args = ["--two"] if mode == "two" else []
    err = _popen("tools/connect4.py", args)
    if err: return err
    return f"Connect Four launched in {mode} mode."

def tool_play_wordgame() -> str:
    err = _popen("tools/wordgame.py")
    if err: return err
    return "Word scramble game is starting."

def tool_draw_me() -> str:
    err = _popen("tools/draw_me.py")
    if err: return err
    return "Draw-me tool launched. It will take a webcam photo and render it as a turtle drawing."

def tool_take_screenshot() -> str:
    err = _popen("tools/image.py")
    if err: return err
    return "Screenshot captured and saved."

def tool_math_plot(expression: str) -> str:
    """Launch the math equation plotter to draw parametric curves."""
    import shlex
    args = shlex.split(expression)
    err = _popen("maths/mathplot.py", args)
    if err: return err
    return f"Math plotter launched for: '{expression}'. CNC simulation window opening."


def tool_run_sequence(commands: str) -> str:
    """
    Run multiple commands sequentially with auto-kill delays.
    Delegates to tools/command_queue.py for timing and lifecycle.
    """
    import json
    import tempfile
    import datetime

    trimmed = commands.strip()
    cmd_list = []

    # Try JSON array
    if trimmed.startswith("["):
        try:
            cmd_list = json.loads(trimmed)
        except json.JSONDecodeError:
            pass

    # Try JSON object (single command)
    if not cmd_list and trimmed.startswith("{"):
        try:
            cmd_list = [json.loads(trimmed)]
        except json.JSONDecodeError:
            pass

    if not cmd_list:
        # Split by semicolon (shell commands) or comma (presets)
        parts = [p.strip() for p in re.split(r'[;,|]', trimmed) if p.strip()]
        for p in parts:
            # If it looks like a shell command (has spaces or slashes), run as-is
            if " " in p or "/" in p:
                cmd_list.append({"cmd": p, "delay": 3, "name": p[:40]})
            else:
                # Treat as mathplot preset
                cmd_list.append({
                    "cmd": f"python3 maths/mathplot.py {p}",
                    "delay": 4,
                    "name": f"Plot: {p}",
                })

    if not cmd_list:
        return "run_sequence: no commands parsed."

    # Default delays: stock=20s, crypto=10s, math=4s, butterfly=40s
    for item in cmd_list:
        if "delay" not in item:
            cmd = item.get("cmd", "").lower()
            if "stock" in cmd or "yahoo" in cmd:
                item["delay"] = 20
            elif "crypto" in cmd or "finance" in cmd:
                item["delay"] = 10
            elif "butterfly" in cmd or "teal" in cmd:
                item["delay"] = 40
            else:
                item["delay"] = 4

    # Write to temp JSON and delegate to command_queue.py
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump(cmd_list, tmp)
        tmp_path = tmp.name
    finally:
        tmp.close()

    try:
        r = subprocess.run(
            [sys.executable, str(BASE_DIR / "tools/command_queue.py"), "--json", tmp_path],
            capture_output=True, text=True, timeout=300,
            cwd=str(BASE_DIR),
            env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")},
        )
        out = (r.stdout or r.stderr or "(no output)").strip()
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        _cmd_log.append({
            "cmd": f"[tool:run_sequence] {trimmed[:80]}",
            "allowed": True,
            "output": out[:500],
            "ts": ts,
        })
        if len(_cmd_log) > 100:
            _cmd_log.pop(0)
        return f"Sequenced {len(cmd_list)} commands:\n{out[:1200]}"
    except Exception as e:
        return f"run_sequence error: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED TOOLS — bind schemas to callables
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    StructuredTool.from_function(
        func=tool_set_alarm, name="set_alarm",
        description="Set a clock alarm at a specific time.",
        args_schema=AlarmInput,
    ),
    StructuredTool.from_function(
        func=tool_set_timer, name="set_timer",
        description="Start a countdown timer for a given duration.",
        args_schema=TimerInput,
    ),
    StructuredTool.from_function(
        func=tool_get_crypto_price, name="get_crypto_price",
        description="Open live cryptocurrency price tracker. Use when user asks about crypto prices.",
        args_schema=CryptoInput,
    ),
    StructuredTool.from_function(
        func=tool_get_stock_info, name="get_stock_info",
        description="Open stock price and 30-day chart for a company. Use when user asks about stocks or shares.",
        args_schema=StockInput,
    ),
    StructuredTool.from_function(
        func=tool_open_news, name="open_news",
        description="Open news website in browser.",
        args_schema=NoInput,
    ),
    StructuredTool.from_function(
        func=tool_send_email, name="send_email",
        description="Launch interactive email composer.",
        args_schema=NoInput,
    ),
    StructuredTool.from_function(
        func=tool_play_tictactoe, name="play_tictactoe",
        description="Launch Tic Tac Toe game.",
        args_schema=NoInput,
    ),
    StructuredTool.from_function(
        func=tool_play_connect4, name="play_connect4",
        description="Launch Connect Four game.",
        args_schema=Connect4Input,
    ),
    StructuredTool.from_function(
        func=tool_play_wordgame, name="play_wordgame",
        description="Launch word scramble game.",
        args_schema=NoInput,
    ),
    StructuredTool.from_function(
        func=tool_draw_me, name="draw_me",
        description="Take webcam photo and draw it as turtle art.",
        args_schema=NoInput,
    ),
    StructuredTool.from_function(
        func=tool_take_screenshot, name="take_screenshot",
        description="Take a screenshot.",
        args_schema=NoInput,
    ),
    StructuredTool.from_function(
        func=tool_run_command, name="run_command",
        description=(
            "Execute an allowlisted shell/terminal command on Limon's machine. "
            "Use when asked to run ls, git status, python scripts, check system info, etc."
        ),
        args_schema=CmdInput,
    ),
    StructuredTool.from_function(
        func=tool_math_plot, name="math_plot",
        description=(
            "Draw a math equation, parametric curve, or preset shape using the CNC simulator. "
            "Use when asked to plot, draw, or visualize any equation, shape, or graph. "
            "Supports presets: circle, heart, rose, spiral, butterfly, lissajous, cardioid, "
            "astroid, lemniscate, sine, parabola, cycloid, star, infinity. "
            "Also supports free-form equations like 'y = x^2' or 'x = r*cos(t), y = r*sin(t)', "
            "and natural language like 'draw a heart' or 'plot butterfly'."
        ),
        args_schema=MathPlotInput,
    ),
    StructuredTool.from_function(
        func=tool_run_sequence, name="run_sequence",
        description=(
            "Run MULTIPLE operations in sequence — each is auto-killed after a delay "
            "(math/graph: 4s, stock: 20s, crypto: 10s, butterfly: 40s). "
            "Use this when the user wants several graphs, stock checks, or mixed operations "
            "in one request. "
            "Accepts command presets (heart, butterfly) or full shell commands. "
            "Example: 'heart, butterfly, spiral' draws 3 shapes sequentially on one window."
        ),
        args_schema=RunSequenceInput,
    ),
]

# Map name → StructuredTool for fast lookup
_TOOL_MAP: dict[str, StructuredTool] = {t.name: t for t in TOOLS}

# ── LLM (lazy init — only created on first Stage 2 call) ─────────────────────
_LLM_WITH_TOOLS = None

def _get_llm():
    global _LLM_WITH_TOOLS
    if _LLM_WITH_TOOLS is None:
        llm = ChatOllama(
            model="gemma4:31b-cloud",
            temperature=0.0,
            num_predict=120,    # enough for tool_call JSON + command string
        )
        _LLM_WITH_TOOLS = llm.bind_tools(TOOLS, tool_choice="any")
    return _LLM_WITH_TOOLS


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — REGEX PRE-FILTER  (with fuzzy coin matching)
# ══════════════════════════════════════════════════════════════════════════════

_COIN_CANONICAL = {
    "bitcoin":      ["bitcoin","btc","bitcoun","bitcion","bicoin"],
    "ethereum":     ["ethereum","eth","etherum","etherium","ether"],
    "binancecoin":  ["bnb","binance"],
    "solana":       ["solana","sol"],
    "dogecoin":     ["dogecoin","doge","dogecoyn"],
    "cardano":      ["cardano","ada"],
    "ripple":       ["ripple","xrp"],
    "litecoin":     ["litecoin","ltc"],
    "polkadot":     ["polkadot","dot"],
    "polygon":      ["polygon","matic"],
    "shiba-inu":    ["shiba","shib"],
    "avalanche-2":  ["avalanche","avax"],
}
_ALIAS_TO_CANON = {
    alias: canon
    for canon, aliases in _COIN_CANONICAL.items()
    for alias in aliases
}

def _fuzzy_coin(word: str) -> str | None:
    word = word.lower().strip()
    if word in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[word]
    if len(word) >= 5:
        close = get_close_matches(word, _ALIAS_TO_CANON.keys(), n=1, cutoff=0.82)
        if close:
            return _ALIAS_TO_CANON[close[0]]
    return None

def _is_crypto_text(lower: str) -> bool:
    if re.search(r'\bcrypto(?:currency)?\b', lower):
        return True
    for word in re.split(r'\W+', lower):
        if _fuzzy_coin(word):
            return True
    return False

def _extract_coin_from(lower: str) -> str:
    for word in re.split(r'\W+', lower):
        c = _fuzzy_coin(word)
        if c:
            return c
    return "bitcoin"

# Tickers that must stay ALL-CAPS and should never be .title()-cased
_COMMON_UPPER: set[str] = {
    "AAPL","TSLA","MSFT","AMZN","GOOGL","META","NVDA","NFLX","AMD","INTC",
    "BABA","UBER","LYFT","SNAP","TWTR","PYPL","SQ","SHOP","COIN","HOOD",
    "GME","AMC","BB","NOK","PLTR","NIO","RIVN","LCID","F","GM","FORD",
    "BA","GE","XOM","CVX","WMT","TGT","COST","KO","PEP","JNJ","PFE",
    "MRNA","BNTX","ABBV","MRK","JPM","BAC","WFC","GS","MS","V","MA",
    "DIS","CMCSA","T","VZ","TMUS","ORCL","IBM","CRM","NOW","ADBE","QCOM",
    "TXN","AVGO","MU","LRCX","AMAT","ASML","TSM","SONY","SAMSNG","005930",
    "BRK","BRKB","SPY","QQQ","VOO","VTI","ARKK","IWM",
}

_TICKER_PAT = re.compile(r'\b([A-Z]{1,5})\b')

def _extract_ticker_from(text: str) -> str | None:
    """
    Return the first ALL-CAPS word that looks like a real known ticker.
    Ignores short noise words (I, A, AT, IS, etc.) via _COMMON_UPPER check.
    Returns None if nothing found.
    """
    for m in _TICKER_PAT.finditer(text):
        word = m.group(1)
        if word in _COMMON_UPPER:
            return word
    return None


def _extract_company_from(text: str) -> str:
    """
    Extract company name from *original-case* text so tickers like 'TSLA'
    aren't mangled to 'Tsla' by .title().
    """
    lower = text.lower()
    cleaned = re.sub(
        r'\b(show me|get me|check|open|what is the|what\'s the|tell me|'
        r'look up|pull up|find|give me|display|fetch|please|'
        r'stock price of|stock of|price of|share price of|'
        r'stock price|share price|stock info|market price|ticker for|'
        r'stock|price|market|shares?|the|a|an)\b',
        ' ', lower, flags=re.IGNORECASE
    )
    # Rebuild from original-case text at positions that survived cleaning
    # Simple approach: strip the same filler from original text
    cleaned_orig = re.sub(
        r'\b(show me|get me|check|open|what is the|what\'s the|tell me|'
        r'look up|pull up|find|give me|display|fetch|please|'
        r'stock price of|stock of|price of|share price of|'
        r'stock price|share price|stock info|market price|ticker for|'
        r'stock|price|market|shares?|the|a|an)\b',
        ' ', text, flags=re.IGNORECASE
    )
    words = [w for w in cleaned_orig.split() if len(w) > 1][:3]
    if not words:
        return ""
    # Keep ALL-CAPS words as-is (tickers); .title() everything else
    titled = []
    for w in words:
        titled.append(w if w.upper() in _COMMON_UPPER else w.title())
    company = " ".join(titled).strip()
    if not company or len(company) > 30:
        return ""
    return company

def _extract_time_from(lower: str) -> str:
    m = re.search(r'alarm\s+(?:for|at)\s+(.+?)(?:\s*$|\s+(?:tomorrow|today))', lower)
    if m: return m.group(1).strip()
    m = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.))', lower)
    if m: return m.group(1).strip()
    return lower.strip()

def _extract_duration_from(lower: str) -> str:
    m = re.search(
        r'(\d+\s*(?:hour|hr|minute|min|second|sec)s?'
        r'(?:\s+(?:and\s+)?\d+\s*(?:minute|min|second|sec)s?)?)',
        lower
    )
    if m: return m.group(1).strip()
    return lower.strip()


# ── REGEX PATTERNS for Stage 1 ────────────────────────────────────────────────

_STOCK_PAT = re.compile(r'\b(stock|share\s*price|ticker|market\s*price)\b')
_ALARM_PAT = re.compile(r'\b(set\s+an?\s+alarm|alarm\s+(?:for|at))\b')
_TIMER_PAT = re.compile(r'\b(set\s+a?\s*timer|start\s+timer)\b|\d+\s*(?:min|minute|hour|hr|sec)\s*(?:timer|countdown)')
_NEWS_PAT  = re.compile(r'\b(open\s+news|latest\s+news|news|headlines)\b')
_EMAIL_PAT = re.compile(r'\b(send\s+(?:an?\s+)?email|email\s+to|compose\s+(?:an?\s+)?email|write\s+(?:a\s+)?mail)\b')
_TTT_PAT   = re.compile(r'\b(tic\s*tac\s*toe|tiktaktoe|tictactoe)\b')
_C4_PAT    = re.compile(r'\b(connect\s*4|connect\s*four|connect4)\b')
_WORD_PAT  = re.compile(r'\b(word\s*game|wordgame|word\s*scramble)\b')
_DRAW_PAT  = re.compile(r'\b(draw\s+me|take\s+(?:my\s+)?photo\s+and\s+draw)\b')
_SHOT_PAT   = re.compile(r'\b(screenshot|take\s+a\s*(?:pic|screen))\b')
_ALL_PAT    = re.compile(r'\b(use\s+(?:every|all)\s+(?:tool|command|ability)|run\s+(?:every|all)\s+(?:tool|command)|show\s+(?:everything|all\s+tools))\b')
_RUN_PAT    = re.compile(
    r'\b(?:run|execute)\s+(?:command\s*:?\s*)?(`?[a-z_\-./]+\s*(?:\|.*)?`?)',
    re.IGNORECASE
)
_CAN_YOU_RUN_PAT = re.compile(
    r'\b(?:can\s+you|could\s+you|please)\s+(`?[a-z][a-z0-9_\-+./]*`?(?:\s+[^\s,.!?]+){0,6})\s*$',
    re.IGNORECASE
)
_MATH_PLOT_PAT = re.compile(
    r'\b(draw|plot|graph)\s+(?:me\s+)?(?:a\s+)?(.+?)\s*$',
    re.IGNORECASE
)
_EXECUTING_PAT = re.compile(
    r'\bEXECUTING\s+(.+?)(?:\s*$|[,;!?])',
    re.IGNORECASE
)


def _regex_stage(text: str) -> dict | None:
    """Returns {intent, params, confidence} or None if uncertain."""
    lower = text.lower()

    # "use every tool" / "run all tools" → special batch intent
    if _ALL_PAT.search(lower):
        return {"intent": "run_all_tools", "params": {}, "confidence": 0.97}

    # Crypto BEFORE stock — so 'ethereum stock price' → crypto
    if _is_crypto_text(lower):
        return {"intent": "get_crypto_price",
                "params": {"coin": _extract_coin_from(lower)},
                "confidence": 0.97}

    if _STOCK_PAT.search(lower):
        company = _extract_company_from(text)   # pass original text, not lower
        if not company:
            return None   # hand off to qwen for clean extraction
        return {"intent": "get_stock_info",
                "params": {"company": company},
                "confidence": 0.97}

    # ── Math / Equation Plot: "draw heart", "plot butterfly", "graph y = x^2" ──
    math_m = _MATH_PLOT_PAT.search(text)
    if math_m:
        expr = math_m.group(2).strip()
        # Multi-preset: "draw heart, butterfly, and spiral" or "draw heart butterfly spiral"
        if re.search(r',|\band\b', expr) or (
            len(expr.split()) >= 3
            and not any(c in expr for c in "=^+-*/()")
        ):
            return {"intent": "run_sequence",
                    "params": {"commands": expr}, "confidence": 0.92}
        return {"intent": "math_plot", "params": {"expression": expr}, "confidence": 0.92}

    # Bare ticker like "TSLA", "AAPL price", "how is NVDA doing" —
    # no "stock" keyword required if we recognise the symbol.
    # Only match when NOT asking for a math plot (checked above).
    ticker = _extract_ticker_from(text)
    if ticker and not _MATH_PLOT_PAT.search(text):
        return {"intent": "get_stock_info",
                "params": {"company": ticker},
                "confidence": 0.95}

    if _ALARM_PAT.search(lower):
        return {"intent": "set_alarm",
                "params": {"time": _extract_time_from(lower)},
                "confidence": 0.97}

    if _TIMER_PAT.search(lower):
        return {"intent": "set_timer",
                "params": {"duration": _extract_duration_from(lower)},
                "confidence": 0.97}

    if _NEWS_PAT.search(lower):
        return {"intent": "open_news", "params": {}, "confidence": 0.97}

    if _EMAIL_PAT.search(lower):
        return {"intent": "send_email", "params": {}, "confidence": 0.97}

    if _TTT_PAT.search(lower):
        return {"intent": "play_tictactoe", "params": {}, "confidence": 0.97}

    if _C4_PAT.search(lower):
        mode = "two" if re.search(r'\b(two|2|friend|player)\b', lower) else "computer"
        return {"intent": "play_connect4", "params": {"mode": mode}, "confidence": 0.97}

    if _WORD_PAT.search(lower):
        return {"intent": "play_wordgame", "params": {}, "confidence": 0.97}

    if _DRAW_PAT.search(lower):
        return {"intent": "draw_me", "params": {}, "confidence": 0.97}

    if _SHOT_PAT.search(lower):
        return {"intent": "take_screenshot", "params": {}, "confidence": 0.97}

    # ── "EXECUTING <command>" — high-priority command trigger ─────────────
    exec_m = _EXECUTING_PAT.search(text)
    if exec_m:
        cmd = exec_m.group(1).strip().strip('`"\'')
        alias = cmd.lower().rstrip('.py').rstrip('.sh')
        if alias in _KNOWN_SCRIPTS:
            cmd = _KNOWN_SCRIPTS[alias]
        elif '/' in alias:
            base = alias.rsplit('/', 1)[-1]
            if base in _KNOWN_SCRIPTS:
                cmd = _KNOWN_SCRIPTS[base]
        else:
            first = cmd.split()[0].lstrip('./')
            if not _CMD_ALLOW.match(first):
                return {"intent": "normal", "params": {}, "confidence": 0.6}
        return {"intent": "run_command", "params": {"command": cmd}, "confidence": 0.96}

    # ── Known script alias with run/launch/execute trigger ───────────────
    for alias, full_cmd in _KNOWN_SCRIPTS.items():
        if re.search(
            rf'\b(?:run|start|launch|open|execute)\s+(?:the\s+)?'
            rf'{re.escape(alias)}(?:\s+(?:script|file))?(?:\s*$|[.!?,;])',
            text, re.IGNORECASE
        ):
            return {"intent": "run_command", "params": {"command": full_cmd}, "confidence": 0.94}

    # ── Command: explicit "run X" / "execute X" ──────────────────────────
    run_m = _RUN_PAT.search(text)
    if run_m:
        cmd = run_m.group(1).strip().strip('`')
        return {"intent": "run_command", "params": {"command": cmd}, "confidence": 0.92}

    # ── Command: "can you X / could you / please X" where X is a known command ──
    can_m = _CAN_YOU_RUN_PAT.search(text)
    if can_m:
        phrase = can_m.group(1).strip().strip('`')
        first_word = phrase.split()[0].lower().lstrip('./')
        if _CMD_ALLOW.match(first_word):
            return {"intent": "run_command", "params": {"command": phrase}, "confidence": 0.88}

    # ── Bare command: text starts with a known allowlisted command ─────────────
    bare = text.strip()
    if bare:
        first = bare.split()[0].lstrip('./')
        if _CMD_ALLOW.match(first):
            # Only for short/command-like messages (skip long conversational text that happens to start with a command word)
            if len(bare) < 120 and not re.search(r'[?.!]\s*$', bare):
                return {"intent": "run_command", "params": {"command": bare}, "confidence": 0.85}

    return None


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — QWEN WITH BOUND STRUCTURED TOOLS
# ══════════════════════════════════════════════════════════════════════════════

def _llm_stage(text: str) -> dict | None:
    """
    Ask qwen2.5:0.5b with bound StructuredTools.
    If it emits a tool_call, invoke the tool and return result.
    If no tool_call (i.e. it's just chat), return intent=chat.
    """
    try:
        llm = _get_llm()
        ai_msg = llm.invoke(text)
        tool_calls = getattr(ai_msg, "tool_calls", None)

        if not tool_calls:
            return {"intent": "chat", "params": {}, "confidence": 0.80}

        # Take only the first tool call (we never need multiple)
        tc   = tool_calls[0]
        name = tc["name"]
        args = tc.get("args", {})

        if name not in _TOOL_MAP:
            return {"intent": "chat", "params": {}, "confidence": 0.50}

        # Validate args through the Pydantic schema
        schema_cls = _TOOL_MAP[name].args_schema
        try:
            validated = schema_cls(**args)
            clean_args = validated.model_dump()
        except Exception:
            clean_args = args   # use raw if validation fails

        return {
            "intent":     name,
            "params":     clean_args,
            "confidence": 0.93,
        }

    except Exception as e:
        print(f"[marin_fier] qwen stage failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TOOL EXECUTOR — called by marin.py after classify() returns a tool intent
# ══════════════════════════════════════════════════════════════════════════════

def execute_tool(intent: str, params: dict) -> str | None:
    """
    Run the StructuredTool for the given intent.
    Returns the tool's context string (what it did), or None if not a tool.
    marin.py injects this context into Marin's LLM prompt.
    """
    import datetime
    if intent not in _TOOL_MAP:
        return None
    try:
        result = _TOOL_MAP[intent].invoke(params)
        # Log tool execution to cmd_log so terminal panel shows it
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        param_str = json.dumps(params)
        _cmd_log.append({
            "cmd": f"[tool:{intent}] {param_str}",
            "allowed": True,
            "output": (result or "(done)")[:500],
            "ts": ts,
        })
        if len(_cmd_log) > 100:
            _cmd_log.pop(0)
        return result
    except Exception as e:
        return f"Tool {intent} failed: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# VIBE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def _detect_vibe(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in ["love","miss","cute","hug","kiss","mwah","sweetheart","ummaah"]):
        return "lovely"
    if any(w in lower for w in ["tease","hehe","playful","naughty"]):
        return "flirty"
    if any(w in lower for w in ["hate","mad","angry","fuck","ugh","damn"]):
        return "angry"
    if any(w in lower for w in ["sad","cry","lonely","down","depressed"]):
        return "sad"
    if any(w in lower for w in ["excited","omg","yay","wow","!!!"]):
        return "excited"
    return "neutral"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

_KNOWN_TOOLS = set(_TOOL_MAP.keys()) | {"run_all_tools"}

def classify(text: str) -> dict:
    """
    Two-stage classification.
    Returns: {intent, params, user_vibe, confidence, _tool_ack}

    _tool_ack is always None here.
    Caller (marin.py) calls execute_tool(intent, params) separately
    so it can build the LLM context string.
    """
    # Stage 1 — regex (fast, no model call)
    result = _regex_stage(text)

    # Stage 2 — qwen with StructuredTool binding
    if result is None:
        result = _llm_stage(text)

    # Absolute fallback
    if result is None:
        result = {"intent": "chat", "params": {}, "confidence": 0.0}

    # Unknown intent → chat
    if result["intent"] not in _KNOWN_TOOLS:
        result["intent"] = "chat"

    result["user_vibe"] = _detect_vibe(text)
    result["_tool_ack"] = None

    print(f"[marin_fier] intent={result['intent']}  "
          f"params={result['params']}  "
          f"vibe={result['user_vibe']}  "
          f"conf={result['confidence']:.2f}")
    return result