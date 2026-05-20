"""
TOOLS.md — Marin Tool Registry
All tools are defined as LangChain @tool functions and used by qwen2.5:0.5b
for AI-powered intent classification and parameter extraction.

TOOL CATEGORIES:
- system: alarm, timer, email, news
- finance: crypto_price, stock_info
- games: tictactoe, connect4, wordgame
- image: draw_me, take_screenshot
"""

from datetime import datetime

# ── System Tools ────────────────────────────────────────────────────────────────

def set_alarm(time_str: str) -> str:
    """
    Set a countdown alarm. Marin speaks the alarm message when it goes off.
    Args:
        time_str: Natural time like '7:30 a.m.', '2:15 p.m.', '23:00', '7:00 AM'.
    """
    import subprocess, sys, os
    from pathlib import Path
    script = Path(__file__).parent / "alarm.py"
    subprocess.Popen(
        [sys.executable, str(script), time_str],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return f"Alarm set for {time_str}. I'll wake you up when it's time~ ⏰"


def set_timer(duration: str) -> str:
    """
    Set a countdown timer.
    Args:
        duration: Natural duration like '30 minutes', '1 hour', '2 hours 30 minutes'.
    """
    import subprocess, sys
    from pathlib import Path
    script = Path(__file__).parent / "timer.py"
    subprocess.Popen(
        [sys.executable, str(script), duration],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return f"Timer started for {duration}! I'll let you know when it's up~ ⏱️"


def send_email(to: str = "", subject: str = "", body: str = "") -> str:
    """
    Open the email composition tool.
    Args:
        to: Recipient name or email address (optional — tool asks interactively).
        subject: Email subject line (optional).
        body: Email body text (optional).
    """
    import subprocess, sys
    from pathlib import Path
    script = Path(__file__).parent / "email_tool.py"
    subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return f"Opening email composer~ Let's write that email to {to or 'your contact'}! ✉️"


def open_news() -> str:
    """Open the latest news in the default browser."""
    import subprocess, sys
    from pathlib import Path
    script = Path(__file__).parent / "news.py"
    subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return "Opening today's news for you~ 📰"


# ── Finance Tools ───────────────────────────────────────────────────────────────

def get_crypto_price(coin: str) -> str:
    """
    Fetch live cryptocurrency price from CoinGecko.
    Args:
        coin: Coin name or symbol like 'bitcoin', 'ethereum', 'solana', 'dogecoin'.
    """
    import subprocess, sys
    from pathlib import Path
    script = Path(__file__).parent / "crypto.py"
    if not coin:
        coin = "bitcoin"
    subprocess.Popen(
        [sys.executable, str(script), coin],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return f"Pulling up {coin} price chart~ 📈"


def get_stock_info(company: str) -> str:
    """
    Fetch stock price and 30-day chart for a company.
    Args:
        company: Company name like 'Apple', 'Tesla', 'Microsoft', 'Google' or ticker like 'AAPL'.
    """
    import subprocess, sys
    from pathlib import Path
    script = Path(__file__).parent / "stock.py"
    subprocess.Popen(
        [sys.executable, str(script), company],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return f"Looking up stock info for {company}! 📊"


# ── Games ────────────────────────────────────────────────────────────────────────

def play_tictactoe() -> str:
    """Launch the Tic Tac Toe game in a new window."""
    import subprocess, sys
    from pathlib import Path
    script = Path(__file__).parent / "tictactoe.py"
    subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return "Tic Tac Toe is launching~ 🎮 Don't cry when I win! Hehehe~ ♡"


def play_connect4(mode: str = "computer") -> str:
    """
    Launch Connect Four.
    Args:
        mode: 'computer' for AI opponent, 'two' for 2-player mode.
    """
    import subprocess, sys
    from pathlib import Path
    script = Path(__file__).parent / "connect4.py"
    args = [] if mode == "computer" else ["--two"]
    subprocess.Popen(
        [sys.executable, str(script)] + args,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return f"Connect Four is opening~ 🔴🟡 Get ready to lose! ({mode} mode)"


def play_wordgame() -> str:
    """Launch the Guess-the-Word game."""
    import subprocess, sys
    from pathlib import Path
    script = Path(__file__).parent / "wordgame.py"
    subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return "Word game is starting~ 🔤 Let's see how big your vocabulary is!"


# ── Image Tools ────────────────────────────────────────────────────────────────

def draw_me() -> str:
    """
    Take a camera photo and draw it as a stencil with turtle animation.
    Requires: webcam (cv2), PIL, turtle.
    """
    import subprocess, sys
    from pathlib import Path
    script = Path(__file__).parent / "draw_me.py"
    subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return "Taking your photo and drawing it~ 🎨 Give me a second!"


def take_screenshot(output_path: str = "") -> str:
    """
    Take a screenshot and save it.
    Args:
        output_path: Optional filename (default: screenshot_TIMESTAMP.png).
    """
    import subprocess
    if not output_path:
        from datetime import datetime
        output_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    subprocess.run(["scrot", output_path], capture_output=True)
    return f"Screenshot saved as {output_path}! 📸"