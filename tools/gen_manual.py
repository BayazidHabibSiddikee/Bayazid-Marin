#!/usr/bin/env python3
"""Generate tools_manual.pdf documenting all CLI tools."""

from pathlib import Path
from fpdf import FPDF

class Manual(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(180, 80, 120)
        self.cell(0, 8, "Marin Tools - CLI Reference", align="C")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(200, 30, 90)
        self.ln(4)
        self.cell(0, 10, title)
        self.ln(8)
        self.set_draw_color(200, 30, 90)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def tool_entry(self, name, cli, desc, example, note=""):
        self.set_font("Courier", "B", 10)
        self.set_text_color(40, 40, 40)
        self.cell(0, 7, f"$ {cli}")
        self.ln(8)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, desc)
        self.ln(2)
        self.set_font("Courier", "", 8)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 5, f"Example:  {example}")
        self.ln(4)
        if note:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(180, 100, 60)
            self.cell(0, 5, f"Note: {note}")
            self.ln(6)
        self.ln(3)


pdf = Manual(orientation="P", unit="mm", format="A4")
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font("Helvetica", "B", 24)
pdf.set_text_color(200, 30, 90)
pdf.ln(20)
pdf.cell(0, 15, "Marin Tools", align="C")
pdf.ln(12)
pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 8, "CLI Reference Manual", align="C")
pdf.ln(20)

# Intro
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(0, 5,
    "All tools live in tools/ and share a consistent --flag interface. "
    "They print a banner, show results on the terminal, and may open a "
    "GUI window (chart, price tracker, game). Tools are launched by "
    "marin_fier.py when Marin detects the matching intent, or can be "
    "run directly from the command line."
)

# STOCK
pdf.section("1. stock.py  -  Stock Price + 30-Day Chart")
pdf.tool_entry(
    "stock.py",
    "python3 tools/stock.py --ticker AAPL",
    "Fetch live price and 30-day chart for a stock ticker. "
    "Opens a matplotlib chart window and prints a price table to terminal "
    "with dates and closing prices.",
    "python3 tools/stock.py --ticker META",
    "Use --company \"Company Name\" to auto-resolve the ticker via Yahoo search."
)

# CRYPTO
pdf.section("2. crypto.py  -  Live Crypto Price Tracker")
pdf.tool_entry(
    "crypto.py",
    "python3 tools/crypto.py --coin ethereum",
    "Open a tkinter window showing live USD price of the given "
    "cryptocurrency, updating every second via CoinGecko API.",
    "python3 tools/crypto.py --coin bitcoin",
    "Defaults to 'bitcoin' if --coin is omitted."
)

# NEWS
pdf.section("3. news.py  -  Open News Website")
pdf.tool_entry(
    "news.py",
    "python3 tools/news.py --source BBC",
    "Open a news website in the default browser. "
    "Supported sources: BBC, AlJazeera, NDTV.",
    "python3 tools/news.py --source AlJazeera",
    "Defaults to BBC if --source is omitted."
)

# ALARM
pdf.section("4. alarm.py  -  Set an Alarm")
pdf.tool_entry(
    "alarm.py",
    "python3 tools/alarm.py --time \"05:00\"",
    "Set a clock alarm. Accepts HH:MM or HH:MM AM/PM. "
    "Runs in the background (forks to child process) and beeps at the "
    "target time using alarm.wav (via pygame mixer).",
    'python3 tools/alarm.py --time "7:30 AM"',
    "The process daemonizes so your terminal stays free."
)

# TIMER
pdf.section("5. timer.py  -  Countdown Timer")
pdf.tool_entry(
    "timer.py",
    "python3 tools/timer.py --duration 300",
    "Start a countdown timer. Duration is in seconds. "
    "Beeps when time runs up using alarm.wav.",
    "python3 tools/timer.py --duration 90",
    "300 seconds = 5 minutes, 90 seconds = 1:30."
)

# TRANSLATE
pdf.section("6. translate.py  -  Dictionary Translator")
pdf.tool_entry(
    "translate.py",
    "python3 tools/translate.py --text \"I love you\" --to bn",
    "Translate English text to a target language. "
    "Prints the result to terminal and speaks it aloud via TTS. "
    "Accepts language names (bangla, french) or codes (bn, fr).",
    'python3 tools/translate.py --text "Good morning" --to french',
    "Uses deep_translator (GoogleTranslate) with googletrans fallback."
)

# IMAGE
pdf.section("7. image.py  -  Screenshot to Stencil Art")
pdf.tool_entry(
    "image.py",
    "python3 tools/image.py --prompt \"Starry Night\"",
    "Capture a screen region (uses maim) and convert it to a black-and-white "
    "stencil image displayed in a window.",
    'python3 tools/image.py --prompt "My Desktop"',
    "Requires maim: sudo apt install maim"
)

# GAMES
pdf.section("8. Games  -  TicTacToe / Connect4 / WordGame")
tools_games = [
    ("tictactoe.py", "python3 tools/tictactoe.py", "Classic Tic Tac Toe game."),
    ("connect4.py", "python3 tools/connect4.py [--two]", "Connect Four. --two for 2-player mode."),
    ("wordgame.py", "python3 tools/wordgame.py", "Word scramble / word game."),
]
for name, cli, desc in tools_games:
    pdf.set_font("Courier", "B", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 6, f"  {cli}")
    pdf.ln(7)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 5, desc)
    pdf.ln(2)

# OTHER
pdf.section("9. Other Tools")
tools_other = [
    ("draw_me.py", "python3 tools/draw_me.py", "Take a webcam photo and render it as turtle drawing."),
    ("email_tool.py", "python3 tools/email_tool.py", "Interactive email composer (GUI)."),
    ("bangla.py", "python3 tools/bangla.py", "Bangla language utility."),
    ("vpa.py", "python3 tools/vpa.py", "VPA utility tool."),
    ("draw.py", "python3 tools/draw.py", "Drawing utility."),
]
for name, cli, desc in tools_other:
    pdf.set_font("Courier", "B", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 6, f"  {cli}")
    pdf.ln(7)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 5, desc)
    pdf.ln(2)

# INTEGRATION
pdf.section("10. Integration with Marin")
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(0, 5,
    "marin_fier.py is the two-stage intent classifier that maps user messages "
    "to tool calls. Stage 1 uses regex for obvious patterns (\"stock\", \"alarm\", "
    "\"ticker TSLA\"). Stage 2 uses qwen2.5:0.5b with LangChain StructuredTool "
    "bindings for fuzzy/ambiguous requests. The execute_tool() function runs "
    "the matching tool via subprocess.Popen in a detached session, so each "
    "tool runs as its own independent process."
)
pdf.ln(4)
pdf.set_font("Courier", "", 8)
pdf.set_text_color(100, 100, 100)
pdf.multi_cell(0, 5,
    "  Intent detection order:  crypto > stock > bare ticker > alarm >\n"
    "  timer > news > email > tictactoe > connect4 > wordgame >\n"
    "  draw_me > screenshot > run_command > qwen-llm fallback"
)

# Save
out = Path(__file__).resolve().parent / "tools_manual.pdf"
pdf.output(str(out))
print(f"Generated: {out}")
