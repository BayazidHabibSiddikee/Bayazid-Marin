# config.py — Bayazid HS-02 shared constants

# ── APP LAUNCHER ──────────────────────────────────────────────────────────
APPS = {
    "chrome":           {"win": "start chrome",        "linux": "google-chrome",         "darwin": "open -a 'Google Chrome'"},
    "firefox":          {"win": "start firefox",       "linux": "firefox",               "darwin": "open -a Firefox"},
    "vscode":           {"win": "code",                "linux": "code",                  "darwin": "code"},
    "vs code":          {"win": "code",                "linux": "code",                  "darwin": "code"},
    "terminal":         {"win": "cmd",                 "linux": "gnome-terminal",        "darwin": "open -a Terminal"},
    "calculator":       {"win": "calc",                "linux": "gnome-calculator",      "darwin": "open -a Calculator"},
    "notepad":          {"win": "notepad",             "linux": "gedit",                 "darwin": "open -a TextEdit"},
    "file explorer":    {"win": "explorer",            "linux": "nautilus",              "darwin": "open -a Finder"},
    "task manager":     {"win": "taskmgr",             "linux": "gnome-system-monitor",  "darwin": "open -a 'Activity Monitor'"},
}

WEB_APPS = {
    "claude":        "https://claude.ai",
    "chatgpt":       "https://chat.openai.com",
    "gemini":        "https://gemini.google.com",
    "youtube":       "https://www.youtube.com",
    "github":        "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "google":        "https://www.google.com",
    "gmail":         "https://mail.google.com",
    "drive":         "https://drive.google.com",
    "notion":        "https://www.notion.so",
    "telegram":      "https://web.telegram.org/k/",
    "discord":       "https://discord.com/app",
    "reddit":        "https://www.reddit.com",
}

# ── MODEL CONFIG ──────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemma4:31b-cloud"
FAST_MODEL    = "qwen2.5:0.5b"   # For quick tasks

# ── SESSION CONFIG ────────────────────────────────────────────────────────
POMODORO_WORK_MINUTES    = 25
POMODORO_BREAK_MINUTES   = 5
POMODORO_LONG_BREAK      = 15
MEMORY_MAX_MESSAGES      = 50

# ── SERVER ────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 6969
UPLOAD_FOLDER = "static/uploads"