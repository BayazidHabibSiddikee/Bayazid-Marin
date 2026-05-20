#!/usr/bin/env python3
# tools/news.py — runs as its own process
# Opens a news website in the default browser.
# Usage: python news.py --source BBC

import sys, argparse
import webbrowser
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.tts import speak_male as talk2

SOURCES = {
    "BBC":        "https://www.bbc.com/news",
    "AlJazeera":  "https://www.aljazeera.com",
    "NDTV":       "https://www.ndtv.com",
}


def open_news(source: str = "BBC"):
    url = SOURCES.get(source, SOURCES["BBC"])
    print(f"\u2192 Fetching latest news from [{source}]")
    try:
        talk2(f"Opening news from {source}.")
        webbrowser.open(url)
    except Exception:
        talk2("Could not open news.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Open news website")
    parser.add_argument('--source', type=str, default="BBC",
                        choices=list(SOURCES.keys()),
                        help="News source: BBC, AlJazeera, NDTV")
    args = parser.parse_args()
    open_news(args.source)
