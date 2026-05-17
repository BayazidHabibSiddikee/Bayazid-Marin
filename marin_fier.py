#!/usr/bin/env python3
# marin_fier.py — Ultra-fast Regex classifier (0 RAM usage)

import re

def classify(text: str) -> dict:
    """Fast classification using Regex. Returns dict with intent, user_vibe."""
    lower = text.lower()
    intent = "chat"
    user_vibe = "neutral"

    # ── INTENTS ────────────────────────────────────────────────────────────
    if re.search(r'tiktaktoe|tic\s*tac\s*toe|tictactoe', lower):
        intent = "play_tiktaktoe"
    elif re.search(r'connect\s*4|connect\s*four|connect4', lower):
        intent = "play_connect4"
    elif re.search(r'word\s*game|wordgame|scramble', lower):
        intent = "play_wordgame"
    elif re.search(r'draw|generate\s*image|create\s*image|make\s*a\s*picture|paint', lower):
        intent = "image_gen"

    # ── USER VIBES ────────────────────────────────────────────────────────
    if any(w in lower for w in ['love', 'miss', 'cute', 'hug', 'kiss', 'mwah', 'ummaah', 'sweetheart']):
        user_vibe = "lovely"
    elif any(w in lower for w in ['tease', 'hehe', 'playful', 'naughty']):
        user_vibe = "flirty"
    elif any(w in lower for w in ['hate', 'mad', 'stupid', 'angry', 'fuck', 'ugh', 'damn']):
        user_vibe = "angry"
    elif any(w in lower for w in ['sad', 'depressed', 'cry', 'lonely', 'down']):
        user_vibe = "sad"
    elif any(w in lower for w in ['excited', 'omg', '!!!', 'yay', 'wow']):
        user_vibe = "excited"
    elif intent.startswith("play_"):
        user_vibe = "playful"

    return {"intent": intent, "user_vibe": user_vibe, "confidence": 0.99}
