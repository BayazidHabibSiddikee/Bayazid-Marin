#!/usr/bin/env python3
# marin_fier.py — Ultra-fast Regex classifier (0 RAM usage)
import re

def classify(text: str) -> dict:
    """Fast classification using Regex. Returns dict with intent, user_vibe."""
    lower = text.lower()
    intent = "chat"
    user_vibe = "neutral"

    # ── GAMES ──────────────────────────────────────────────────────────────────
    if re.search(r'tiktaktoe|tic\s*tac\s*toe|tictactoe', lower):
        intent = "tictactoe_start"          # ← was "play_tiktaktoe" (key mismatch!)
    elif re.search(r'connect\s*4|connect\s*four|connect4', lower):
        intent = "play_connect4"
    elif re.search(r'word\s*game|wordgame|scramble', lower):
        intent = "play_wordgame"

    # ── TOOLS ──────────────────────────────────────────────────────────────────
    elif re.search(r'\bset\s+(an?\s+)?alarm\b|alarm\s+for', lower):
        intent = "alarm"
    elif re.search(r'\btimer\b.*\b(\d+|minute|second|hour)\b|\b(\d+|minute|second|hour)\b.*\btimer\b', lower):
        intent = "timer"
    elif re.search(r'\b(crypto|bitcoin|ethereum|bnb|solana|dogecoin|coin\s*price)\b', lower):
        intent = "crypto"
    elif re.search(r'\b(stock|share\s*price|market\s*price|ticker)\b', lower):
        intent = "stock"
    elif re.search(r'\b(news|headlines|latest\s*news|open\s*news)\b', lower):
        intent = "news"
    elif re.search(r'\b(send\s*(an?\s*)?email|email\s+to|compose\s*(an?\s*)?email|write\s*(an?\s*)?mail)\b', lower):
        intent = "email"

    # ── IMAGE / DRAW ────────────────────────────────────────────────────────────
    elif re.search(r'\b(draw|generate\s*image|create\s*image|make\s*a\s*picture|paint)\b', lower):
        intent = "image_gen"

    # ── USER VIBES ─────────────────────────────────────────────────────────────
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
    elif intent.startswith("play_") or intent in ("tictactoe_start",):
        user_vibe = "playful"

    return {"intent": intent, "user_vibe": user_vibe, "confidence": 0.99}