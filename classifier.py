#!/usr/bin/env python3
# classifier.py — Study-focused intent classifier (regex, zero overhead)

import re

def classify(text: str) -> dict:
    """
    Classify user message into study-relevant intents.
    Returns: {intent, sub_intent, urgency, confidence}
    """
    lower = text.lower().strip()
    intent = "chat"
    sub_intent = None
    urgency = "normal"

    # ── TIMER INTENTS ──────────────────────────────────────────────────────
    if re.search(r'/timer|start\s+(timer|session|focus)', lower):
        intent = "timer"
        if re.search(r'stop|end|done|finish', lower):
            sub_intent = "stop"
        elif re.search(r'status|check|how long', lower):
            sub_intent = "status"
        elif re.search(r'stats|total|history|summary', lower):
            sub_intent = "stats"
        else:
            sub_intent = "start"

    # ── QUIZ INTENTS ───────────────────────────────────────────────────────
    elif re.search(r'quiz|test me|examine|practice questions?|mcq', lower):
        intent = "quiz"
        if re.search(r'easy|beginner|simple', lower):
            sub_intent = "easy"
        elif re.search(r'hard|advanced|expert|difficult', lower):
            sub_intent = "hard"
        else:
            sub_intent = "medium"

    # ── TEACH INTENTS ──────────────────────────────────────────────────────
    elif re.search(r'teach|explain|what is|how does|how do|why does|clarify|i don\'?t understand', lower):
        intent = "teach"
        if re.search(r'quick|brief|tldr|short|summary', lower):
            sub_intent = "quick"
        elif re.search(r'deep|full|detail|thorough|complete', lower):
            sub_intent = "deep"
        else:
            sub_intent = "standard"

    # ── STUDY PLAN INTENTS ─────────────────────────────────────────────────
    elif re.search(r'study plan|learning plan|roadmap|curriculum|schedule|how to learn', lower):
        intent = "study_plan"

    # ── CODE INTENTS ───────────────────────────────────────────────────────
    elif re.search(r'review\s+my\s+code|check\s+(my\s+)?code|fix\s+(this|my)', lower):
        intent = "code_review"
    elif re.search(r'error|exception|bug|crash|traceback|not working|failed|undefined', lower):
        intent = "debug"
        urgency = "high"
    elif re.search(r'write\s+(a\s+)?(code|function|script|program|class)|generate\s+code|implement', lower):
        intent = "code_gen"

    # ── PRODUCTIVITY INTENTS ───────────────────────────────────────────────
    elif re.search(r'pomodoro|break|take\s+a\s+rest|focus mode', lower):
        intent = "productivity"

    # ── IMAGE / VISION ─────────────────────────────────────────────────────
    elif re.search(r'look at|analyze\s+(this\s+)?(image|photo|picture|diagram|circuit|schematic)', lower):
        intent = "vision"

    # ── URGENCY MODIFIERS ──────────────────────────────────────────────────
    if re.search(r'urgent|asap|immediately|right now|deadline|emergency', lower):
        urgency = "high"

    return {
        "intent": intent,
        "sub_intent": sub_intent,
        "urgency": urgency,
        "confidence": 0.95
    }


def extract_timer_task(text: str) -> str:
    """Pull the task name from a timer start command."""
    patterns = [
        r'/timer\s+start\s+(.+)',
        r'start\s+(?:timer|session|focus)\s+(?:for|on)?\s*(.+)',
        r'focus\s+on\s+(.+)',
        r'working\s+on\s+(.+)',
    ]
    for p in patterns:
        m = re.search(p, text.lower())
        if m:
            return m.group(1).strip().title()
    return ""


def extract_topic(text: str) -> str:
    """Pull the subject/topic from a teach or quiz request."""
    patterns = [
        r'(?:teach|explain|quiz|test)\s+(?:me\s+)?(?:about|on)?\s+(.+)',
        r'what\s+is\s+(.+)',
        r'how\s+does?\s+(.+?)\s+work',
        r'study\s+plan\s+(?:for|on)\s+(.+)',
    ]
    for p in patterns:
        m = re.search(p, text.lower())
        if m:
            topic = m.group(1).strip()
            # Remove trailing question marks, filler words
            topic = re.sub(r'[?!.]+$', '', topic).strip()
            return topic
    return text  # fallback: use full message as topic


def extract_quiz_params(text: str) -> dict:
    """Extract quiz parameters from message."""
    lower = text.lower()
    num = 5  # default
    m = re.search(r'(\d+)\s*(?:question|q)', lower)
    if m:
        num = min(int(m.group(1)), 20)  # cap at 20

    difficulty = "medium"
    if re.search(r'easy|beginner|simple', lower):
        difficulty = "easy"
    elif re.search(r'hard|advanced|expert', lower):
        difficulty = "hard"

    return {"num_questions": num, "difficulty": difficulty}