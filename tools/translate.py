#!/usr/bin/env python3
# tools/translate.py — Turtle Dictionary Translator, runs as its own process
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Suppress ALSA noise
_dn = os.open(os.devnull, os.O_WRONLY)
_se = os.dup(2)
os.dup2(_dn, 2)
try:
    import turtle as t
finally:
    os.dup2(_se, 2)
    os.close(_se)
    os.close(_dn)

import pyttsx3

LANG_CODES = {
    "english":  "en",
    "chinese":  "zh",
    "spanish":  "es",
    "french":   "fr",
    "japanese": "ja",
    "portuguese":"pt",
    "russian":  "ru",
    "korean":   "ko",
    "german":   "de",
    "italian":  "it",
    "bangla":   "bn",
    "arabic":   "ar",
    "hindi":    "hi",
    "turkish":  "tr",
    "dutch":    "nl",
}


class TurtleTranslator:
    def __init__(self):
        self.screen = t.Screen()
        self.screen.setup(800, 600)
        self.screen.bgcolor('light blue')
        self.screen.title("Dictionary Translator")
        self.writer = t.Turtle()
        self.writer.hideturtle(); self.writer.up(); self.writer.speed(0)
        self._tts_engine = pyttsx3.init()
        self._tts_engine.setProperty('rate', 140)
        self._draw_interface()

    def _draw_interface(self):
        self.writer.clear()
        self.writer.goto(0, 250)
        self.writer.write("Dictionary Translator", align="center", font=("Arial", 24, "bold"))
        self.writer.goto(0, 200)
        self.writer.write("Enter English text to translate", align="center", font=("Arial", 14, "normal"))
        langs = ", ".join(sorted(LANG_CODES.keys()))
        self.writer.goto(0, 150)
        self.writer.write(langs, align="center", font=("Arial", 11, "normal"))
        t.update()

    def _speak(self, text: str, lang_code: str):
        try:
            voices = self._tts_engine.getProperty('voices')
            if voices:
                # Try to find a voice for the target language
                for v in voices:
                    if lang_code[:2] in v.languages or lang_code[:2] in v.id.lower():
                        self._tts_engine.setProperty('voice', v.id)
                        break
            self._tts_engine.say(text)
            self._tts_engine.runAndWait()
        except Exception as e:
            print(f"[TTS] {e}")

    def _translate(self, text: str, dest: str) -> str:
        try:
            from deep_translator import GoogleTranslator
            result = GoogleTranslator(source='en', target=dest).translate(text)
            return result if result else "Translation failed."
        except ImportError:
            try:
                from googletrans import Translator
                result = Translator().translate(text, dest=dest)
                return result.text
            except Exception:
                return "Translation service unavailable. Install deep-translator: pip install deep-translator"
        except Exception as e:
            return f"Error: {e}"

    def run(self):
        while True:
            text = t.textinput("Input", "Enter English text (or 'quit'):")
            if not text or text.lower() == 'quit':
                break
            lang = t.textinput("Language", "Target language:")
            if not lang or lang.lower() == 'quit':
                break
            lang = lang.lower().strip()
            if lang not in LANG_CODES:
                self._draw_interface()
                self.writer.goto(0, 50)
                self.writer.write("Invalid language!", align="center", font=("Arial", 14, "normal"))
                continue
            self._draw_interface()
            self.writer.goto(0, 50)
            self.writer.write(f"English: {text}", align="center", font=("Arial", 14, "normal"))
            translated = self._translate(text, LANG_CODES[lang])
            self.writer.goto(0, 0)
            self.writer.color("blue")
            self.writer.write(f"{lang.title()}: {translated}", align="center", font=("Arial", 16, "bold"))
            self.writer.color("black")
            t.update()
            self._speak(translated, LANG_CODES[lang])
        t.bye()


if __name__ == '__main__':
    TurtleTranslator().run()
