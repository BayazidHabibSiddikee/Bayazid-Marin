#!/usr/bin/env python3
# tools/image.py — takes screenshot and converts to stencil
# Works on Linux via maim + convert
from PIL import Image, ImageDraw
import os, time, subprocess, sys

def capture_and_draw():
    scale = 0.8
    threshold = 128

    # Linux: use maim for screenshot
    screenshot_path = "/tmp/marin_screenshot.png"
    try:
        subprocess.run(["maim", "-s", screenshot_path], check=True)
    except FileNotFoundError:
        print("SPEAK: maim not installed. Run: sudo apt install maim")
        sys.exit(1)

    img = Image.open(screenshot_path).convert("L")
    os.remove(screenshot_path)

    img = img.resize((int(img.width * scale), int(img.height * scale)))

    im = Image.new("RGBA", (img.width, img.height), "white")
    draw = ImageDraw.Draw(im)

    for y in range(img.height):
        for x in range(img.width):
            if img.getpixel((x, y)) < threshold:
                draw.point((x, y), fill="black")

    im.show()

if __name__ == "__main__":
    capture_and_draw()