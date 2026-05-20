#!/usr/bin/env python3
# tools/draw_me.py — takes camera photo and draws it with turtle animation
# Works on Linux via ffmpeg cv2 camera + PIL
import cv2
from PIL import Image, ImageDraw
import tkinter as tk

scale = 0.8
threshold = 128

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening camera")
    exit(1)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Error capturing frame")
    exit(1)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
img = Image.fromarray(gray)

img = img.resize((int(img.width * scale), int(img.height * scale)))

root = tk.Tk()
root.title("Real-Time Drawn Image")
canvas = tk.Canvas(root, width=img.width, height=img.height, bg="white")
canvas.pack()

def draw_image():
    for y in range(img.height):
        for x in range(img.width):
            if img.getpixel((x, y)) < threshold:
                canvas.create_line(x, y, x + 1, y, fill="black")
        root.update()

root.after(10, draw_image)
root.mainloop()