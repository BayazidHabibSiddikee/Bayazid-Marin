import cv2
from PIL import Image, ImageTk
import tkinter as tk

# Settings
scale = 0.8
threshold = 128

# Capture camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error opening camera")
    exit()

ret, frame = cap.read()
cap.release()

if not ret:
    print("Error capturing frame")
    exit()

# Convert to PIL grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
img = Image.fromarray(gray)

# Resize
img = img.resize((int(img.width * scale), int(img.height * scale)))

# Tkinter window
root = tk.Tk()
root.title("Real-Time Drawn Image")
canvas = tk.Canvas(root, width=img.width, height=img.height, bg="white")
canvas.pack()

def draw_image():
    for y in range(img.height):
        for x in range(img.width):
            if img.getpixel((x, y)) < threshold:
                canvas.create_line(x, y, x+1, y, fill="black")
        root.update()  # refresh row-by-row

# Start drawing
root.after(10, draw_image)
root.mainloop()
