from turtle import *
from time import sleep
from tkinter import messagebox
import random
import sys, os
from PIL import Image
import pyautogui

class ImageDrawer:
    def __init__(self, path=None, scale=1, dot_size=2):
        self.scale = scale
        self.dot_size = dot_size
        self.path = path or self._get_image_path()
        self.img = self._load_and_process_image()
        self.screen = Screen()
        self._setup_screen()
        self.colors = [
            (255, 182, 193),   # Light pink
            (173, 216, 230),   # Baby blue
            (152, 251, 152),   # Mint green
            (230, 230, 250),   # Lavender
            (255, 218, 185),   # Peach
            (255, 255, 224),   # Pale yellow
            (135, 206, 235),   # Sky blue
            (240, 128, 128),   # Light coral
            (255, 253, 208),   # Cream
            (200, 162, 200),   # Lilac
            (211, 211, 211),   # Light gray
            (245, 245, 220),   # Beige
            (255, 255, 240),   # Ivory
            (255, 218, 185),   # Pastel peach
            (175, 238, 238),   # Soft aqua
            (176, 224, 230),   # Powder blue
        ]
        self.bright_colors = ['white', 'yellow', 'light yellow']
        self.dark_colors = ['black', 'darkblue', 'purple']

    def _get_image_path(self):
        try:
            return pyautogui.screenshot()
        except:
            return r"E:\YT\images\remove-photos-background-removed.png"

    def _load_and_process_image(self):
        try:
            if isinstance(self.path, str):
                img = Image.open(self.path)
            elif isinstance(self.path, Image.Image):
                img = self.path
            else:
                raise ValueError("Invalid path or image type")
            img = img.convert("L")
            img = img.convert("RGB")
            img = img.resize((int(img.width * self.scale), int(img.height * self.scale)))
            return img
        except FileNotFoundError:
            print(f"Error: Image file not found at {self.path}. Please check the path and file name.")
            sys.exit()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            sys.exit()

    def _setup_screen(self):
        self.screen.colormode(255)
        self.screen.setup(int(self.img.width * self.dot_size/2), int(self.img.height * self.dot_size/2), 100, 50)
        self.screen.bgcolor('light yellow')
        self.screen.tracer(True)

    def draw_random_elements(self):
        i = int(self.img.width * self.dot_size / 2)
        j = int(self.img.height * self.dot_size / 2)
        shape('turtle')
        c = Turtle('turtle')
        c.shapesize(1)
        c.color("black", "red")
        for _ in range(1):
            for k in range(-i, i, 20):
                c.shape(random.choice(['turtle', 'circle', 'arrow', 'square', 'triangle', 'classic']))
                shape(random.choice(['turtle', 'circle', 'arrow', 'square', 'triangle', 'classic']))
                speed(0.0001)
                color("black", random.choice(self.colors))
                c.speed(0.0001)
                c.color("black", random.choice(self.colors))
                c.goto(k, random.randint(-j, j))
                pensize(random.randint(1, 4))
                pencolor(random.choice(self.colors))
                goto(k, random.randint(-j, j))
                sleep(0.005)
                c.color("black", random.choice(self.colors))
                c.dot(10)
            c.goto(100, 0)
            sleep(0.5)
        c.hideturtle()
        hideturtle()

    def draw_image(self):
        x_offset = -self.img.width / 2
        y_offset = self.img.height / 2
        self.screen.colormode(255)
        self.screen.tracer(False)
        t = Turtle()
        t.up()
        t.speed(0)
        try:
            for y in range(self.img.height):
                for x in range(self.img.width):
                    r, g, b = self.img.getpixel((x, y))
                    draw_x = (x + x_offset) * self.dot_size
                    draw_y = (y_offset - y) * self.dot_size
                    brightness = r
                    color = random.choice(self.bright_colors) if brightness > 127 else random.choice(self.dark_colors)
                    t.goto(draw_x, draw_y)
                    t.dot(self.dot_size, color)
            t.hideturtle()
            self.screen.tracer(True)
            self.screen.update()
            messagebox.showinfo("Info", "Black and white image drawing complete!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            sys.exit()

    def run(self):
        self.draw_random_elements()
        self.draw_image()
        done()
        try:
            bye()
        except:
            pass

if __name__ == "__main__":
    drawer = ImageDrawer()
    drawer.run()