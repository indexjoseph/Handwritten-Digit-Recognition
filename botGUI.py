from keras.models import load_model
from tkinter import *
import tkinter as tk
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

model = load_model('mnist.h5')


def predict_digit(img):
    # resize image to 28x28 pixels
    img = img.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    draw(img)
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    # predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


def draw(n):
    plt.imshow(n, cmap=plt.cm.binary)
    plt.show()


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=200, height=200, bg="black", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        self.canvas.pack()
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

        # pil image
        self.image1 = Image.new("RGB", (300, 300), "black")
        self.draw = ImageDraw.Draw(self.image1)

    def clear_all(self):
        self.canvas.delete("all")
        self.image1.close()
        self.image1 = Image.new("RGB", (300, 300), "black")
        self.draw = ImageDraw.Draw(self.image1)

    def classify_handwriting(self):
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        """im = ImageGrab.grab().crop((x, y, x1, y1))"""
        im = self.image1.crop((x, y, x1, y1))
        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 4
        self.canvas.create_line(self.x - r, self.y - r, self.x + r, self.y + r, fill='white', width=5)
        self.draw.line([self.x - r, self.y - r, self.x + r, self.y + r], "white")


app = App()
mainloop()
