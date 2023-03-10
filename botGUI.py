############################
# Authors: Brady Smith, Joseph Oledeji, Andrew Monroe
# Date: 2/24/2023
# Description: Simple pygame program that allows the user to draw a digit and then
# uses the trained neural network to recognize the digit.
############################
import os
import cv2  # pip install opencv-python
import numpy as np  # pip install numpy
import pygame  # pip install pygame
import sys
import tensorflow as tf  # pip install tensorflow
from pygame.locals import *

WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDARYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
IMAGESAVE = False
PREDICT = True
MODEL = tf.keras.models.load_model("mnist.model")  # Load the model.
PATH = "./test_digits"
count = 1

LABELS = {
    0: "Zero", 1: "One",
    2: "Two", 3: "Three",
    4: "Four", 5: "Five",
    6: "Six", 7: "Seven",
    8: "Eight", 9: "Nine"}

pygame.init()

FONT = pygame.font.SysFont("comicsansms", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption("Digit Board")

iswriting = False

number_xcord = []
number_ycord = []

image_cnt = 0

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        # Draws the digit with the mouse.
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        # Triggers the event that the mouse is drawing.
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        # Triggers the event that the mouse is no longer drawing.
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            # trys to get the bounding box of the digit.
            try:
                rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINC, 0), min(WINDOWSIZEX,
                                                                                    number_xcord[-1] + BOUNDARYINC)
                rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARYINC, 0), min(number_ycord[-1] + BOUNDARYINC,
                                                                                    WINDOWSIZEX)
            except IndexError:
                continue

            number_xcord = []
            number_ycord = []

            # maps the saved pixel array to a numpy array that the model can use.
            ing_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(
                np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_cnt += 1

            # uses the model to predict the digit.
            if PREDICT:
                image = cv2.resize(ing_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255

                np.save(os.path.join(PATH, f'image{count}.npy'), image)
                count += 1

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_min_y

                DISPLAYSURF.blit(textSurface, textRecObj)
        # If n is pressed the screen is cleared
        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

        # run pygame
        pygame.display.update()
