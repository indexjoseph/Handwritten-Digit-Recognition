############################
# Authors: Brady Smith, Joseph Oledeji, Andrew Monroe
# Date: 2/24/2023
# Description: This program is a digit recognition program that uses a convolutional neural network to recognize digits.
############################
import numpy as np  # pip install numpy
import seaborn as sns; sns.set()  # pip install seaborn
import tensorflow as tf  # pip install tensorflow
import cv2  # pip install opencv-python
import os

model = tf.keras.models.load_model("mnist.model")  # Load the model
count = 1
# looks at all the images in the test_digits folder and predicts what the digit is
while os.path.exists(f"test_digits/image{count}.npy"):
    ing_arr = np.load(f"test_digits/image{count}.npy")
    ax = sns.heatmap(ing_arr, annot=True, fmt="f", cbar=None, xticklabels=False, yticklabels=False)
    image = cv2.resize(ing_arr, (28, 28))
    image = np.pad(image, (10, 10), 'constant', constant_values=0)
    image = cv2.resize(image, (28, 28)) / 255
    prediction = np.argmax(model.predict(image.reshape(1, 28, 28, 1)))
    print(prediction)
    count += 1
