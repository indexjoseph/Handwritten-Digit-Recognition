############################
# Authors: Brady Smith, Joseph Oledeji, Andrew Monroe
# Date: 2/24/2023
# Description: This program is a digit recognition program that uses a convolutional neural network to recognize digits.
############################
import tensorflow as tf # pip install tensorflow
import numpy as np  # array manipulator # pip install numpy

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# train data is 60000 images of 28x28 pixels
# test data is 10000 images of 28x28 pixels

# Pre-processing of the data
x_train = x_train.astype(np.float32) / 255  # converts the data to a float32 by column basis and divides it by 255.
x_test = x_test.astype(np.float32) / 255

# Reshape the data
x_train = np.expand_dims(x_train, -1)  # shrinks the dimensions of the array by one dimension.
x_test = np.expand_dims(x_test, -1)

y_train = tf.keras.utils.to_categorical(y_train)  # converts integer array to a binary array.
y_test = tf.keras.utils.to_categorical(y_test)

# stack of layers, processes sequentially.
model = tf.keras.models.Sequential()
# takes in a 2d array of "pixels" and outputs a "pooled" 2d array of pixels.
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# condenses the 2d array made from above statement.
model.add(tf.keras.layers.MaxPool2D((2, 2)))
# does the same as above.
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Flatten())  # Turns the 28x28 image into a 1x784 array.
model.add(tf.keras.layers.Dropout(0.25))  # random values are set to 0 to reduce "over accuracy".
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# 10 neurons, softmax makes sure all the neurons add up to one answer.

model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
# setting characteristics of the model, categorical cross entropy is used to calculate the loss shown in the output
# and accuracy is the metric used in the output.

model.fit(x_train, y_train, epochs=5)
# trains the model
# epochs is the number of times the model is trained (iterations).

model.save("mnist.model")

loss, acc = model.evaluate(x_test, y_test)  # tests the model for accuracy using 10,000 test images.

print(loss)
print(acc)
