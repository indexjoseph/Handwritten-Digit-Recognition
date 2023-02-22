import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Pre processing of the data
x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255

#Reshape the data
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Flatten())  # Turns the 28x28 image into a 1x784 array
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(10, activation='softmax')) # 10 neurons, softmax makes sure all the neurons add up to one answer

model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.save("mnist.model")

loss, acc = model.evaluate(x_test, y_test)

print(loss)
print(acc)
