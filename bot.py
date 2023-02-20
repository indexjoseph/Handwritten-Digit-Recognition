import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K
#  to split the data of training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


batch_size = 128
num_classes = 10
epochs = 5
#x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
##input_shape = (28, 28, 1)
# conversion of class vectors to matrices of  binary class
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255

x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)

#model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))

model = Sequential()
model.add(Flatten())
model.add(Dense(batch_size, activation=tf.nn.relu))
model.add(Dense(batch_size, activation=tf.nn.relu))
model.add(Dense(num_classes, activation=tf.nn.softmax))
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=epochs)
print("The model has successfully trained")
model.save('mnist.h5')
print("Saving the bot as mnist.h5")

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
