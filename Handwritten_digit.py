import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#
# x_train = x_train.reshape((60000, 28, 28, 1)).astype("float32") / 255
# x_test = x_test.reshape((10000, 28, 28, 1)).astype("float32") / 255
#
# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)
#
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation="relu"))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation="relu"))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation="relu"))
# model.add(layers.Dense(10, activation="softmax"))
#
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#
# model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
#
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(f"Test accuracy: {test_acc}")
#
# model.save("Handwritten.model")

model = tf.keras.models.load_model("Handwritten.model")

num = 1
while os.path.isfile(f"imgs/img{num}.png"):
    try:
        img = cv2.imread(f"imgs/img{num}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"{num}. Number is: {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(e)
    finally:
        num += 1
