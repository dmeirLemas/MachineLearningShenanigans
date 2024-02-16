# Import necessary libraries
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv("data.csv")
data.drop("id", axis=1, inplace=True)

x = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.8, random_state=42
)

input_shape = x.shape[1]

print(input_shape)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation="relu", input_shape=(input_shape,)))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(8, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

history = model.fit(
    x_train,
    y_train,
    epochs=60,
    batch_size=64,
    validation_data=(x_test, y_test),
)

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

model.save("breast_imp.model")

y_pred = model.predict(x_test)

y_pred = y_pred > 0.5

name = ["Benign", "Malignant"]
classification_rep = classification_report(y_test, y_pred, target_names=name)
print("\nClassification Report:")
print(classification_rep)
