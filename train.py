import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train/255.0
X_test = X_test/255.0

model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(units=128, activation="relu"),
    Dense(units=64, activation="relu"),
    Dense(units=10, activation="linear")
])

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), 
              optimizer=Adam(learning_rate=0.001))

print("Hold on will take a moment")
model.fit(X_train, y_train, epochs=20, verbose=0)
model.save("mnist_digit_model.keras")

# Evaluate
logits = model.predict(X_test, verbose=0)
pred = np.argmax(logits, axis=1)
print(f"Test Set Accuracy: {np.mean(pred == y_test) * 100:.2f}%\n")
