import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()    # Load the MNIST dataset

x_train = tf.keras.utils.normalize(x_train, axis=1)    # Normalize the data
x_test = tf.keras.utils.normalize(x_test, axis=1)    # Normalize the data

model = tf.keras.models.Sequential()    # Create a model
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))    # Flatten the input
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))    # Add a dense layer with 128 neurons
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))    # Add a dense layer with 128 neurons
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))    # Add a dense layer with 10 neurons

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])    # Compile the model
model.fit(x_train, y_train, epochs=3)    # Fit the model

