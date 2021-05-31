import numpy as np
import cv2
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from dataloader2 import data_load_generator
d = data_load_generator()

HEIGHT = 128
WIDTH = 128
NUM_OUTPUTS = 1
input_shape = (HEIGHT, WIDTH, 1)

# Import TensorBoard
from tensorflow.keras.callbacks import TensorBoard

# Define Tensorboard as a Keras callback
tensorboard = TensorBoard(
  log_dir='.\logs',
  histogram_freq=1,
  write_images=True
)
keras_callbacks = [
  tensorboard
]

model = Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(Conv2D(64,(3,3), activation = 'relu',padding="same", input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
# Add another:
model.add(Conv2D(64,(3,3), padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Add a softmax layer with 1 output unit
model.add(Dense(NUM_OUTPUTS, activation='sigmoid'))

batch_size = 128
epochs = 30
fileList = d.get_file_name()

testX, testY = d.get_vaildation_data()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
H = model.fit_generator(d.imageLoader(fileList,batch_size),validation_data=(testX, testY), steps_per_epoch=30, epochs=epochs, callbacks=keras_callbacks)
model.save('detectv1.h5')
model.save('detectv1')
