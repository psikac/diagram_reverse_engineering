import numpy as np
import pandas as pd
import random

from PIL import Image

import os
import glob

import matplotlib.pyplot as plt
plt.rc('image', cmap='gray')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras import layers
from keras.models import load_model

categories = ['circle', 'square', 'star', 'triangle']

def initialize_model():
    model = Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(im_height, im_width, 1), padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(3,3)))

    model.add(layers.Flatten())
    model.add(layers.Dense(120,activation='relu'))
    model.add(layers.Dense(60,activation='relu'))
    model.add(layers.Dropout(rate=0.5))

    model.add(layers.Dense(4, activation='softmax'))

    return model

im_width = 100
im_height = 100

data = []
target = []

for cat in categories:
    filelist = glob.glob('./shapes/' + cat + '/*.png')
    print(filelist)
    target.extend([cat for _ in filelist])
    #data.extend([np.array(Image.open(fname).resize((im_width,im_height)) for fname in filelist)])
    data.extend([np.array(Image.open(fname).resize((im_width, im_height))) for fname in filelist])

data_array = np.stack(data, axis=0)

X_train, X_test, y_train, y_test = train_test_split(data_array, np.array(target), test_size=0.2, stratify=target)

X_train_norm = np.round((X_train/255), 3).copy()
X_test_norm = np.round((X_test/255), 3).copy()

encoder = LabelEncoder().fit(y_train)

y_train_cat = encoder.transform(y_train)
y_test_cat = encoder.transform(y_test)

y_train_oh = to_categorical(y_train_cat)
y_test_oh = to_categorical(y_test_cat)


X_train_norm = X_train_norm.reshape(-1, 100, 100, 1)
X_test_norm = X_test_norm.reshape(-1, 100, 100, 1)

model = initialize_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
es = EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True)

history = model.fit(X_train_norm, y_train_oh, batch_size=16, epochs=1000, validation_data=(X_test_norm, y_test_oh), callbacks=[es])

model.evaluate(X_test_norm, y_test_oh, verbose=0)
predictions = model.predict(X_test_norm)

fig = plt.figure(figsize=(20,15))
gs = fig.add_gridspec(4, 4)
#
for line in range(0, 3):
    for row in range(0, 3):
        num_image = random.randint(0, X_test_norm.shape[0])
        ax = fig.add_subplot(gs[line, row])
        ax.axis('off')
        ax.set_title("Predicted: " + categories[list(np.round(predictions[num_image])).index(1)])
        ax.imshow(X_test_norm[num_image])
fig.suptitle("Predicted label for the displayed shapes", fontsize=25, x=0.42)

model.save('computer_vision/cnn_model.h5')