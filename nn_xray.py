# -*- coding: utf-8 -*-
"""
Title:
Author: Erik Walcz
Created on Thu Apr 29 10:58:12 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import time
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

NAME = "XRAY-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

DATADIR = "data/"
CATEGORIES = ["hand", "skull"]

IMG_SIZE = 50

training_data = []

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
create_training_data()

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])
    
    
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# =============================================================================
# pickle_out = open("X.pickle","wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()
# 
# pickle_out = open("y.pickle","wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()
# 
# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)
# 
# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)
# =============================================================================

X=np.array(X/255.0)
y=np.array(y)

model = Sequential()

model.add(Conv2D(20, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(20, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=50, validation_split=0.3, callbacks=[tensorboard])