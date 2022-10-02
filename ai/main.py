import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import models
from keras import layers
from keras import preprocessing
from keras import optimizers
from sklearn.metrics import classification_report

import os
import cv2

import dill
dill.dump_session('notebook_env.db')

dill.load_session('notebook_env.db')


labels = ['Blight', 'Healthy', 'Mildew', 'Rust', 'Spot']
img_size = 224
def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

train = get_data('datasets/Train')
val = get_data('datasets/Validation')


x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

datagen = preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


model = models.Sequential()

model.add(layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=(224,224,3), kernel_regularizer =tf.keras.regularizers.l2(l=0.01)))
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.25))

""""
model.add(layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer =tf.keras.regularizers.l2( l=0.01)))
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.25))
"""

model.add(layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer =tf.keras.regularizers.l2(l=0.01)))
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
initializer = tf.keras.initializers.HeNormal()
model.add(layers.Dense(128, kernel_initializer=initializer))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, kernel_initializer=initializer))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation="softmax"))

model.summary()

opt = optimizers.Adam(lr=0.001)
model.compile(optimizer = opt, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
              metrics = ['accuracy'])

history = model.fit(x_train,y_train,epochs = 50 , validation_data = (x_val, y_val))

predictions = model.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = labels))