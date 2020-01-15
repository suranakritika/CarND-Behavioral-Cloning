# import argparse, os
import cv2
import csv  
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, MaxPooling2D, Lambda, Conv2D
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
# %matplotlib inline
img = os.listdir("opts/data/IMG")

def reader(path):
    lines = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        return lines

def process_img(lines, row, correction):
    images = []
    measurements = []
    for line in lines:
        sourcepath = line[row]
        filename = sourcepath.split('\\')[-1]
        current_path = 'opts/data/IMG/' + filename
        image = mpimg.imread(current_path)
        images.append(image)
        measurement = (float(line[3]) + correction)
        measurements.append(measurement)
    return images, measurements

lines = reader("opts/data/driving_log.csv")
image_center, measure_center = process_img(lines, 0, 0.0)
image_left, measure_left = process_img(lines, 1, 0.2)
image_right, measure_right = process_img(lines, 2, -0.2)

cam_images = []
steering_measure = []
cam_images.extend(image_center)
cam_images.extend(image_left)
cam_images.extend(image_right)
steering_measure.extend(measure_center)
steering_measure.extend(measure_left)
steering_measure.extend(measure_right)

augmented_image, augmented_measurement = [], []
for image, measurement in zip(cam_images, steering_measure):
    augmented_image.append(image)
    augmented_measurement.append(measurement)
    augmented_image.append(cv2.flip(image, 1))
    augmented_measurement.append(measurement * -1.0)

X_train = np.array(augmented_image)
y_train = np.array(augmented_measurement)
print(X_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160, 320, 3)))
model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()
# model.compile(loss = 'mse', optimizer = 'adam')
# model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 10)
# model.save('model.h5')
# exit()
