import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D

lines = []

data_folder = '/Users/daard/Documents/CarND/data-p3/'
with open(data_folder + 'driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for i, line in enumerate(lines):
    if i > 0:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = data_folder + 'IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurements.append(float(line[3]))

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
# model.add(Convolution2D(32, 3, 3, input_shape=(160, 320, 3)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')