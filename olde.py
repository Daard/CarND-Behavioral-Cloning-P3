import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D


lines = []

# data_folder = '/Users/daard/Documents/CarND/data-p3/'
data_folder = '/home/carnd/data/'
with open(data_folder + 'driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for i, line in enumerate(lines):
    if i > 0:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = data_folder + 'IMG/' + filename
            image = cv2.imread(current_path)
            images.append(image)
        m = float(line[3])
        measurements.extend([m, m + 0.2, m - 0.2])

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_measurements.append(cv2.flip(image, 1))
    augmented_measurements.append(-measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
model.add(Cropping2D(((70, 25), (0, 0)), input_shape =(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')