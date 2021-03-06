import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

samples = []
#data_folder = '/Users/daard/Documents/CarND/data-p3/'
data_folder = '/home/carnd/data/'
with open(data_folder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def preprocess(image):
    # I used nvidia's preprocess function
    # Crop
    cropped  = image[70:135,10:350]
    # Resize
    resized = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA)
    # Blur
    blurred = cv2.GaussianBlur(resized, (5,5), 0)
    # Convert to YUV
    final_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
    return final_image
    

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = data_folder + 'IMG/' + batch_sample[i].split('/')[-1]
                    #nvidia's preprocess image function
                    image = preprocess(cv2.imread(name))
                    images.append(image)
                center_angle = float(batch_sample[3])
                angles.extend([center_angle, center_angle + 0.2, center_angle - 0.2])
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(-angle)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield shuffle(X_train, y_train)


def model():
    model = Sequential()
   # model.add(Cropping2D(((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66, 200, 3), output_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = model()
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=6 * len(train_samples), validation_data=validation_generator,
                    nb_val_samples=6 * len(validation_samples), nb_epoch=30)

model.save('model.h5')

print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

# TODO: generate more data :
# two or three laps of center lane driving
# one lap of recovery driving from the sides
# one lap focusing on driving smoothly around curves
# TODO: visualize


