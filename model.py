from keras.models import load_model, Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Cropping2D, Convolution2D
import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#tunable parameter to adjust left/right steering angles
correction = 0.2

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, int(batch_size/6)): #batch_size/6 since we have six images
            batch_samples = samples[offset:offset+int(batch_size/6)]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = './data/IMG/'+batch_sample[0].split('/')[-1]
                left_name = './data/IMG/'+batch_sample[1].split('/')[-1]
                right_name = './data/IMG/'+batch_sample[2].split('/')[-1]

                names = [center_name, left_name, right_name]
                try: 
                    steering_center = float(batch_sample[3])
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction
                except ValueError:
                    continue

                for i, name in names:
                    image = cv2.imread(name)
                    images.append(image)
                    if i==0:
                        steering = steering_center
                    elif i==1:
                        steering = steering_left
                    elif i==2:
                        steering = steering_right
                    angles.append(steering)

					#Add flipped image
                    image_flipped = np.fliplr(image)
                    angle_flipped = -steering
                    images.append(image_flipped)
                    angles.append(angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,
        input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
	len(train_samples)*6, validation_data=validation_generator, \
	nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

model.save('model.h5')  # creates a HDF5 file 'model.h5'

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

