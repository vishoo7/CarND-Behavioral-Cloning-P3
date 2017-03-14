import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Cropping2D, Convolution2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def generator(data, batch_size=32, correction=.2):
    num_samples = len(data)
    batch_size = int(batch_size / 6)  # batch_size/6 since we have six images

    while 1:  # Loop forever so the generator never terminates
        shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = './data/IMG/' + batch_sample[0].split('/')[-1]
                left_name = './data/IMG/' + batch_sample[1].split('/')[-1]
                right_name = './data/IMG/' + batch_sample[2].split('/')[-1]

                names = [center_name, left_name, right_name]
                steering_center = batch_sample[3]

                for i, name in enumerate(names):
                    image = cv2.imread(name)

                    if i == 0:  # center
                        steering = steering_center
                    elif i == 1:  # left
                        steering = steering_center + correction
                    elif i == 2:  # right
                        steering = steering_center - correction

                    # Add regular image
                    angles.append(steering)
                    images.append(image)

                    # Add flipped image
                    image_flipped = np.fliplr(image)
                    angles.append(steering * -1.0)
                    images.append(image_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def create_model():
    model = Sequential()
    # normalization
    model.add(Lambda(lambda x: x / 255.0 - 0.5,
                     input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(Dense(50))
    model.add(Dropout(.5))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


def parse_csv(filename):
    lines = []
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            try:
                line[3] = float(line[3])
            except ValueError:
                continue

            # only use first 4 columns
            lines.append(line[:4])
    return lines


if __name__ == "__main__":
    samples = parse_csv('./data/driving_log.csv')
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # tunable parameter to adjust left/right steering angles
    steering_correction = 0.22

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32, correction=steering_correction)
    validation_generator = generator(validation_samples, batch_size=32, correction=steering_correction)

    number_of_training_samples = len(train_samples) * 6
    number_of_validation_samples = len(validation_samples) * 6

    nvidia_model = create_model()
    nvidia_model.summary()
    nvidia_model.compile(loss='mse', optimizer='adam')

    history_object = nvidia_model.fit_generator(train_generator,
                                                samples_per_epoch=number_of_training_samples,
                                                validation_data=validation_generator,
                                                nb_val_samples=number_of_validation_samples,
                                                nb_epoch=3, verbose=1)

    nvidia_model.save('model.h5')  # creates a HDF5 file 'model.h5'

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
