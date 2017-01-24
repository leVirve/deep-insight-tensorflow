from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D

from models.base import BaseNet


class CNet(BaseNet):

    NAME = 'CNet'

    def build_model(self, train=True):
        layers = [
            Convolution2D(32, 3, 3, activation='relu', input_shape=self.input_shape),
            Convolution2D(32, 3, 3, activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax'),
        ]

        model = Sequential(layers=layers, name=self.NAME)

        if train:
            model.compile(
                optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

        return model
