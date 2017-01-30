from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
import tensorflow as tf
from tensorflow.python.layers import layers

from models.base import BaseNet


class KerasCNN(BaseNet):

    NAME = 'KerasCNN'

    def build_model(self):
        layers = [
            Convolution2D(
                32, *(5, 5),
                border_mode='same',
                activation='relu',
                input_shape=self.input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Convolution2D(
                64, *(5, 5), border_mode='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.4),
            Dense(10, activation='softmax'),
        ]
        model = Sequential(layers=layers, name=self.NAME)
        return model

    def compile(self):
        self.model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return self


class TFCNN:

    NAME = 'TFCNN'

    def __init__(self):
        self.model = None

    def build_model(self, x, is_train=True):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        conv1 = layers.conv2d(
            x, 32, [5, 5], padding='same', activation=tf.nn.relu, name='conv1')
        pool1 = layers.max_pooling2d(
            conv1, pool_size=[2, 2], strides=2, name='pool1')
        conv2 = layers.conv2d(
            pool1,
            64, [5, 5],
            padding='same',
            activation=tf.nn.relu,
            name='conv2')
        pool2 = layers.max_pooling2d(
            conv2, pool_size=[2, 2], strides=2, name='pool2')
        flat1 = tf.reshape(pool2, [-1, 7 * 7 * 64], name='flatten')
        dense = layers.dense(
            flat1, units=1024, activation=tf.nn.relu, name='fc1')
        dropout = layers.dropout(
            dense, rate=0.4, training=is_train, name='dropout')
        logits = layers.dense(dropout, units=10, name='fc2')
        return logits
