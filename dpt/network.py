from functools import partial, partialmethod

import keras
import tensorflow as tf
import numpy as np
from tensorflow import python as tfs

from dpt.tools import tf_summary, tf_scope


class KerasCNN:

    NAME = 'KerasCNN'

    def __init__(self, image_shape=None):
        self.input_shape = image_shape
        self.model = self.build_model()

    def build_model(self):
        layers = [
            self.conv2d(32, [5, 5], input_shape=self.input_shape, name='conv1'),
            self.pool2d(name='pool1'),
            self.conv2d(64, [5, 5], name='conv2'),
            self.pool2d(name='pool2'),
            self.flatten(name='flatten'),
            self.dense(1024, activation='relu', name='fc1'),
            self.dropout(0.4, name='dropout'),
            self.dense(10, activation='softmax', name='fc2'),
        ]
        model = keras.models.Sequential(layers=layers, name=self.NAME)
        return model

    def compile(self):
        self.model.compile(
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return self

    conv2d = partial(keras.layers.Conv2D, padding='same', activation='relu')
    pool2d = partial(keras.layers.MaxPooling2D, pool_size=(2, 2))
    dropout = keras.layers.Dropout
    flatten = keras.layers.Flatten
    dense = keras.layers.Dense


class TensorCNN:

    NAME = 'TensorCNN'

    def __init__(self, images, labels, step=0, is_train=True):
        self.images = images
        self.labels = labels
        self.step = step
        self.is_train = is_train

    def build_graph(self):
        tf.summary.image('input', self.images)
        self.prediction = self.build_model(self.images)
        self.loss = self.build_loss()
        self.optimize = self.build_optimize()
        self.accuracy = self.build_accuracy()
        self.summary = tf.summary.merge_all()
        self.train_op = [self.optimize, self.loss]
        return self

    def build_model(self, x):
        x = tf.transpose(x, perm=[0, 3, 1, 2])  # NHWC -> NCHW
        x = self.conv2d(x, 32, [5, 5], name='conv1')
        x = self.pool2d(x, name='pool1')
        x = self.conv2d(x, 64, [5, 5], name='conv2')
        x = self.pool2d(x, name='pool2')
        x = self.flatten(x, [-1, np.prod([d.value for d in x.shape if d.value])], name='flatten')
        x = self.dense(x, 1024, activation=tf.nn.relu, name='fc1')
        x = self.dropout(x, 0.4, training=self.is_train, name='dropout')
        x = self.dense(x, 10, name='fc2')
        return x

    @tf_summary('histogram')
    def layer(self, f, *args, **kwargs):
        return f(*args, **kwargs)

    @tf_summary(name='loss')
    def build_loss(self):
        param = {'labels': self.labels, 'logits': self.prediction}
        return tf.losses.sparse_softmax_cross_entropy(**param, scope='loss')

    @tf_summary(name='accuracy')
    @tf_scope(scope='accuracy')
    def build_accuracy(self):
        pred_class = tf.argmax(self.prediction, 1, name='pred_class')
        mean_t, update_op = tf.metrics.accuracy(pred_class, self.labels, name='metric_acc')
        return update_op

    def build_optimize(self):
        step = tf.Variable(self.step, name='global_step', trainable=False)
        return tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, global_step=step)

    conv2d = partialmethod(layer, partial(tfs.layers.conv2d, padding='same', activation=tf.nn.relu, data_format='channels_first'))
    pool2d = partialmethod(layer, partial(tfs.layers.max_pooling2d, pool_size=[2, 2], strides=2, data_format='channels_first'))
    flatten = partialmethod(layer, tf.reshape)
    dense = partialmethod(layer, tfs.layers.dense)
    dropout = partialmethod(layer, tfs.layers.dropout)
