import functools

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
import tensorflow as tf
from tensorflow.python.layers import layers

from models.base import BaseNet


def lazy_property(f):
    attr = '_cached_' + f.__name__
    @property
    def decorator(self):
        if not hasattr(self, attr):
            setattr(self, attr, f(self))
        return getattr(self, attr)
    return decorator


def property_scope(f, *args, **kwargs):
    attr = '_cached_' + f.__name__
    @property
    @functools.wraps(f)
    def decorator(self):
        if not hasattr(self, attr):
            with tf.name_scope(f.__name__, *args, **kwargs):
                setattr(self, attr, f(self))
        return getattr(self, attr)
    return decorator


class KerasCNN(BaseNet):

    NAME = 'KerasCNN'

    def build_model(self):
        layers = [
            Convolution2D(32, *(5, 5), border_mode='same', activation='relu', input_shape=self.input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Convolution2D(64, *(5, 5), border_mode='same', activation='relu'),
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
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return self


class TFCNN:

    NAME = 'TFCNN'

    def __init__(self, images, labels, step=0, is_train=True, is_sparse=False):
        self.images = images
        self.labels = labels
        self.is_train = is_train
        self.is_sparse = is_sparse
        self.step = tf.Variable(step, name='global_step', trainable=False)
        self.build_graph()
        self.summary = tf.summary.merge_all()

    def build_graph(self):
        self.optimize
        self.accuracy

    @lazy_property
    def logits(self):
        conv1 = layers.conv2d(self.images, 32, [5, 5], padding='same', activation=tf.nn.relu, name='conv1')
        pool1 = layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='pool1')
        conv2 = layers.conv2d(pool1, 64, [5, 5], padding='same', activation=tf.nn.relu, name='conv2')
        pool2 = layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='pool2')
        flat1 = tf.reshape(pool2, [-1, 7 * 7 * 64], name='flatten')
        dense = layers.dense(flat1, units=1024, activation=tf.nn.relu, name='fc1')
        dropout = layers.dropout(dense, rate=0.4, training=self.is_train, name='dropout')
        logits = layers.dense(dropout, units=10, name='fc2')
        return logits

    @property_scope
    def loss(self):
        cross_entropy = (
            tf.losses.sparse_softmax_cross_entropy
            if self.is_sparse else
            tf.losses.softmax_cross_entropy)
        if self.is_sparse:
            xentropy = cross_entropy(logits=self.logits, labels=self.labels)
        else:
            xentropy = cross_entropy(logits=self.logits, onehot_labels=self.labels)

        loss = tf.reduce_mean(xentropy, name='loss')
        tf.summary.scalar('loss', loss)
        return loss

    @property_scope
    def optimize(self):
        return tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, global_step=self.step)

    @property_scope
    def accuracy(self):
        if self.is_sparse:
            correct_pred = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), self.labels)
        else:
            correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy
