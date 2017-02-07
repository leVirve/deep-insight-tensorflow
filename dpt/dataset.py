import os
import pickle

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from dpt.tools.tfrecord import generate, read_and_decode


class MNist():

    path = 'data/mnist-data'
    image_shape = (28, 28, 1)
    classes = 10

    def __init__(self, batch_size, *args, **kwargs):
        self.raw = input_data.read_data_sets(self.path, *args, **kwargs)
        self.batch_size = batch_size
        self.num_train_batch = self.raw.train.num_examples // batch_size

    @property
    def test(self):
        return self.raw.test

    @property
    def train(self):
        return self.raw.train

    @property
    def train_set(self):
        return (self.raw.train.images, self.raw.train.labels)

    @property
    def test_set(self):
        return (self.raw.test.images, self.raw.test.labels)

    def next_batch(self):
        return self.raw.train.next_batch(self.batch_size)


class MNistRecorder():

    metafile = 'tfrecord.meta'
    path = 'data/mnist'
    image_shape = (28, 28, 1)

    def __init__(self, records={}):
        self.records = records
        self.meta = {}
        self.num_batch = 0

    def generate(self, images, labels, filename):
        filepath = self.get_fullpath(filename)
        generate(images, labels, filepath)
        with open(self.get_fullpath(self.metafile), 'wb+') as f:
            self.meta[filename] = {'num_examples': len(labels)}
            pickle.dump(self.meta, f)

    def read_and_decode(self, phase='test', epochs=None, preprocess=1):
        filename = self.records.get(phase)
        params = {
            'filepath': self.get_fullpath(filename),
            'epochs': epochs,
            'preprocess': preprocess,
            'shape': self.image_shape,
            'crop_shape': (24, 24, 1),
        }
        img, label = read_and_decode(**params)
        with open(self.get_fullpath(self.metafile), 'rb') as f:
            meta = pickle.load(f)
        return [img, label], meta[filename]['num_examples']

    def fetch(self, cfg, train):
        return self.fetch_train(cfg) if train else self.fetch_test(cfg)

    def fetch_train(self, cfg):
        params = {'phase': 'train', 'epochs': cfg.epochs, 'preprocess': cfg.preprocess_level}
        batched, num_examples = self.read_and_decode(**params)

        img_batch, label_batch = tf.train.shuffle_batch(
            batched,
            batch_size=cfg.batch_size,
            capacity=cfg.capacity,
            min_after_dequeue=cfg.min_after_dequeue,
            num_threads=cfg.num_threads)
        img_batch = tf.image.resize_bilinear(img_batch, *self.image_shape[:2])
        tf.summary.image('training_images', img_batch)

        self.num_batch = num_examples // cfg.batch_size
        return img_batch, label_batch

    def fetch_test(self, cfg):
        params = {'phase': 'test', 'preprocess': 0}
        batched, num_examples = self.read_and_decode(**params)

        img_batch, label_batch = tf.train.batch(
            batched,
            batch_size=cfg.batch_size,
            capacity=cfg.capacity,
            num_threads=cfg.num_threads)

        self.num_batch = num_examples // cfg.batch_size
        return img_batch, label_batch

    def get_fullpath(self, filename):
        return os.path.join(self.path, filename)
