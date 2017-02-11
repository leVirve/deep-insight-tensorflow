import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from dpt.tools.tfrecord import generate, read_and_decode


class MNist():

    path = 'data/mnist-data'
    image_shape = (28, 28, 1)
    classes = 10

    def __init__(self, batch_size=0, *args, **kwargs):
        self.raw = input_data.read_data_sets(self.path, *args, **kwargs)
        self.batch_size = batch_size
        self.num_train_batch = self.raw.train.num_examples // batch_size if batch_size else None

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

    image_shape = MNist.image_shape
    crop_shape = (24, 24, 1)

    def __init__(self, config):
        self.config = config
        self._dataset = MNist(reshape=False)

    def generate(self):
        generate(*self._dataset.train_set, self.config.train.tfrecord.filepath)
        generate(*self._dataset.test_set, self.config.test.tfrecord.filepath)

    def read_batched(self, filepath, process_level, epoch_limit):
        params = {'shape': self.image_shape, 'crop_shape': self.crop_shape}
        return read_and_decode(filepath, epochs=epoch_limit, preprocess=process_level, **params)

    def fetch(self, train):
        phase = 'train' if train else 'test'
        phase_cfg = self.config.cfg[phase]

        process_level = phase_cfg.tfrecord.preprocess_level
        epoch_limit = self.config.train.epochs if train else None

        num_batch = getattr(self._dataset.raw, phase).num_examples // phase_cfg.batch_size
        batched = self.read_batched(phase_cfg.tfrecord.filepath, process_level, epoch_limit)
        if train:
            img_batch, label_batch = tf.train.shuffle_batch(batched, **self.config.batcher_params.train)
            img_batch = tf.image.resize_bilinear(img_batch, self.image_shape[:2])
        else:
            print(self.config.batcher_params.test)
            img_batch, label_batch = tf.train.batch(batched, **self.config.batcher_params.test)

        return img_batch, label_batch, num_batch
