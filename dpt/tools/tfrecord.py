import os
import pickle

import numpy as np
import tensorflow as tf


class Recorder():

    metafile = 'tfrecord.meta'

    def __init__(self, working_dir='.'):
        self.working_dir = working_dir
        self.meta = {}
        self.num_examples = []

    def generate(self, images, labels, filename):

        def to_img_raw(img):
            return img.tobytes()

        def to_label(label):
            return int(np.where(label==1)[0])

        os.makedirs(self.working_dir, exist_ok=True)

        writer = tf.python_io.TFRecordWriter(self.get_fullpath(filename))
        for img, label in zip(images, labels):
            img_raw = to_img_raw(img)
            label = to_label(label)
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            }))
            writer.write(example.SerializeToString())
        writer.close()

        with open(self.get_fullpath(self.metafile), 'wb+') as f:
            self.meta[filename] = {'num_examples': len(labels)}
            pickle.dump(self.meta, f)

    def read_and_decode(self, filename, epochs=None, preprocess=1):
        path = self.get_fullpath(filename)
        filename_queue = tf.train.string_input_producer([path], num_epochs=epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string),
            })

        img = tf.decode_raw(features['img_raw'], tf.float32)
        img = tf.reshape(img, [28, 28, 1])
        label = tf.cast(features['label'], tf.int32)

        if preprocess == 2:
            img = tf.random_crop(img, [24, 24, 1])
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=63)
            img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
            img = tf.image.per_image_standardization(img)
        elif preprocess == 1:
            img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
            img = tf.image.per_image_standardization(img)

        with open(self.get_fullpath(self.metafile), 'rb') as f:
            meta = pickle.load(f)
            self.num_examples.append(meta[filename]['num_examples'])

        return img, label

    def train_tfrecord(self, filename, cfg):
        img, label = self.read_and_decode(filename, epochs=cfg.epochs, preprocess=cfg.preprocess_level)
        img_batch, label_batch =  tf.train.shuffle_batch(
            [img, label],
            batch_size=cfg.batch_size,
            capacity=cfg.capacity,
            min_after_dequeue=cfg.min_after_dequeue,
            num_threads=cfg.num_threads)
        img_batch = tf.image.resize_bilinear(img_batch, [28, 28])
        tf.summary.image('training_images', img_batch)
        return img_batch, label_batch

    def test_tfrecord(self, filename, cfg):
        img, label = self.read_and_decode(filename, preprocess=0)
        return tf.train.batch(
            [img, label],
            batch_size=cfg.batch_size,
            capacity=cfg.capacity,
            num_threads=cfg.num_threads)

    def get_fullpath(self, filename):
        return os.path.join(self.working_dir, filename)
