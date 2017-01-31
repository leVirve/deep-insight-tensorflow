import tensorflow as tf

from datasets import MNist


def train_tfrecord(filename, cfg):
    img, label = MNist.read_and_decode(filename=filename, epochs=cfg.epochs)
    return tf.train.shuffle_batch(
        [img, label],
        batch_size=cfg.batch_size,
        capacity=cfg.capacity,
        min_after_dequeue=cfg.min_after_dequeue,
        num_threads=cfg.num_threads)


def test_tfrecord(filename, cfg):
    img, label = MNist.read_and_decode(filename=filename)
    return tf.train.batch(
        [img, label],
        batch_size=cfg.batch_size,
        capacity=cfg.capacity,
        num_threads=cfg.num_threads)
