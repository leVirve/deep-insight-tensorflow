import numpy as np
import tensorflow as tf


def data_to_tfrecord(images, labels, filename):
    print('Converting data into %s ...' % filename)

    def to_img_raw(img):
        return img.tobytes()


    def to_label(label):
        return int(np.where(label==1)[0])

    writer = tf.python_io.TFRecordWriter(filename)
    for img, label in zip(images, labels):
        img_raw = to_img_raw(img)
        label = to_label(label)

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename, epochs=None, preprocess=1):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw' : tf.FixedLenFeature([], tf.string),
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

    return img, label


def train_tfrecord(filename, cfg):
    img, label = read_and_decode(filename=filename, epochs=cfg.epochs, preprocess=cfg.preprocess_level)
    img_batch, label_batch =  tf.train.shuffle_batch(
        [img, label],
        batch_size=cfg.batch_size,
        capacity=cfg.capacity,
        min_after_dequeue=cfg.min_after_dequeue,
        num_threads=cfg.num_threads)
    img_batch = tf.image.resize_bilinear(img_batch, [28, 28])
    return img_batch, label_batch


def test_tfrecord(filename, cfg):
    img, label = read_and_decode(filename=filename, preprocess=0)
    return tf.train.batch(
        [img, label],
        batch_size=cfg.batch_size,
        capacity=cfg.capacity,
        num_threads=cfg.num_threads)
