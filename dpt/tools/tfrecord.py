import os

import tensorflow as tf


def generate(images, labels, filepath):

    working_dir = os.path.dirname(filepath)
    os.makedirs(working_dir, exist_ok=True)

    writer = tf.python_io.TFRecordWriter(filepath)
    for img, label in zip(images, labels):
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
        writer.write(example.SerializeToString())
    writer.close()

    return len(labels)


def read_and_decode(filepath, epochs=None, preprocess=1, **kwargs):

    filename_queue = tf.train.string_input_producer([filepath], num_epochs=epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
        })

    img_shape = kwargs.get('shape')
    cropped_shape = kwargs.get('crop_shape')

    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, img_shape)
    label = tf.cast(features['label'], tf.int32)

    if preprocess == 2:
        img = tf.random_crop(img, cropped_shape)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=63)
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        img = tf.image.per_image_standardization(img)
    elif preprocess == 1:
        img = tf.image.resize_image_with_crop_or_pad(img, *cropped_shape[:2])
        img = tf.image.per_image_standardization(img)

    return img, label
