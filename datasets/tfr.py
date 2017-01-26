import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/mnist-data', one_hot=True)

import numpy as np
import matplotlib.pyplot as plt


def to_img_raw(img):
    return img.tobytes()


def to_label(label):
    return int(np.where(label==1)[0])


def data_to_tfrecord(images, labels, filename):
    print("Converting data into %s ..." % filename)

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


def read_and_decode(filename, is_train=None):
    filename_queue = tf.train.string_input_producer([filename])

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

    if is_train:
        img = tf.random_crop(img, [24, 24, 1])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=63)
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        img = tf.image.per_image_standardization(img)
    elif is_train is False:
        img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
        img = tf.image.per_image_standardization(img)

    label = tf.cast(features['label'], tf.int32)
    return img, label

data_to_tfrecord(mnist.train.images, mnist.train.labels, filename='../data/mnist-train.tfrecord')
data_to_tfrecord(mnist.test.images, mnist.test.labels, filename='../data/mnist-test.tfrecord')

img, label = read_and_decode(filename='mnist-train.tfrecord')
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=4,
                                                capacity=50000,
                                                min_after_dequeue=10000,
                                                num_threads=1)
print("img_batch   : %s" % img_batch._shape)
print("label_batch : %s" % label_batch._shape)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(3):  # number of mini-batch (step)
        print("Step %d" % i)
        val, l = sess.run([img_batch, label_batch])
        # exit()
        print(val.shape, l)

    coord.request_stop()
    coord.join(threads)
    sess.close()

assert 0

# Create the graph, etc.
init_op = tf.global_variables_initializer()

# Create a session for running operations in the Graph.
sess = tf.Session()

# Initialize the variables (like the epoch counter).
sess.run(init_op)

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    while not coord.should_stop():
        # Run training steps or whatever
        sess.run(train_op)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()