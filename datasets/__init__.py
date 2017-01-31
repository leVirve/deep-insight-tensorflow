import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class MNist:

    path = 'data/mnist-data'
    image_shape = (28, 28, 1)
    classes = 10

    def __init__(self, batch_size, *args, **kwargs):
        self.raw = input_data.read_data_sets(self.path, one_hot=True, *args, **kwargs)
        self.batch_size = batch_size
        self.num_train_batch = self.raw.train.num_examples // batch_size

    def next_batch(self):
        return self.raw.train.next_batch(self.batch_size)

    @property
    def test_set(self):
        return self.raw.test

    @staticmethod
    def read_and_decode(filename, epochs=None):
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

        return img, label
