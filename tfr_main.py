import time
import tensorflow as tf
import numpy as np

if False:
    from datasets.mnist import MNist
    mnist = MNist()
    train_set, test_set = mnist.load()
else:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights, biases):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, 0.75)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([10]))
}


''' Data part '''
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
    label = tf.cast(features['label'], tf.int32)

    return img, label


epochs = 20
batch_size = 128
total_batch = mnist.train.num_examples // batch_size

img, label = read_and_decode(filename='data/mnist-train.tfrecord', is_train=None)
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=batch_size,
                                                capacity=5000,
                                                min_after_dequeue=1000,
                                                num_threads=32)
img, label = read_and_decode(filename='data/mnist-train.tfrecord', is_train=None)
x_test_batch, y_test_batch = tf.train.batch([img, label],
                                            batch_size=batch_size,
                                            capacity=500,
                                            num_threads=32)

''' Data part (End) '''

pred = conv_net(img_batch, weights, biases)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label_batch))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

test_pred = conv_net(x_test_batch, weights, biases)

correct_pred = tf.equal(tf.cast(tf.argmax(test_pred, 1), tf.int32), y_test_batch)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    s = time.time()

    try:

        for epoch in range(epochs):
            avg_cost = 0.
            for i in range(total_batch):
                _, c = sess.run([optimizer, cost])
                avg_cost += c / total_batch
            print('Epoch {:04d}: cost = {:.9f}'.format(epoch + 1, avg_cost))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    print('Elasped time:', time.time() - s)

    print("Testing Accuracy:", sess.run(accuracy))

    coord.join(threads)
    sess.close()
