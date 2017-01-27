import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def get_weights(kernel_shape):
    return tf.get_variable('weights', kernel_shape,
                           initializer=tf.random_normal_initializer())


def get_biases(bias_shape):
    return tf.get_variable('biases', bias_shape,
                           initializer=tf.constant_initializer(0.0))


def define_scope(function):
    def decorator(*args, **kwargs):
        with tf.variable_scope(kwargs.get('name')):
            return function(*args, **kwargs)
    return decorator


@define_scope
def conv(x, kernel_shape, bias_shape, strides=1, name=None):
    w, b = get_weights(kernel_shape), get_biases(bias_shape)
    r = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(r, b))


@define_scope
def pool(x, k=2, name=None):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


@define_scope
def dense(x, kernel_shape, bias_shape, name=None):
    w, b = get_weights(kernel_shape), get_biases(bias_shape)
    return tf.add(tf.matmul(x, w), b)


def cnet(x):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv(x, [5, 5, 1, 32], [32], name='conv1')
    pool1 = pool(conv1, name='pool1')
    conv2 = conv(pool1, [5, 5, 32, 64], [64], name='conv2')
    pool2 = pool(conv2, name='pool2')
    flat1 = tf.reshape(pool2, [-1, 7 * 7 * 64])
    fc1 = dense(flat1, [7 * 7 * 64, 1024], [1024], name='fc1')
    with tf.variable_scope('fc1'):
        fc1 = tf.nn.dropout(tf.nn.relu(fc1), 0.75)
    fc2 = dense(fc1, [1024, 10], [10], name='fc2')

    return fc2


x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.float32, [None, 10])


with tf.name_scope('loss'):
    pred = cnet(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

with tf.name_scope('optimize'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

epochs, batch_size = 20, 1024
total_batch = mnist.train.num_examples // batch_size

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    s = time.time()
    for epoch in range(epochs):
        avg_cost = 0.
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print('Epoch {:02d}: cost = {:.9f}'.format(epoch + 1, avg_cost))
    print('Elasped time:', time.time() - s)

    acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y: mnist.test.labels})
    print('Testing Accuracy: {:.2f}%'.format(acc * 100))
