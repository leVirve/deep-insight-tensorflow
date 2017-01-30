import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.network import TFCNN

tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets('data/mnist-data', one_hot=True)
train_dir = './logs/train'
epochs, batch_size = 20, 1024
total_batch = mnist.train.num_examples // batch_size

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    y = tf.placeholder(tf.float32, [None, 10])

step = tf.Variable(0, name='global_step', trainable=False)
net = TFCNN(x, y, step)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(train_dir, sess.graph)
    saver = tf.train.Saver()

    s = time.time()
    for epoch in range(epochs):
        avg_loss = 0.
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            summary, _, c = sess.run([merged, net.optimize, net.loss], feed_dict={x: batch_x, y: batch_y})
            avg_loss += c / total_batch
        train_writer.add_summary(summary, epoch)
        print('Epoch {:02d}: cost = {:.9f}'.format(epoch + 1, avg_loss))
    print('Elasped time:', time.time() - s)

    acc = sess.run(net.accuracy, feed_dict={x: mnist.test.images,
                                            y: mnist.test.labels})
    print('Testing Accuracy: {:.2f}%'.format(acc * 100))

    saver.save(sess, train_dir, global_step=step)
