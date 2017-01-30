import time
import tensorflow as tf

from models.network import TFCNN
from datasets import MNist

tf.logging.set_verbosity(tf.logging.INFO)

train_dir = './logs/train'
epochs, batch_size = 20, 1024
dataset = MNist(batch_size=batch_size)

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
        loss = 0.
        for i in range(dataset.num_train_batch):
            batch_x, batch_y = dataset.next_batch()
            summary, _, c = sess.run([merged, net.optimize, net.loss], feed_dict={x: batch_x, y: batch_y})
            loss += c
        train_writer.add_summary(summary, epoch)
        print('Epoch {:02d}: loss = {:.9f}'.format(epoch + 1, loss / dataset.num_train_batch))
    print('Elasped time:', time.time() - s)

    acc = sess.run(net.accuracy, feed_dict={x: dataset.test_set.images, y: dataset.test_set.labels})
    print('Testing Accuracy: {:.2f}%'.format(acc * 100))

    saver.save(sess, train_dir, global_step=step)
