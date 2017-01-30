import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.network import TFCNN

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train_dir = './logs/train'
epochs, batch_size = 20, 1024
total_batch = mnist.train.num_examples // batch_size

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    y = tf.placeholder(tf.float32, [None, 10])

logits = TFCNN().build_model(x)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name='loss')
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step=step)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

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
            summary, _, c = sess.run([merged, train_op, loss], feed_dict={x: batch_x, y: batch_y})
            avg_loss += c / total_batch
        train_writer.add_summary(summary, epoch)
        print('Epoch {:02d}: cost = {:.9f}'.format(epoch + 1, avg_loss))
    print('Elasped time:', time.time() - s)

    acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y: mnist.test.labels})
    print('Testing Accuracy: {:.2f}%'.format(acc * 100))

    saver.save(sess, train_dir, global_step=step)
