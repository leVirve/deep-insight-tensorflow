from config import gpu_dev, config

import time
import tensorflow as tf

from models.network import TFCNN
from datasets import MNist

train_dir = './logs/train'
epochs, batch_size = 20, 1024
dataset = MNist(batch_size=batch_size, reshape=False)


min_after_dequeue = 100
capacity = min_after_dequeue + 3 * batch_size
img, label = MNist.read_and_decode(filename='data/mnist-train.tfrecord', epochs=epochs)
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                min_after_dequeue=min_after_dequeue,
                                                num_threads=32)
img, label = MNist.read_and_decode(filename='data/mnist-test.tfrecord')
x_test_batch, y_test_batch = tf.train.batch([img, label],
                                            batch_size=batch_size,
                                            capacity=capacity,
                                            num_threads=32)

with tf.device('/gpu:%d' % gpu_dev):
    net = TFCNN(img_batch, label_batch, is_sparse=True).build_graph()
    tf.get_variable_scope().reuse_variables()
    net_test = TFCNN(x_test_batch, y_test_batch, is_train=False, is_sparse=True).build_graph()

with tf.Session(config=config) as sess:
    sess.run(tf.group(
        tf.global_variables_initializer(), tf.local_variables_initializer()))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        epoch = 0
        s = time.time()
        while not coord.should_stop():
            avg_loss = 0.
            for i in range(dataset.num_train_batch):
                _, c = sess.run([net.optimize, net.loss])
                avg_loss += c / dataset.num_train_batch
            epoch += 1
            print('Epoch {:02d}: loss = {:.9f}'.format(epoch, avg_loss))
    except tf.errors.OutOfRangeError:
        print('Done training after {} epochs.'.format(epochs))
        print('Elasped time:', time.time() - s)
    finally:
        coord.request_stop()

    print('Testing Accuracy: {:.2f}%'.format(sess.run(net_test.accuracy) * 100))

    coord.join(threads)
    sess.close()
