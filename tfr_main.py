from tools import config as cfg

import tensorflow as tf

from tools import cli
from tools.tfrecord import train_tfrecord, test_tfrecord
from models.network import TFCNN
from datasets import MNist

dataset = MNist(batch_size=cfg.batch_size, reshape=False)


def initial(is_train):
    global net, saver
    if is_train:
        img_batch, label_batch = train_tfrecord('data/mnist-train.tfrecord', cfg)
    else:
        img_batch, label_batch = test_tfrecord('data/mnist-test.tfrecord', cfg)
    with tf.device(cfg.gpu_device):
        net = TFCNN(
            img_batch, label_batch,
            is_sparse=True, is_train=is_train).build_graph()
    saver = tf.train.Saver()


def train():
    sess.run(
        tf.group(tf.global_variables_initializer(),
                 tf.local_variables_initializer()))
    train_writer = tf.summary.FileWriter(cfg.train_dir, sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        epoch = 0
        while not coord.should_stop():
            avg_loss = 0.
            for i in range(dataset.num_train_batch):
                _, c, summary = sess.run(net.train_op)
                avg_loss += c / dataset.num_train_batch
            if epoch % 2 == 0:
                saver.save(sess, cfg.model_path, global_step=net.step)
            train_writer.add_summary(summary, epoch)
            epoch += 1
            print('Epoch {:02d}: loss = {:.9f}'.format(epoch, avg_loss))
    except tf.errors.OutOfRangeError:
        print('Done training after {} epochs.'.format(cfg.epochs))
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


def evaluate():
    model_path = tf.train.latest_checkpoint(cfg.model_dir)
    saver.restore(sess, model_path)
    print('Testing Accuracy: {:.2f}%'.format(sess.run(net.accuracy) * 100))


if __name__ == '__main__':
    mode = cli.args.mode
    initial(mode == 'train')
    func = {
        'train': train,
        'eval': evaluate,
    }.get(mode, evaluate)

    with tf.Session(config=cfg.config) as sess:
        func()
