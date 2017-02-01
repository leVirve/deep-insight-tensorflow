from dpt.tools import config as cfg

import tensorflow as tf

from dpt.tools import cli
from dpt.dataset import MNist
from dpt.tools.tfrecord import Recorder
from dpt.network import TFCNN


def initial(is_train):
    global net, saver, reader, num_train_batch
    reader = Recorder(working_dir='data/mnist/')
    if is_train:
        img_batch, label_batch = reader.train_tfrecord('mnist-train.tfrecord', cfg)
    else:
        img_batch, label_batch = reader.test_tfrecord('mnist-test.tfrecord', cfg)
    num_train_batch = reader.num_examples[0] // cfg.batch_size
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
            for i in range(num_train_batch):
                _, c, summary = sess.run(net.train_op)
                avg_loss += c / num_train_batch
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

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print('Testing Accuracy: {:.2f}%'.format(sess.run(net.accuracy) * 100))
    coord.request_stop()
    coord.join(threads)
    sess.close()


def gen_tfrecord():
    dataset = MNist(batch_size=cfg.batch_size, reshape=False)
    recorder = Recorder(working_dir='data/mnist/')
    recorder.generate(dataset.train.images, dataset.train.labels, filename='mnist-train.tfrecord')
    recorder.generate(dataset.test.images, dataset.test.labels, filename='mnist-test.tfrecord')


if __name__ == '__main__':
    mode = cli.args.mode
    if mode in ['train', 'eval']:
        initial(mode == 'train')
    func = {
        'gen': gen_tfrecord,
        'train': train,
        'eval': evaluate,
    }.get(mode, evaluate)

    with tf.Session(config=cfg.config) as sess:
        func()
