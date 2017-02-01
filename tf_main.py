from dpt.tools import config as cfg

import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from dpt.tools import cli
from dpt.network import TFCNN
from dpt.dataset import MNist

dataset = MNist(batch_size=cfg.batch_size, reshape=False)


def initial(is_train):
    global x, y, net, saver
    with tf.device(cfg.gpu_device):
        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.float32, [None, *dataset.image_shape], name='image')
            y = tf.placeholder(tf.float32, [None, dataset.classes], name='label')
        net = TFCNN(x, y, is_train=is_train).build_graph()
    saver = tf.train.Saver()


def train():
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(cfg.train_dir, sess.graph)

    for epoch in range(cfg.epochs):
        loss = 0.
        for i in range(dataset.num_train_batch):
            batch_x, batch_y = dataset.next_batch()
            _, c, summary = sess.run(net.train_op, feed_dict={x: batch_x, y: batch_y})
            loss += c
        train_writer.add_summary(summary, epoch)
        if epoch % 2 == 0:
            saver.save(sess, cfg.model_path, global_step=net.step)
        print('Epoch {:02d}: loss = {:.9f}'.format(epoch + 1, loss / dataset.num_train_batch))


def evaluate():
    model_path = tf.train.latest_checkpoint(cfg.model_dir)
    saver.restore(sess, model_path)
    acc = sess.run(net.accuracy, feed_dict={x: dataset.test.images, y: dataset.test.labels})
    print('Testing Accuracy: {:.2f}%'.format(acc * 100))


def export():
    model_path = tf.train.latest_checkpoint(cfg.model_dir)
    saver.restore(sess, model_path)
    graph = convert_variables_to_constants(sess, sess.graph_def, ['accuracy/pred_class'])
    tf.train.write_graph(graph, cfg.model_dir, 'exported_graph.pb', as_text=False)


def predict():
    with open(cfg.model_dir + 'exported_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        output = tf.import_graph_def(
            graph_def,
            input_map={'inputs/image:0': dataset.test.images[:10]},
            return_elements=['accuracy/pred_class:0'],
            name='pred')
        print(sess.run(output))


if __name__ == '__main__':
    mode = cli.args.mode
    initial(mode == 'train')
    func = {
        'train': train,
        'eval': evaluate,
        'export': export,
        'predict': predict,
    }.get(mode, evaluate)

    with tf.Session(config=cfg.config) as sess:
        func()
