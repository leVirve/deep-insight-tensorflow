import os
from functools import partialmethod

import tensorflow as tf
from keras.callbacks import TensorBoard
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from dpt.dataset import MNist
from dpt.network import KerasCNN, TensorCNN
from dpt.tools import tfrecord, timeit


class BasicFramework():

    command_exception_fmt = '{} has no such command: {}'
    need_setup = ['train', 'evaluate']

    def __init__(self, cfg):
        self.net = None
        self.cfg = cfg

    def setup(self, mode):
        raise Exception('Not implemented')

    def execute(self, mode):
        if self.net is None and mode in self.need_setup:
            self.setup(mode)
        runner = getattr(self, mode, None)
        if not callable(runner):
            raise Exception(self.command_exception_fmt.format(self.name, mode))
        runner()
        self.finish()

    def finish(self):
        pass


class KerasFramework(BasicFramework):

    name = 'KerasFramework'
    need_setup = ['train', 'evaluate', 'predict']
    weights_path = 'data/weights/{}_weights.h5'

    def setup(self, mode):
        dataset = MNist(batch_size=self.cfg.batch_size, reshape=False, one_hot=True)
        self.dataset = dataset
        self.net = KerasCNN(image_shape=dataset.image_shape)
        return self

    def train(self):
        callbacks = [
            TensorBoard(
                histogram_freq=2,
                write_graph=True,
                write_images=False,
                log_dir=self.cfg.train_dir)
        ]
        net = self.net
        net.compile()
        net.model.fit(*self.dataset.train_set,
                      validation_data=self.dataset.test_set,
                      nb_epoch=self.cfg.epochs,
                      batch_size=self.cfg.batch_size,
                      callbacks=callbacks)
        self._save_weights()

    def evaluate(self):
        self._load_weights()
        self.net.compile()
        _, accuracy = self.net.model.evaluate(
            *self.dataset.test_set, batch_size=self.cfg.batch_size)
        print('== %s ==\nTest accuracy: %.2f%%' % (self.net.NAME, accuracy * 100))

    def predict(self):
        self._load_weights()
        print(self.net.model.predict(self.dataset.raw.test.images))

    def _get_weight_name(self):
        return self.weights_path.format(self.net.NAME)

    def _load_weights(self):
        return self.net.model.load_weights(self._get_weight_name())

    def _save_weights(self):
        weights_name = self._get_weight_name()
        dir_path = os.path.dirname(weights_name)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        return self.net.model.save_weights(weights_name)


class TensorflowFramework(BasicFramework):

    name = 'TensorflowFramework'
    need_setup = ['train', 'evaluate', 'export', 'predict']
    exported_graphdef = 'tf_graphdef.pb'

    def setup(self, mode):
        self.is_train = mode == 'train'
        self.net = self._build_graph()
        self.saver = tf.train.Saver()
        self.session = tf.Session(config=self.cfg.config)
        self.writer = tf.summary.FileWriter(self.cfg.train_dir, self.session.graph)
        self.session.run(tf.group(tf.global_variables_initializer(),
                                  tf.local_variables_initializer()))
        return self

    def _build_graph(self):
        x, y = self._build_inputs()
        with tf.device(self.cfg.gpu_device):
            return TensorCNN(x, y, is_train=self.is_train).build_graph()

    def _build_inputs(self):
        dataset = MNist(batch_size=self.cfg.batch_size, reshape=False)
        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.float32, [None, *dataset.image_shape], name='image')
            y = tf.placeholder(tf.int32, [None], name='label')
        self.dataset = dataset
        self.batch_per_step = self.dataset.num_train_batch
        self._predict_input = dataset.test.images[:10]
        self.x = x
        self.y = y
        return x, y

    def _save_session(self):
        self.saver.save(
            self.session, self.cfg.model_path, global_step=self.net.step)

    def _restore_session(self):
        latest_ckpt = tf.train.latest_checkpoint(self.cfg.model_dir)
        self.saver.restore(self.session, latest_ckpt)

    def _restore_graph_def(self):
        with open(self.cfg.model_dir + self.exported_graphdef, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def

    @timeit
    def _train_an_epoch(self, num_iter, **kwargs):
        loss = 0.
        for _ in range(num_iter):
            _, c, summary = self.session.run(self.net.train_op, **kwargs)
            loss += c / num_iter
        return loss, summary

    def _train_summary(self, epoch, summary):
        if epoch % 2 == 0:
            self._save_session()
        self.writer.add_summary(summary, epoch)

    def train(self):
        for epoch in range(1, self.cfg.epochs + 1):
            x, y = self.dataset.next_batch()
            loss, summary = self._train_an_epoch(self.batch_per_step, feed_dict={self.x: x, self.y: y})
            self._train_summary(epoch, summary)
            print('Epoch {:02d}: loss = {:.9f}'.format(epoch, loss))

    def evaluate(self):
        self._restore_session()
        acc = self.session.run(
            self.net.accuracy,
            feed_dict={
                self.x: self.dataset.test.images,
                self.y: self.dataset.test.labels})
        print('Testing Accuracy: {:.2f}%'.format(acc * 100))

    def export(self):
        self._restore_session()
        graph = convert_variables_to_constants(
            self.session, self.session.graph_def, ['accuracy/pred_class'])
        tf.train.write_graph(graph, self.cfg.model_dir, self.exported_graphdef, as_text=False)

    def predict(self):
        graph_def = self._restore_graph_def()
        output = tf.import_graph_def(
            graph_def,
            input_map={'inputs/image:0': self._predict_input},
            return_elements=['accuracy/pred_class:0'],
            name='pred')
        print(self.session.run(output))

    def finish(self):
        if hasattr(self, 'session'):
            self.session.close()
            self.writer.close()


class TensorflowStdFramework(TensorflowFramework):

    name = 'TensorflowStdFramework'
    need_setup = ['train', 'evaluate']

    records = {'train': 'mnist-train.tfrecord', 'test': 'mnist-test.tfrecord'}
    exported_graphdef = 'tfr_graphdef.pb'

    @timeit
    def _build_inputs(self):
        batch_reader = tfrecord.Recorder(self.records, working_dir='data/mnist/')
        x, y, batch_per_step = batch_reader.fetch(self.cfg, self.is_train)
        self.batch_per_step = batch_per_step
        return x, y

    def runner(self, f):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
        try:
            f(coord)
        except tf.errors.OutOfRangeError:
            print('Done training after {} epochs.'.format(self.cfg.epochs))
        finally:
            coord.request_stop()
        coord.join(threads)

    def train(self):
        def worker(coord):
            epoch = 0
            while not coord.should_stop():
                loss, summary = self._train_an_epoch(self.batch_per_step)
                epoch += 1
                self._train_summary(epoch, summary)
                print('Epoch {:02d}: loss = {:.9f}'.format(epoch, loss))
        self.runner(worker)

    def evaluate(self):
        def worker(coord):
            num_iter = self.batch_per_step
            step, avg_acc = 0, 0.
            while step < num_iter and not coord.should_stop():
                acc = self.session.run(self.net.accuracy)
                avg_acc += acc / num_iter
                step += 1
            print('Testing Accuracy: {:.2f}%'.format(avg_acc * 100))
        self._restore_session()
        self.runner(worker)

    def gen_tfrecord(self):
        dataset = MNist(batch_size=self.cfg.batch_size, reshape=False)
        recorder = tfrecord.Recorder(working_dir='data/mnist/')
        recorder.generate(*dataset.train_set, filename=self.records['train'])
        recorder.generate(*dataset.test_set, filename=self.records['test'])

    def _fake_interface(self, mode):
        self._build_inputs = super()._build_inputs
        self._build_graph = super()._build_graph
        self.setup(mode)
        runner = getattr(super(), mode)
        return runner()

    predict = partialmethod(_fake_interface, 'predict')
    export = partialmethod(_fake_interface, 'export')
