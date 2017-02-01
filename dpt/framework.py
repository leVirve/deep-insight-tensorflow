''' for KerasFramework '''
from keras.callbacks import TensorBoard

''' for TensorflowFramework / TensorflowStdFramework'''
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from dpt.dataset import MNist
from dpt.network import KerasCNN, TensorCNN
from dpt.tools import tfrecord


class BasicFramework:

    command_exception_fmt = '{} has no such command: {}'
    need_setup = ['train', 'evaluate']

    def __init__(self, cfg):
        self.dataset = None
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


class KerasFramework(BasicFramework):

    name = 'KerasFramework'

    def setup(self, mode):
        dataset = MNist(batch_size=self.cfg.batch_size, reshape=False)
        self.dataset = dataset
        self.net = KerasCNN(image_shape=dataset.image_shape)
        self.train_set = (dataset.raw.train.images, dataset.raw.train.labels)
        self.test_set = (dataset.raw.test.images, dataset.raw.test.labels)
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
        net.model.fit(*self.train_set,
                      validation_data=self.test_set,
                      nb_epoch=self.cfg.epochs,
                      batch_size=self.cfg.batch_size,
                      callbacks=callbacks)
        net.save()

    def evaluate(self):
        net = self.net
        net.load()
        net.compile()
        _, accuracy = net.model.evaluate(
            *self.test_set, batch_size=self.cfg.batch_size)
        print('== %s ==\nTest accuracy: %.2f%%' % (net.NAME, accuracy * 100))

    def predict(self):
        self.net.load()
        print(self.net.model.predict(self.dataset.raw.test.images))


class TensorflowFramework(BasicFramework):

    name = 'TensorflowFramework'
    need_setup = ['train', 'evaluate', 'export', 'predict']

    def setup(self, mode):
        dataset = MNist(batch_size=self.cfg.batch_size, reshape=False)
        with tf.device(self.cfg.gpu_device):
            with tf.name_scope('inputs'):
                x = tf.placeholder(tf.float32, [None, *dataset.image_shape], name='image')
                y = tf.placeholder(tf.float32, [None, dataset.classes], name='label')
            net = TensorCNN(x, y, is_train=mode == 'train').build_graph()
        self.dataset = dataset
        self.net = net
        self.x, self.y = x, y
        self.saver = tf.train.Saver()
        self.session = tf.Session(config=self.cfg.config)
        return self

    def shutdown(self):
        session = getattr(self, 'session')
        if session:
            session.close()

    def train(self):
        cfg = self.cfg
        net = self.net
        sess = self.session
        dataset = self.dataset
        num_train_batch = dataset.num_train_batch

        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(cfg.train_dir, sess.graph)
        for epoch in range(cfg.epochs):
            loss = 0.
            for i in range(num_train_batch):
                batch_x, batch_y = dataset.next_batch()
                _, c, summary = sess.run(net.train_op,
                                         feed_dict={self.x: batch_x,
                                                    self.y: batch_y})
                loss += c
            train_writer.add_summary(summary, epoch)
            if epoch % 2 == 0:
                self.saver.save(sess, cfg.model_path, global_step=net.step)
            print('Epoch {:02d}: loss = {:.9f}'.format(epoch + 1, loss / num_train_batch))

    def evaluate(self):
        model_path = tf.train.latest_checkpoint(self.cfg.model_dir)
        self.saver.restore(self.session, model_path)
        acc = self.session.run(self.net.accuracy,
                               feed_dict={self.x: self.dataset.test.images,
                                          self.y: self.dataset.test.labels})
        print('Testing Accuracy: {:.2f}%'.format(acc * 100))

    def export(self):
        sess = self.session
        model_path = tf.train.latest_checkpoint(self.cfg.model_dir)
        self.saver.restore(sess, model_path)
        graph = convert_variables_to_constants(sess, sess.graph_def, ['accuracy/pred_class'])
        tf.train.write_graph(graph, self.cfg.model_dir, 'exported_graph.pb', as_text=False)

    def predict(self):
        with open(self.cfg.model_dir + 'exported_graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            output = tf.import_graph_def(
                graph_def,
                input_map={'inputs/image:0': self.dataset.test.images[:10]},
                return_elements=['accuracy/pred_class:0'],
                name='pred')
            print(self.session.run(output))


class TensorflowStdFramework(BasicFramework):

    name = 'TensorflowStdFramework'

    def setup(self, mode):
        is_train = mode == 'train'
        img_batch, label_batch, num_train_batch = self._reader_batch(is_train)
        with tf.device(self.cfg.gpu_device):
            net = TensorCNN(img_batch, label_batch,
                            is_sparse=True, is_train=is_train).build_graph()
        self.net = net
        self.saver = tf.train.Saver()
        self.session = tf.Session(config=self.cfg.config)
        self.num_train_batch = num_train_batch
        return self

    def shutdown(self):
        session = getattr(self, 'session')
        if session:
            session.close()

    def _reader_batch(self, train):
        reader = tfrecord.Recorder(working_dir='data/mnist/')
        batch_generator = getattr(reader, 'train_tfrecord' if train else 'test_tfrecord')
        tfrecord_file = 'mnist-train.tfrecord' if train else 'mnist-test.tfrecord'
        img_batch, label_batch = batch_generator(tfrecord_file, self.cfg)
        num_train_batch = reader.num_examples[0] // self.cfg.batch_size
        return img_batch, label_batch, num_train_batch

    def train(self):
        cfg = self.cfg
        net = self.net
        sess = self.session
        num_train_batch = self.num_train_batch

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
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
                    self.saver.save(sess, cfg.model_path, global_step=net.step)
                train_writer.add_summary(summary, epoch)
                epoch += 1
                print('Epoch {:02d}: loss = {:.9f}'.format(epoch, avg_loss))
        except tf.errors.OutOfRangeError:
            print('Done training after {} epochs.'.format(cfg.epochs))
        finally:
            coord.request_stop()
        coord.join(threads)

    def evaluate(self):
        sess = self.session

        model_path = tf.train.latest_checkpoint(self.cfg.model_dir)
        self.saver.restore(sess, model_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('Testing Accuracy: {:.2f}%'.format(sess.run(self.net.accuracy) * 100))
        coord.request_stop()
        coord.join(threads)

    def gen_tfrecord(self):
        dataset = MNist(batch_size=self.cfg.batch_size, reshape=False)
        recorder = tfrecord.Recorder(working_dir='data/mnist/')
        recorder.generate(dataset.train.images, dataset.train.labels, filename='mnist-train.tfrecord')
        recorder.generate(dataset.test.images, dataset.test.labels, filename='mnist-test.tfrecord')
