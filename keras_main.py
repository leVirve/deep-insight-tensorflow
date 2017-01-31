import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.network import KerasCNN

mnist = input_data.read_data_sets("data/mnist-data", one_hot=True, reshape=False)
train_set = (mnist.train.images, mnist.train.labels)
test_set = (mnist.test.images, mnist.test.labels)


def train(net):
    from keras.callbacks import TensorBoard
    callbacks = [
            TensorBoard(
                log_dir='./logs',
                histogram_freq=2, write_graph=True, write_images=False)
        ]
    net.compile()
    net.model.fit(*train_set, validation_data=test_set,
                   nb_epoch=20, batch_size=512, callbacks=callbacks)
    net.save()


def evaluate(net):
    net.load()
    net.compile()
    _, accuracy = net.model.evaluate(*test_set, batch_size=512)
    print('== %s ==\nTest accuracy: %.2f%%' % (net.NAME, accuracy * 100))


def predict(net):
    net.load()
    print(net.model.predict(mnist.test.images))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Play with models')
    p.add_argument('mode', action="store")
    args = p.parse_args()

    func = {
        'train': train,
        'eval': evaluate,
        'test': predict
    }.get(args.mode, evaluate)

    with tf.device('/gpu:0'):
        net = KerasCNN(image_shape=(28, 28, 1))
        func(net)
