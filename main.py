import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.network import CNet

mnist = input_data.read_data_sets("data/mnist-data", one_hot=True, reshape=False)


def main(args):

    train = (mnist.train.images, mnist.train.labels)
    test = (mnist.test.images, mnist.test.labels)

    cnet = CNet(train=True, image_shape=(28, 28))
    model = cnet.model

    if args.mode == 'train':

        from keras.callbacks import TensorBoard
        callbacks = [
                TensorBoard(log_dir='./logs', histogram_freq=2, write_graph=True, write_images=False)
            ]
        model.fit(*train, validation_data=test, nb_epoch=10, batch_size=512, callbacks=callbacks)
        cnet.save()

    elif args.mode == 'eval':

        cnet.load()
        _, accuracy = model.evaluate(*test, batch_size=512)
        print('== %s ==\nTest accuracy: %.2f%%' % (cnet.NAME, accuracy * 100))

    else:

        cnet = CNet(image_shape=mnist.image_size)
        cnet.load()
        print(cnet.model.predict(test[0]))

    import gc
    gc.collect()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Play with models')
    p.add_argument('mode', action="store")
    args = p.parse_args()

    with tf.device('/gpu:0'):
        main(args)
