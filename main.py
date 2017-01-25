import argparse

from datasets.mnist import MNist
from models.network import CNet


def main(args):

    mnist = MNist()
    train_set, test_set = mnist.load()
    train = (train_set.x, train_set.y)
    test = (test_set.x, test_set.y)

    import tensorflow as tf
    with tf.device('/gpu:0'):
        cnet = CNet(train=True, image_shape=mnist.image_size)
        model = cnet.model

    if args.mode == 'train':

        model.fit(*train, validation_data=test, nb_epoch=1, batch_size=512)
        cnet.save()

    elif args.mode == 'eval':

        cnet.load()
        _, accuracy = model.evaluate(*test, batch_size=512)
        print('== %s ==\nTest accuracy: %.2f%%' % (cnet.NAME, accuracy * 100))

    else:

        cnet = CNet(image_shape=mnist.image_size)
        cnet.load()
        print(cnet.model.predict(test[0]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Play with models')
    p.add_argument('mode', action="store")
    args = p.parse_args()

    main(args)
