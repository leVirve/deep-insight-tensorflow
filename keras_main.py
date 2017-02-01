from dpt.tools import config as cfg

import argparse

from dpt.network import KerasCNN
from dpt.dataset import MNist

dataset = MNist(batch_size=cfg.batch_size, reshape=False)
train_set = (dataset.raw.train.images, dataset.raw.train.labels)
test_set = (dataset.raw.test.images, dataset.raw.test.labels)


def train(net):
    from keras.callbacks import TensorBoard
    callbacks = [
            TensorBoard(
                log_dir=cfg.train_dir,
                histogram_freq=2, write_graph=True, write_images=False)
        ]
    net.compile()
    net.model.fit(*train_set, validation_data=test_set,
                   nb_epoch=cfg.epochs, batch_size=cfg.batch_size, callbacks=callbacks)
    net.save()


def evaluate(net):
    net.load()
    net.compile()
    _, accuracy = net.model.evaluate(*test_set, batch_size=cfg.batch_size)
    print('== %s ==\nTest accuracy: %.2f%%' % (net.NAME, accuracy * 100))


def predict(net):
    net.load()
    print(net.model.predict(dataset.raw.test.images))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Play with models')
    p.add_argument('mode', action="store")
    args = p.parse_args()

    func = {
        'train': train,
        'eval': evaluate,
        'test': predict
    }.get(args.mode, evaluate)

    net = KerasCNN(image_shape=dataset.image_shape)
    func(net)
