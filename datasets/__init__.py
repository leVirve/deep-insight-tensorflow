from tensorflow.examples.tutorials.mnist import input_data


class MNist:

    path = 'data/mnist-data'
    image_shape = (28, 28, 1)
    classes = 10

    def __init__(self, batch_size, *args, **kwargs):
        self.raw = input_data.read_data_sets(self.path, one_hot=True, *args, **kwargs)
        self.batch_size = batch_size
        self.num_train_batch = self.raw.train.num_examples // batch_size

    def next_batch(self):
        return self.raw.train.next_batch(self.batch_size)

    @property
    def test(self):
        return self.raw.test

    @property
    def train(self):
        return self.raw.train
