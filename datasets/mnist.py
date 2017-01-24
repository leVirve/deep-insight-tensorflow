from datasets.base import DataSubset


class MNist:

    image_size = (28, 28)

    def __init__(self):
        self.train_set = None
        self.test_set = None

    def load(self):
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.train_set = DataSubset(x_train, y_train, 'Train')
        self.test_set = DataSubset(x_test, y_test, 'Test')

        return self.train_set, self.test_set
