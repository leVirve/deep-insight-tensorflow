import os


class BaseNet:

    NAME = 'NoneNet'
    WEIGHTS_FOLDER_FORMAT = 'data/weights/{}_weights.h5'

    def __init__(self, image_shape=None, train=False, **kwargs):
        self.epoch = kwargs.get('epoch') or 500
        self.batch_size = kwargs.get('batch_size') or 32
        self.verbose = 1 if kwargs.get('verbose') else 0
        self.input_shape = self._calc_input_shape(image_shape)

        self.model = self.build_model(train)
        self._register_methods()

    def _calc_input_shape(self, input_shape):
        from keras import backend as K
        dim_orderings = {'th': (1, *input_shape), 'tf': (*input_shape, 1)}
        return dim_orderings[K.image_dim_ordering()]

    def _register_methods(self):
        self.evaluate = self.model.evaluate
        self.fit = self.model.fit
        self.predict = self.model.predict
        self.load_weights = self.model.load_weights
        self.save_weights = self.model.save_weights

    def save_model(self, filename=None):
        model_name = filename or self.WEIGHTS_FOLDER_FORMAT.format(self.NAME)
        dir_path = os.path.dirname(model_name)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        return self.model.save_weights(model_name)

    def build_model(self):
        raise Exception('Not implemented')

    def run(self, train, validate, save=False, **kwargs):
        self.fit(*train,
                 nb_epoch=self.epoch,
                 batch_size=self.batch_size,
                 validation_data=validate,
                 verbose=self.verbose)
        _, accuracy = self.evaluate(*validate, batch_size=32, verbose=0)
        if save:
            self.save_model(kwargs.get('filename'))
        print('=== %s ===\nTest accuracy: %.2f%%' %
              (self.NAME, accuracy * 100))
