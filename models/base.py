import os


class BaseNet:

    NAME = 'NoneNet'
    WEIGHTS_FOLDER_FORMAT = 'data/weights/{}_weights.h5'

    def __init__(self, image_shape=None, **kwargs):
        self.input_shape = self._calc_input_shape(image_shape)
        self.model = self.build_model()

    def _calc_input_shape(self, input_shape):
        from keras import backend as K
        dim_orderings = {'th': (1, *input_shape), 'tf': (*input_shape, 1)}
        return dim_orderings[K.image_dim_ordering()]

    def build_model(self):
        raise Exception('Not implemented')

    def compile(self):
        raise Exception('Not implemented')

    def get_weight_name(self, filename):
        return filename or self.WEIGHTS_FOLDER_FORMAT.format(self.NAME)

    def load(self, filename=None):
        weights_name = self.get_weight_name(filename)
        return self.model.load_weights(weights_name)

    def save(self, filename=None):
        weights_name = self.get_weight_name(filename)
        dir_path = os.path.dirname(weights_name)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        return self.model.save_weights(weights_name)
