import os


class BaseNet:

    NAME = 'NoneNet'
    WEIGHTS_FOLDER_FORMAT = 'data/weights/{}_weights.h5'

    def __init__(self, image_shape=None, train=False, **kwargs):
        self.input_shape = self._calc_input_shape(image_shape)
        self.model = self.build_model(train)

    def _calc_input_shape(self, input_shape):
        from keras import backend as K
        dim_orderings = {'th': (1, *input_shape), 'tf': (*input_shape, 1)}
        return dim_orderings[K.image_dim_ordering()]

    def build_model(self):
        raise Exception('Not implemented')

    def save(self, filename=None):
        model_name = filename or self.WEIGHTS_FOLDER_FORMAT.format(self.NAME)
        dir_path = os.path.dirname(model_name)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        return self.save_weights(model_name)
