class DataSubset:

    def __init__(self, x, y, name):
        self.x = self._reshape_img_dim_ordering(x)
        self.y = self._to_categorical(y)
        self.name = name

    def _reshape_img_dim_ordering(self, x):
        from keras import backend as K
        if K.image_dim_ordering() == 'th':
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        else:
            x = x.reshape(*x.shape, 1)
        x = x.astype('float32') / 255
        return x

    def _to_categorical(self, y):
        from keras.utils import np_utils
        return np_utils.to_categorical(y)

    def __repr__(self):
        return '<DataSubset: %s>' % self.name
