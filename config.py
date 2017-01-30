''' Use only one gpu (device #1) '''
import os
gpu_dev = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_dev)

import tensorflow as tf
config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True

tf.logging.set_verbosity(tf.logging.WARN)

# TODO: FLAGS from tf.app.run()
