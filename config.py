import os
import yaml

with open('cfg.yaml', 'r') as f:
    cfg = yaml.load(f)

gpu_device_id = cfg['gpu_device_id']
gpu_device = '/gpu:{}'.format(gpu_device_id)

train = cfg['train']
epochs = train['epochs']
batch_size = train['batch_size']
train_dir = train['train_dir']

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device_id)

import tensorflow as tf
config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True

tf.logging.set_verbosity(tf.logging.WARN)
