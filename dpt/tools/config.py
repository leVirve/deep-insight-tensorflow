import os
import yaml
from easydict import EasyDict


def create_parent_dir(fullpath):
    dir_path = os.path.dirname(fullpath)
    os.makedirs(dir_path, exist_ok=True)


def setup_gpu_env(device_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)


def build_tf_config():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    return config


def build_tfrecord_config(train_cfg):
    tfr_cfg = train_cfg.tfrecord
    param_test = {
        'batch_size': train_cfg.batch_size,
        'num_threads': tfr_cfg.num_threads,
        'capacity': tfr_cfg.capacity if tfr_cfg.capacity else tfr_cfg.min_after_dequeue * 3 + train_cfg.batch_size,
    }
    param_train = {'min_after_dequeue': tfr_cfg.min_after_dequeue, **param_test}
    return EasyDict(train=param_train, test=param_test)

with open('config.yml', 'r') as f:
    cfg = EasyDict(yaml.load(f))


# export first level dict
train = cfg.train
test = cfg.test
model = cfg.model

# initialization functions
create_parent_dir(model.model_path)
setup_gpu_env(cfg.gpu_device_id)

# exported config set
tf_config = build_tf_config()
batcher_params = build_tfrecord_config(train)
gpu_device = '/gpu:{}'.format(cfg.gpu_device_id)
