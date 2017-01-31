import os
import yaml


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

with open('cfg.yaml', 'r') as f:
    cfg = yaml.load(f)

gpu_device_id = cfg['gpu_device_id']
gpu_device = '/gpu:{}'.format(gpu_device_id)

train = cfg['train']
epochs = train['epochs']
batch_size = train['batch_size']
train_dir = train['train_dir']

tfrecord = train['tfrecord']
min_after_dequeue = tfrecord['min_after_dequeue']
num_threads = tfrecord['num_threads']
preprocess_level = tfrecord['preprocess_level']
capacity = min_after_dequeue + 3 * batch_size

model = cfg['model']
model_path = model['model_path']
model_dir = model['model_dir']

create_parent_dir(model_path)
setup_gpu_env(gpu_device_id)

config = build_tf_config()
