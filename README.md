# dpt

## Requirements:
- Python 3.5
- `tensorflow` >= `1.0.0rc`
- `keras`

Can install through freezed requirements file:
```bash
$ pip install -r requirements.txt
```

### Implementations
Three different implementations:
- Keras: `keras_main.py`
- Tensorflow: `tf_main.py`
- Tensorflow with tfrecord: `tfr_record.py`

## Train

```bash
$ CUDA_VISIBLE_DEVICES=1 python keras_main.py train
$ python tf_main.py train
$ python tfr_main.py train
```

## Evaluate

```bash
$ CUDA_VISIBLE_DEVICES=1 python keras_main.py eval
$ python tf_main.py eval
$ python tfr_main.py eval
```

## Predict

```bash
$ CUDA_VISIBLE_DEVICES=1 python keras_main.py predict
```

## Graph
Network of MNIST can visulize in Tensorboard.

![mnist in tensorflow](doc/img/mnist-tsb-graph.png)

## TODO
- merge tf_main.py tfr_main.py
- merge tf main