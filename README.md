# deep-insight-tensorflow

## Requirements:
- Python 3.5
- `tensorflow` >= `1.0.0rc`
- `keras`

Can install dependencies through freezed requirements file:
```bash
$ pip install -r requirements.txt
```

### Implementations
Three different implementations:
- Keras: `keras_main.py`
- Tensorflow: `tf_main.py`
- Tensorflow with [TFRecord](https://www.tensorflow.org/how_tos/reading_data/#standard_tensorflow_format): `tfr_record.py`

## Train

```bash
# Train with Keras
$ python keras_main.py train

# Train with Tensorflow and use regular images + labels as input
$ python tf_main.py train

# Train with Tensorflow and use pre-generated TFRecord as input
$ python tfr_main.py train
```

or in short with `Makefile`

```bash
# Train with Keras
$ make keras

# Train with Tensorflow and use regular images + labels as input
$ make tf

# Train with Tensorflow and use pre-generated TFRecord as input
$ make tfr
```

## Evaluate

```bash
# Evaluate with Keras
$ python keras_main.py eval

# Evaluate with Tensorflow and use regular images + labels as input
$ python tf_main.py eval

# Evaluate with Tensorflow and use pre-generated TFRecord as input
$ python tfr_main.py eval
```

## Predict

```bash
$ python keras_main.py predict
```

## Graph
Network of MNIST can visulize in Tensorboard.

![mnist in tensorflow](doc/img/mnist-tsb-graph.png)
