# dpt

## Requirements:
- Python 3.5
- `tensorflow` >= 1.0.0rc
- `keras`

Can install through freezed requirements file:
```bash
$ pip install -r requirements.txt
```

## Train

Trainning on specific devices (`with tf.device` statement seems not working currently)
```bash
$ CUDA_VISIBLE_DEVICES=1 python main.py train
```

## Evaluate

```bash
$ CUDA_VISIBLE_DEVICES=1 python main.py eval
```

## Predict

```bash
$ CUDA_VISIBLE_DEVICES=1 python main.py predict
```

## Graph
Network of MNIST can visulize in Tensorboard.

![mnist in tensorflow](doc/img/mnist-tsb-graph.png)

## TODO
- merge tf_main.py tfr_main.py
- merge tf main