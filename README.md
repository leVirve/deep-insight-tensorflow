# dpt

## Requirements:
- Python 3.5
- `tensorflow` or `theano`
- `keras`

Can install through freezed requirements file:
```bash
$ pip install -r requirements.txt
```

## Train

Trainning on specific devices (`with tf.device` statement seems not working currently)
```bash
CUDA_VISIBLE_DEVICES=1 python main.py train
```

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=1 python main.py eval
```

## Predict

```bash
CUDA_VISIBLE_DEVICES=1 python main.py predict
```
