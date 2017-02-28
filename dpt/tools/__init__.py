import time
import functools

import tensorflow as tf


def timeit(f):
    @functools.wraps(f)
    def decorator(*args, **kwargs):
        s = time.time()
        result = f(*args, **kwargs)
        print('= Function {}() = elapsed time: {:.6f}'.format(f.__name__, time.time() - s))
        return result
    return decorator


def tf_scope(scope=None):

    def wrapper(f):
        @functools.wraps(f)
        def decorator(*args, **kwargs):
            with tf.name_scope(scope):
                return f(*args, **kwargs)
        return decorator

    return wrapper


def tf_summary(summary_type='scalar', name=None):
    summary_calls = {
        'histogram': tf.summary.histogram,
        'scalar': tf.summary.scalar,
    }
    caller = summary_calls[summary_type]

    def wrapper(f):
        @functools.wraps(f)
        def decorator(*args, **kwargs):
            result = f(*args, **kwargs)
            caller(kwargs.get('name', name), result)
            return result
        return decorator

    return wrapper
