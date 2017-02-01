import time
import functools


def timeit(f):
    @functools.wraps(f)
    def decorator(*args, **kwargs):
        s = time.time()
        result = f(*args, **kwargs)
        print('= Function {}() = elapsed time: {:.6f}'.format(f.__name__, time.time() - s))
        return result
    return decorator
