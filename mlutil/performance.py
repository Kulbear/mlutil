import time


def timeit(func):
    def wrap(*args, **kwargs):
        time_flag = time.time()
        temp_result = func(*args, **kwargs)
        print('[{}] Time used: {}'.format(func.__name__,
                                          time.time() - time_flag))
        return temp_result

    return wrap
