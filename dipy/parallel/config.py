from multiprocessing import cpu_count
from contextlib import contextmanager

_dipy_num_cpu = 1

def active_multithreading(num_cpu=None):
    global _dipy_num_cpu
    if num_cpu is None:
        _dipy_num_cpu = cpu_count()
    elif num_cpu > 0:
        _dipy_num_cpu = num_cpu

def deactive_multithreading():
    global _dipy_num_cpu
    _dipy_num_cpu = 1

@contextmanager
def multitheading_on(num_cpu=None):
    print '{} in'.format(num_cpu)
    previous_state = _dipy_num_cpu
    active_multithreading(num_cpu)
    yield
    active_multithreading(previous_state)
    print '{} out'.format(_dipy_num_cpu)

