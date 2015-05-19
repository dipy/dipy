from multiprocessing import cpu_count
from contextlib import contextmanager

_dipy_num_cpu = 1

def activate_multithreading(num_cpu=None):
    global _dipy_num_cpu
    if num_cpu is None:
        _dipy_num_cpu = cpu_count()
    elif num_cpu > 0:
        _dipy_num_cpu = num_cpu

def deactivate_multithreading():
    global _dipy_num_cpu
    _dipy_num_cpu = 1

@contextmanager
def multithreading_on(num_cpu=None):
    previous_state = _dipy_num_cpu
    activate_multithreading(num_cpu)
    yield
    activate_multithreading(previous_state)

