"""
Configuration for multiprocessing
"""
from multiprocessing import cpu_count, Pool
from contextlib import contextmanager
import atexit


_dipy_num_cpu = 1
manager = None


def activate_multithreading(num_cpu=None):
    """
    Function to activate multiprocessing.

    Parameters
    ----------
    num_cpu : int
        Number of CPU's.
        default: None
    """
    global _dipy_num_cpu
    global manager
    if num_cpu is None:
        _dipy_num_cpu = cpu_count()
    elif num_cpu <= 0:
        raise ValueError("num_cpu must be positive")
    else:
        _dipy_num_cpu = num_cpu

    if manager is not None:
        manager.shut_down()
        # raise NotImplemented()
    if _dipy_num_cpu > 1:
        manager = PoolMananger(_dipy_num_cpu)
    else:
        manager = None


def deactivate_multithreading():
    """
    Function to deactivate multiprocessing.
    """
    global _dipy_num_cpu
    _dipy_num_cpu = 1


@contextmanager
def multithreading_on(num_cpu=None):
    previous_state = _dipy_num_cpu
    activate_multithreading(num_cpu)
    try:
        yield
    finally:
        activate_multithreading(previous_state)


class PoolMananger(object):

    active = False

    def __init__(self, n_cpu):
        self.n_cpu = n_cpu
        self.pool = Pool(n_cpu)
        self.active = True

    def shut_down(self):
        if self.active:
            self.pool.close()
            self.pool.join()
            self.active = False
            self.pool = None


def cleanup():
    if manager is not None:
        manager.shut_down()

atexit.register(cleanup)
