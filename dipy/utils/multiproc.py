"""Function for determining the effective number of processes to be used."""

import os
from multiprocessing import cpu_count
from warnings import warn


def determine_num_processes(num_processes):
    """Determine the effective number of processes for parallelization.

    - For `num_processes = None`` return the maximum number of cores retrieved
    by cpu_count().

    - For ``num_processes > 0``, return this value.

    - For ``num_processes < 0``, return the maximal number of cores minus
    ``num_processes + 1``. In particular ``num_processes = -1`` will use as
    many cores as possible.

    - For ``num_processes = 0`` a ValueError is raised.

    Parameters
    ----------
    num_processes : int or None
        Desired number of processes to be used.
    """
    if not isinstance(num_processes, int) and num_processes is not None:
        raise TypeError("num_processes must be an int or None")

    if num_processes == 0:
        raise ValueError("num_processes cannot be 0")

    try:
        if num_processes is None:
            return cpu_count()

        if num_processes < 0:
            return max(1, cpu_count() + num_processes + 1)
    except NotImplementedError:
        warn("Cannot determine number of cores. Using only 1.")
        return 1

    return num_processes


def disable_np_threads():
    """Reduce OPENBLAS and MKL thread numbers.

    Notes
    -----
    The goal of this function is to avoid oversubscription by spawning too
    many threads when you use multiprocessing module.

    See Also
    --------
    ``enable_np_threads``
    """
    current_openblas = os.environ.get('OPENBLAS_NUM_THREADS', '')
    current_mkl = os.environ.get('MKL_NUM_THREADS', '')

    # import ipdb; ipdb.set_trace()
    os.environ['DIPY_OPENBLAS_NUM_THREADS'] = current_openblas
    os.environ['DIPY_MKL_NUM_THREADS'] = current_mkl
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def enable_np_threads():
    """Reactivate OPENBLAS and MKL thread numbers."""
    if 'DIPY_OPENBLAS_NUM_THREADS' in os.environ:
        os.environ['OPENBLAS_NUM_THREADS'] = \
            os.environ.pop('DIPY_OPENBLAS_NUM_THREADS', '')
        if os.environ['OPENBLAS_NUM_THREADS'] in ['', None]:
            os.environ.pop('OPENBLAS_NUM_THREADS', '')

    if 'DIPY_MKL_NUM_THREADS' in os.environ:
        os.environ['MKL_NUM_THREADS'] = \
            os.environ.pop('DIPY_MKL_NUM_THREADS', '')
        if os.environ['MKL_NUM_THREADS'] in ['', None]:
            os.environ.pop('MKL_NUM_THREADS', '')
