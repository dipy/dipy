"""Function for determining the effective number of processes to be used."""

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
