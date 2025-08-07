#!python

import os

cimport safe_openmp as openmp
have_openmp = <int> openmp.have_openmp

__all__ = ['have_openmp', 'default_threads', 'cpu_count', 'thread_count']


def cpu_count():
    """Return number of cpus as determined by omp_get_num_procs."""
    if have_openmp:
        return openmp.omp_get_num_procs()
    else:
        return 1


def thread_count():
    """Return number of threads as determined by omp_get_max_threads."""
    if have_openmp:
        return openmp.omp_get_max_threads()
    else:
        return 1


def _get_default_threads():
    """Default number of threads for OpenMP.

    This function prioritizes the OMP_NUM_THREADS environment variable.
    """
    if have_openmp:
        try:
            default_threads = int(os.environ.get('OMP_NUM_THREADS', None))
            if default_threads < 1:
                raise ValueError("invalid number of threads")
        except (ValueError, TypeError):
            default_threads = openmp.omp_get_num_procs()
        return default_threads
    else:
        return 1
default_threads = _get_default_threads()


def determine_num_threads(num_threads):
    """Determine the effective number of threads to be used for OpenMP calls

    - For ``num_threads = None``,
      - if the ``OMP_NUM_THREADS`` environment variable is set, return that
      value
      - otherwise, return the maximum number of cores retrieved by
      ``openmp.opm_get_num_procs()``.

    - For ``num_threads > 0``, return this value.

    - For ``num_threads < 0``, return the maximal number of threads minus
      ``|num_threads + 1|``. In particular ``num_threads = -1`` will use as
      many threads as there are available cores on the machine.

    - For ``num_threads = 0`` a ValueError is raised.

    Parameters
    ----------
    num_threads : int or None
        Desired number of threads to be used.
    """
    if not isinstance(num_threads, int) and num_threads is not None:
        raise TypeError("num_threads must be an int or None")

    if num_threads == 0:
        raise ValueError("num_threads cannot be 0")

    if num_threads is None:
        return default_threads

    if num_threads < 0:
        return max(1, cpu_count() + num_threads + 1)

    return num_threads


cdef void set_num_threads(num_threads):
    """Set the number of threads to be used by OpenMP

    This function does nothing if OpenMP is not available.

    Parameters
    ----------
    num_threads : int
        Effective number of threads for OpenMP accelerated code.
    """
    if openmp.have_openmp:
        openmp.omp_set_dynamic(0)
        openmp.omp_set_num_threads(num_threads)


cdef void restore_default_num_threads():
    """Restore OpenMP to using the default number of threads.

    This function does nothing if OpenMP is not available
    """
    if openmp.have_openmp:
        openmp.omp_set_num_threads(<int> default_threads)


def _set_omp_threads(num_threads):
    """Function for testing set_num_threads."""
    set_num_threads(num_threads)


def _restore_omp_threads():
    """Function for testing restore_default_num_threads."""
    restore_default_num_threads()
