#!python

import os
cimport cython
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


cdef void set_num_threads(num_threads):
    """Set the number of threads to be used by OpenMP

    This function does nothing if OpenMP is not available.

    Parameters
    ----------
    num_threads : int
        Desired number of threads for OpenMP accelerated code.
    """
    cdef:
        int threads_to_use
    if num_threads is not None:
        threads_to_use = num_threads
    else:
        threads_to_use = <int> default_threads

    if openmp.have_openmp:
        openmp.omp_set_dynamic(0)
        openmp.omp_set_num_threads(threads_to_use)


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
