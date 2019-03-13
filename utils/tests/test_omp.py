""" Testing OpenMP utilities
"""

import os
from dipy.utils.omp import (cpu_count, thread_count, default_threads,
                            _set_omp_threads, _restore_omp_threads,
                            have_openmp)
from numpy.testing import assert_equal, run_module_suite


def test_set_omp_threads():
    if have_openmp:
        # set threads to default
        _restore_omp_threads()
        assert_equal(thread_count(), default_threads)

        # change number of threads
        nthreads_new = default_threads + 1
        _set_omp_threads(nthreads_new)
        assert_equal(thread_count(), nthreads_new)

        # restore back to default
        _restore_omp_threads()
        assert_equal(thread_count(), default_threads)
    else:
        assert_equal(thread_count(), 1)
        assert_equal(cpu_count(), 1)


def test_default_threads():
    if have_openmp:
        try:
            expected_threads = int(os.environ.get('OMP_NUM_THREADS', None))
            if expected_threads < 1:
                raise ValueError("invalid number of threads")
        except (ValueError, TypeError):
            expected_threads = cpu_count()
    else:
        expected_threads = 1
    assert_equal(default_threads, expected_threads)


if __name__ == '__main__':

    run_module_suite()
