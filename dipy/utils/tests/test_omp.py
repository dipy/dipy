""" Testing OpenMP utilities
"""

import os
from dipy.utils.omp import (cpu_count, thread_count, default_threads,
                            _set_omp_threads, _restore_omp_threads,
                            have_openmp, determine_num_threads)

import pytest
from dipy.utils.parallel import has_ray, has_joblib
from numpy.testing import assert_equal, assert_raises


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


@pytest.mark.skipif(has_joblib or has_ray,
                    reason="joblib and/or ray are installed")
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


def test_determine_num_threads():
    # 0 should raise an error
    assert_raises(ValueError, determine_num_threads, 0)

    # A string should raise an error
    assert_raises(TypeError, determine_num_threads, "1")

    # 1 should be 1
    assert_equal(determine_num_threads(1), 1)

    # A positive integer should not change
    assert_equal(determine_num_threads(4), 4)

    # A big negative number should be 1
    assert_equal(determine_num_threads(-10000), 1)

    # -2 should be one less than -1 (if there are more than 1 cores)
    if determine_num_threads(-1) > 1:
        assert_equal(determine_num_threads(-1),
                     determine_num_threads(-2) + 1)
