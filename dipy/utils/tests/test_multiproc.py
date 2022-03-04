""" Testing multiproc utilities
"""
import os

from dipy.utils.multiproc import (determine_num_processes, disable_np_threads,
                                  enable_np_threads)
from numpy.testing import assert_equal, assert_raises


def test_determine_num_processs():
    # Test that the correct number of effective num_processes is returned

    # 0 should raise an error
    assert_raises(ValueError, determine_num_processes, 0)

    # A string should raise an error
    assert_raises(TypeError, determine_num_processes, "0")

    # 1 should be 1
    assert_equal(determine_num_processes(1), 1)

    # A positive integer should not change
    assert_equal(determine_num_processes(4), 4)

    # None and -1 should be equal (all cores)
    assert_equal(determine_num_processes(None), determine_num_processes(-1))

    # A big negative number should be 1
    assert_equal(determine_num_processes(-10000), 1)

    # -2 should be one less than -1 (if there are more than 1 cores)
    if determine_num_processes(-1) > 1:
        assert_equal(determine_num_processes(-1),
                     determine_num_processes(-2) + 1)


def test_np_threads():
    openblas_num_threads = os.environ.get('OPENBLAS_NUM_THREADS', '')
    mkl_num_threads = os.environ.get('MKL_NUM_THREADS', '')

    disable_np_threads()
    assert_equal(os.environ.get('OPENBLAS_NUM_THREADS', ''), '1')
    assert_equal(os.environ.get('MKL_NUM_THREADS', ''), '1')
    assert_equal(os.environ.get('DIPY_OPENBLAS_NUM_THREADS', ''),
                 openblas_num_threads)
    assert_equal(os.environ.get('DIPY_MKL_NUM_THREADS', ''), mkl_num_threads)

    enable_np_threads()
    assert_equal(os.environ.get('OPENBLAS_NUM_THREADS', ''),
                 openblas_num_threads)
    assert_equal(os.environ.get('MKL_NUM_THREADS', ''), mkl_num_threads)


