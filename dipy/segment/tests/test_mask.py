import warnings

import numpy as np

from scipy.ndimage import generate_binary_structure, binary_dilation
from scipy.ndimage.filters import median_filter

from ..mask import otsu, bounding_box, crop, applymask, multi_median

from numpy.testing import assert_equal, run_module_suite


def test_mask():
    vol = np.zeros((30, 30, 30))
    vol[15, 15, 15] = 1
    struct = generate_binary_structure(3, 1)
    voln = binary_dilation(vol, structure=struct, iterations=4).astype('f4')
    initial = np.sum(voln > 0)
    mask = voln.copy()
    thresh = otsu(mask)
    mask = mask > thresh
    initial_otsu = np.sum(mask > 0)
    assert_equal(initial_otsu, initial)

    mins, maxs = bounding_box(mask)
    voln_crop = crop(mask, mins, maxs)
    initial_crop = np.sum(voln_crop > 0)
    assert_equal(initial_crop, initial)

    applymask(voln, mask)
    final = np.sum(voln > 0)
    assert_equal(final, initial)

    # Test multi_median.
    median_test = np.arange(25).reshape(5,5)
    median_control = median_test.copy()
    medianradius = 3
    median_test = multi_median(median_test, medianradius, 3)

    medarr = np.ones_like(median_control.shape) * ((medianradius * 2) +1)
    median_filter(median_control, medarr, output=median_control)
    median_filter(median_control, medarr, output=median_control)
    median_filter(median_control, medarr, output=median_control)
    assert_equal(median_test, median_control)


def test_bounding_box():
    vol = np.zeros((100, 100, 50), dtype=int)

    # Check the more usual case
    vol[10:90, 11:40, 5:33] = 3
    mins, maxs = bounding_box(vol)
    assert_equal(mins, [10, 11, 5])
    assert_equal(maxs, [90, 40, 33])

    # Check a 2d case
    mins, maxs = bounding_box(vol[10])
    assert_equal(mins, [11, 5])
    assert_equal(maxs, [40, 33])

    vol[:] = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Trigger a warning.
        num_warns = len(w)
        mins, maxs = bounding_box(vol)
        # Assert number of warnings has gone up by 1
        assert_equal(len(w), num_warns + 1)

        # Check that an empty array returns zeros for both min & max
        assert_equal(mins, [0, 0, 0])
        assert_equal(maxs, [0, 0, 0])

        # Check the 2d case
        mins, maxs = bounding_box(vol[0])
        assert_equal(len(w), num_warns + 2)
        assert_equal(mins, [0, 0])
        assert_equal(maxs, [0, 0])


if __name__ == '__main__':
    run_module_suite()
