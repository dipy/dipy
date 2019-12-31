import warnings

import numpy as np
from scipy.ndimage import generate_binary_structure, binary_dilation
from scipy.ndimage.filters import median_filter

from dipy.segment.mask import (otsu, bounding_box, crop, applymask,
                               multi_median, median_otsu)

from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_raises,
                           run_module_suite)
from dipy.data import get_fnames
from dipy.io.image import load_nifti_data


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
    median_test = np.arange(25).reshape(5, 5)
    median_control = median_test.copy()
    medianradius = 3
    median_test = multi_median(median_test, medianradius, 3)

    medarr = np.ones_like(median_control.shape) * ((medianradius * 2) + 1)
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


def test_median_otsu():
    fname = get_fnames('S0_10')
    data = load_nifti_data(fname)
    data = np.squeeze(data.astype('f8'))
    dummy_mask = data > data.mean()
    data_masked, mask = median_otsu(data, median_radius=3, numpass=2,
                                    autocrop=False, vol_idx=None,
                                    dilate=None)
    assert_equal(mask.sum() < dummy_mask.sum(), True)
    data2 = np.zeros(data.shape + (2,))
    data2[..., 0] = data
    data2[..., 1] = data

    data2_masked, mask2 = median_otsu(data2, median_radius=3, numpass=2,
                                      autocrop=False, vol_idx=[0, 1],
                                      dilate=None)
    assert_almost_equal(mask.sum(), mask2.sum())

    _, mask3 = median_otsu(data2, median_radius=3, numpass=2,
                           autocrop=False, vol_idx=[0, 1],
                           dilate=1)
    assert_equal(mask2.sum() < mask3.sum(), True)

    _, mask4 = median_otsu(data2, median_radius=3, numpass=2,
                           autocrop=False, vol_idx=[0, 1],
                           dilate=2)
    assert_equal(mask3.sum() < mask4.sum(), True)

    # For 4D volumes, can't call without vol_idx input:
    assert_raises(ValueError, median_otsu, data2)


if __name__ == '__main__':
    run_module_suite()
