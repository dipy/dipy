import numpy as np
from numpy.testing import assert_equal, run_module_suite
from scipy.ndimage import generate_binary_structure, binary_dilation
from scipy.ndimage.filters import median_filter
from dipy.segment.mask import (medotsu, otsu, binary_threshold,
                               bounding_box, crop, applymask, multi_median)



def test_mask():
    vol = np.zeros((30, 30, 30))
    vol[15, 15, 15] = 1
    
    struct = generate_binary_structure(3, 1)
    voln = binary_dilation(vol, structure=struct, iterations=4).astype('f4')
    initial = np.sum(voln > 0)

    mask = voln.copy()

    thresh = otsu(mask)
    binary_threshold(mask, thresh)
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

        
if __name__ == '__main__':
    run_module_suite()


