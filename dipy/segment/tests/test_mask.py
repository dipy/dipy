import numpy as np
from numpy.testing import assert_equal, run_module_suite
from scipy.ndimage import generate_binary_structure, binary_dilation
from dipy.segment.mask import (medotsu, otsu, binary_threshold,
                               bounding_box, crop, applymask)


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

    # How do we test median_filter? median_filtering is not made
    # to do well on binary noisy images, it is not the right denoising
    #
    # On natural images or medical images, it does a much better job
        
if __name__ == '__main__':
    run_module_suite()


