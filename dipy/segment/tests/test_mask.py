import numpy as np
from numpy.testing import assert_equal, run_module_suite
from scipy.ndimage import generate_binary_structure, binary_dilation
from dipy.segment.mask import hist_mask


def test_mask():

    vol = np.zeros((30, 30, 30))
    vol[15, 15, 15] = 1
    struct = generate_binary_structure(3, 1)
    voln = binary_dilation(vol, structure=struct, iterations=4).astype('f4')
    initial = np.sum(voln > 0)
    voln = 5 * voln + np.random.random(voln.shape)
    mask = hist_mask(voln, m=0.9, M=.99)
    assert_equal(np.sum(mask > 0), initial)
    # subplot(211)
    # imshow(voln[:,:,15])
    # subplot(212)
    # imshow(mask[:,:,15])
    # show()


if __name__ == '__main__':
    run_module_suite()
    