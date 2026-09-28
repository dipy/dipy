import numpy as np
import numpy.testing as npt

from dipy.testing.decorators import set_random_number_generator
from dipy.utils.histeq import histeq


def _uniformity_error(arr):
    """Spread of the output histogram; smaller means better equalized."""
    counts, _ = np.histogram(arr, bins=16, range=(0, 255))
    return counts.std()


@set_random_number_generator()
def test_histeq(rng=None):
    img = rng.random((16, 24, 8)) ** 3

    out = histeq(img)

    npt.assert_equal(out.shape, img.shape)
    assert out.min() >= 0
    assert out.max() <= 255
    flat_in = img.ravel()
    flat_out = out.ravel()
    order = np.argsort(flat_in)
    assert np.all(np.diff(flat_out[order]) >= 0)


@set_random_number_generator()
def test_histeq_num_bins(rng=None):
    img = rng.random((16, 24, 8)) ** 3

    coarse = histeq(img, num_bins=4)
    fine = histeq(img, num_bins=256)

    assert not np.allclose(coarse, fine)
    assert _uniformity_error(fine) < _uniformity_error(coarse)
