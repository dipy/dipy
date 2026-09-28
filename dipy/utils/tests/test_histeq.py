import numpy as np
import numpy.testing as npt

from dipy.testing.decorators import set_random_number_generator
from dipy.utils.histeq import histeq


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


def test_histeq_num_bins():
    img = np.linspace(0, 1, 100).reshape(10, 10)
    coarse = histeq(img, num_bins=4)
    fine = histeq(img, num_bins=256)
    assert len(np.unique(coarse)) <= len(np.unique(fine))
