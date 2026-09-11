import warnings

import numpy as np

from dipy.nn.utils import (
    get_padded_shape,
    normalize,
    recover_img,
    transform_img,
    unnormalize,
)
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator()
def test_norm(rng=None):
    temp = rng.random((8, 8, 8)) * 10
    temp2 = normalize(temp)
    temp2 = unnormalize(temp2, -1, 1, 0, 10)
    np.testing.assert_almost_equal(temp, temp2, 1)


@set_random_number_generator()
def test_transform(rng=None):
    temp = rng.random((28, 30, 34))
    temp2, params = transform_img(
        temp,
        np.eye(4),
        target_voxsize=tuple(np.ones(3) * 2),
        final_size=(14, 15, 16),
    )
    with warnings.catch_warnings():
        scipy_affine_txfm_msg = (
            "The behavior of affine_transform with a 1-D "
            "array supplied for the matrix parameter has changed in "
            "SciPy 0.18.0."
        )
        warnings.filterwarnings(
            "ignore", message=scipy_affine_txfm_msg, category=UserWarning
        )
        temp2 = recover_img(temp2, params)
    np.testing.assert_almost_equal(np.array(temp.shape), np.array(temp2.shape))


def test_get_padded_shape():
    cases = [
        ((32, 64, 96), 32, (32, 64, 96)),
        ((33, 65, 95), 32, (64, 96, 96)),
        ((1, 31, 32), 32, (32, 32, 32)),
        ((5, 10), 4, (8, 12)),
    ]

    for shape, multiple, expected_shape in cases:
        np.testing.assert_equal(get_padded_shape(shape, multiple), expected_shape)

    for multiple in [0, -1]:
        np.testing.assert_raises(ValueError, get_padded_shape, (32, 32, 32), multiple)
