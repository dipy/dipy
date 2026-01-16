import warnings

import numpy as np

from dipy.nn.utils import normalize, recover_img, transform_img, unnormalize
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
