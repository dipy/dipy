import warnings

import numpy as np
from dipy.nn.utils import normalize, unnormalize, transform_img, recover_img
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator()
def test_norm(rng=None):
    temp = rng.random((8, 8, 8)) * 10
    temp2 = normalize(temp)
    temp2 = unnormalize(temp2, -1, 1, 0, 10)
    np.testing.assert_almost_equal(temp, temp2, 1)


@set_random_number_generator()
def test_transform(rng=None):
    temp = rng.random((30, 31, 32))
    temp2, new_affine, ori_shape = transform_img(temp, np.eye(4),
                                                 init_shape=(32, 32, 32),
                                                 voxsize=np.ones(3)*2)
    with warnings.catch_warnings():
        scipy_affine_txfm_msg = (
            "The behavior of affine_transform with a 1-D "
            "array supplied for the matrix parameter has changed in "
            "SciPy 0.18.0."
        )
        warnings.filterwarnings(
            "ignore", message=scipy_affine_txfm_msg,
            category=UserWarning)
        temp2 = recover_img(temp2, new_affine, ori_shape, temp.shape,
                            init_shape=(32, 32, 32), voxsize=np.ones(3)*2)
    np.testing.assert_almost_equal(np.array(temp.shape), np.array(temp2.shape))
