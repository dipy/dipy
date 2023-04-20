import numpy as np
from dipy.nn.utils import normalize, unnormalize, transform_img, recover_img

def test_norm():
    temp = np.random.random((8, 8, 8)) * 10
    temp2 = normalize(temp)
    temp2 = unnormalize(temp2, -1, 1, 0, 10)
    np.testing.assert_almost_equal(temp, temp2, 1)

def test_transform():
    temp = np.random.random((30, 31, 32))
    temp2, new_affine = transform_img(temp, np.eye(4), (32, 32, 32))
    temp2 = recover_img(temp2, new_affine, temp.shape)
    np.testing.assert_almost_equal(np.array(temp.shape), np.array(temp2.shape))