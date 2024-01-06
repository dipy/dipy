import warnings
import numpy as np
import numpy.testing as npt
from dipy.viz.horizon.util import check_img_dtype, check_img_shapes


def test_check_img_shapes():
    affine = np.array([[1., 0., 0., -98.],
                       [0., 1., 0., -134.],
                       [0., 0., 1., -72.],
                       [0., 0., 0., 1.]])

    data = 255 * np.random.rand(197, 233, 189)
    data1 = 255 * np.random.rand(197, 233, 189)
    images = [
        (data, affine),
        (data1, affine)
    ]
    npt.assert_equal(check_img_shapes(images), True)

    data1 = 255 * np.random.rand(200, 233, 189)
    images = [
        (data, affine),
        (data1, affine)
    ]
    npt.assert_equal(check_img_shapes(images), False)


def check_for_warnings(warn_printed, w_msg):
    selected_w = [w for w in warn_printed if issubclass(w.category,
                                                        UserWarning)]
    assert len(selected_w) >= 1
    msg = [str(m.message) for m in selected_w]
    npt.assert_equal(w_msg in msg, True)


def test_check_img_dtype():
    affine = np.array([[1., 0., 0., -98.],
                       [0., 1., 0., -134.],
                       [0., 0., 1., -72.],
                       [0., 0., 0., 1.]])

    data = 255 * np.random.rand(197, 233, 189)
    images = [
        (data, affine),
    ]

    npt.assert_equal(check_img_dtype(images)[0], images[0])

    data = np.random.rand(5, 5, 5).astype(np.int64)
    images = [
        (data, affine),
    ]

    with warnings.catch_warnings(record=True) as l_warns:
        npt.assert_equal(check_img_dtype(images)[0][0].dtype, np.int32)
        check_for_warnings(l_warns, 'int64 is not supported, falling back to'
                           + ' int32')

    data = np.random.rand(5, 5, 5).astype(np.float16)
    images = [
        (data, affine),
    ]

    with warnings.catch_warnings(record=True) as l_warns:
        npt.assert_equal(check_img_dtype(images)[0][0].dtype, np.float32)
        check_for_warnings(l_warns, 'float16 is not supported, falling back to'
                           + ' float32')

    data = np.random.rand(5, 5, 5).astype(np.bool_)
    images = [
        (data, affine),
    ]

    with warnings.catch_warnings(record=True) as l_warns:
        check_img_dtype(images)
        check_for_warnings(l_warns, 'skipping image 1, passed image is not in '
                           + 'numerical format')
