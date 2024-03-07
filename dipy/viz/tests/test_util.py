import warnings
import numpy as np
import numpy.testing as npt
from dipy.direction.peaks import PeaksAndMetrics
from dipy.testing import check_for_warnings
from dipy.testing.decorators import set_random_number_generator
from dipy.viz.horizon.util import (check_img_dtype, check_img_shapes,
                                   check_peak_size, is_binary_image,
                                   show_ellipsis, unpack_surface)


@set_random_number_generator()
def test_check_img_shapes(rng):
    affine = np.array([[1., 0., 0., -98.],
                       [0., 1., 0., -134.],
                       [0., 0., 1., -72.],
                       [0., 0., 0., 1.]])

    data = 255 * rng.random((197, 233, 189))
    data1 = 255 * rng.random((197, 233, 189))
    images = [
        (data, affine),
        (data1, affine)
    ]
    npt.assert_equal(check_img_shapes(images), (True, False))

    data1 = 255 * rng.random((200, 233, 189))
    images = [
        (data, affine),
        (data1, affine)
    ]
    npt.assert_equal(check_img_shapes(images), (False, False))

    data = 255 * rng.random((197, 233, 189, 10))
    data1 = 255 * rng.random((197, 233, 189))
    images = [
        (data, affine),
        (data1, affine)
    ]

    npt.assert_equal(check_img_shapes(images), (True, True))

    data = 255 * rng.random((197, 233, 189, 15))
    data1 = 255 * rng.random((197, 233, 189, 15))
    images = [
        (data, affine),
        (data1, affine)
    ]

    npt.assert_equal(check_img_shapes(images), (True, True))

    data = 255 * rng.random((198, 233, 189, 14))
    data1 = 255 * rng.random((198, 233, 189, 15))
    images = [
        (data, affine),
        (data1, affine)
    ]

    npt.assert_equal(check_img_shapes(images), (True, False))

    data = 255 * rng.random((197, 233, 189, 15))
    data1 = 255 * rng.random((198, 233, 189, 14))
    images = [
        (data, affine),
        (data1, affine)
    ]

    npt.assert_equal(check_img_shapes(images), (False, False))

    data = 255 * rng.random((197, 233, 189, 15))
    data1 = 255 * rng.random((198, 233, 189, 15))
    images = [
        (data, affine),
        (data1, affine)
    ]

    npt.assert_equal(check_img_shapes(images), (False, False))


@set_random_number_generator()
def test_check_img_dtype(rng):
    affine = np.array([[1., 0., 0., -98.],
                       [0., 1., 0., -134.],
                       [0., 0., 1., -72.],
                       [0., 0., 0., 1.]])

    data = 255 * rng.random((197, 233, 189))
    images = [
        (data, affine),
    ]

    npt.assert_equal(check_img_dtype(images)[0], images[0])

    data = rng.random((5, 5, 5)).astype(np.int64)
    images = [
        (data, affine),
    ]

    with warnings.catch_warnings(record=True) as l_warns:
        npt.assert_equal(check_img_dtype(images)[0][0].dtype, np.int32)
        check_for_warnings(l_warns, 'int64 is not supported, falling back to'
                           + ' int32')

    data = rng.random((5, 5, 5)).astype(np.float16)
    images = [
        (data, affine),
    ]

    with warnings.catch_warnings(record=True) as l_warns:
        npt.assert_equal(check_img_dtype(images)[0][0].dtype, np.float32)
        check_for_warnings(l_warns, 'float16 is not supported, falling back to'
                           + ' float32')

    data = rng.random((5, 5, 5)).astype(np.bool_)
    images = [
        (data, affine),
    ]

    with warnings.catch_warnings(record=True) as l_warns:
        check_img_dtype(images)
        check_for_warnings(l_warns, 'skipping image 1, passed image is not in '
                           + 'numerical format')


def test_show_ellipsis():
    text = 'IAmALongFileName'
    text_size = 10
    available_size = 5
    result_text = '...' + text[-5:]

    npt.assert_equal(show_ellipsis(text, text_size, available_size),
                     result_text)

    available_size = 12
    npt.assert_equal(show_ellipsis(text, text_size, available_size), text)


@set_random_number_generator()
def test_is_binary_image(rng):
    data = 255 * rng.random((197, 233, 189))
    npt.assert_equal(False, is_binary_image(data))

    data = rng.integers(0, 1, size=(10, 20, 100, 200))

    npt.assert_equal(True, is_binary_image(data))


@set_random_number_generator()
def test_unpack_surface(rng):
    vertices = rng.random((100, 4))
    faces = rng.integers(0, 100, size=(100, 3))

    with npt.assert_raises(ValueError):
        unpack_surface((vertices, faces))

    vertices = rng.random((100, 3))
    faces = rng.integers(0, 100, size=(100, 4))

    with npt.assert_raises(ValueError):
        unpack_surface((vertices, faces, '/test/filename.pial'))

    vertices = rng.random((100, 3))
    faces = rng.integers(0, 100, size=(100, 3))
    v, f, fname = unpack_surface((vertices, faces, '/test/filename.pial'))
    npt.assert_equal(vertices, v)
    npt.assert_equal(faces, f)
    npt.assert_equal('/test/filename.pial', fname)


@set_random_number_generator()
def test_check_peak_size(rng):
    peak_dirs = rng.random((100, 100, 100, 10, 6))

    pam = PeaksAndMetrics()
    pam.peak_dirs = peak_dirs

    npt.assert_equal(True, check_peak_size([pam]))
    npt.assert_equal(True, check_peak_size([pam, pam]))
    npt.assert_equal(False, check_peak_size([pam], (100, 100, 1), True))
    npt.assert_equal(False, check_peak_size([pam], (100, 100, 100), False))

    pam1 = PeaksAndMetrics()
    peak_dirs_1 = rng.random((100, 100, 50, 10, 6))
    pam1.peak_dirs = peak_dirs_1

    npt.assert_equal(False, check_peak_size([pam, pam1]))
    npt.assert_equal(False, check_peak_size([pam, pam1], (100, 100, 100),
                                            True))
    npt.assert_equal(False, check_peak_size([pam, pam1], (100, 100, 100),
                                            False))
