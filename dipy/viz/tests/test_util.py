import numpy as np
import numpy.testing as npt

from dipy.direction.peaks import PeaksAndMetrics
from dipy.testing.decorators import set_random_number_generator
from dipy.viz.horizon.util import (
    check_img_dtype,
    check_img_shapes,
    check_peak_size,
    show_ellipsis,
    unpack_data,
    unpack_surface,
)


@set_random_number_generator()
def test_check_img_shapes(rng):
    affine = np.array(
        [
            [1.0, 0.0, 0.0, -98.0],
            [0.0, 1.0, 0.0, -134.0],
            [0.0, 0.0, 1.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    data = 255 * rng.random((197, 233, 189))
    data1 = 255 * rng.random((197, 233, 189))
    images = [(data, affine), (data1, affine)]
    npt.assert_equal(check_img_shapes(images), (True, False))

    data1 = 255 * rng.random((200, 233, 189))
    images = [(data, affine), (data1, affine)]
    npt.assert_equal(check_img_shapes(images), (False, False))

    data = 255 * rng.random((197, 233, 189, 10))
    data1 = 255 * rng.random((197, 233, 189))
    images = [(data, affine), (data1, affine)]

    npt.assert_equal(check_img_shapes(images), (True, True))

    data = 255 * rng.random((197, 233, 189, 15))
    data1 = 255 * rng.random((197, 233, 189, 15))
    images = [(data, affine), (data1, affine)]

    npt.assert_equal(check_img_shapes(images), (True, True))

    data = 255 * rng.random((198, 233, 189, 14))
    data1 = 255 * rng.random((198, 233, 189, 15))
    images = [(data, affine), (data1, affine)]

    npt.assert_equal(check_img_shapes(images), (True, False))

    data = 255 * rng.random((197, 233, 189, 15))
    data1 = 255 * rng.random((198, 233, 189, 14))
    images = [(data, affine), (data1, affine)]

    npt.assert_equal(check_img_shapes(images), (False, False))

    data = 255 * rng.random((197, 233, 189, 15))
    data1 = 255 * rng.random((198, 233, 189, 15))
    images = [(data, affine), (data1, affine)]

    npt.assert_equal(check_img_shapes(images), (False, False))


@set_random_number_generator()
def test_check_img_dtype(rng):
    affine = np.array(
        [
            [1.0, 0.0, 0.0, -98.0],
            [0.0, 1.0, 0.0, -134.0],
            [0.0, 0.0, 1.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    data = 255 * rng.random((197, 233, 189))
    images = [
        (data, affine),
    ]

    npt.assert_equal(check_img_dtype(images)[0], images[0])

    data = rng.random((5, 5, 5)).astype(np.int64)
    images = [
        (data, affine),
    ]
    npt.assert_equal(check_img_dtype(images)[0][0].dtype, np.int32)
    assert len(check_img_dtype(images)) == 1

    data = rng.random((5, 5, 5)).astype(np.float16)
    images = [
        (data, affine),
    ]
    npt.assert_equal(check_img_dtype(images)[0][0].dtype, np.float32)
    assert len(check_img_dtype(images)) == 1

    data = rng.random((5, 5, 5)).astype(np.bool_)
    images = [
        (data, affine),
    ]
    assert len(check_img_dtype(images)) == 0


def test_show_ellipsis():
    text = "IAmALongFileName"
    text_size = 10
    available_size = 5
    result_text = f"...{text[-5:]}"

    npt.assert_equal(show_ellipsis(text, text_size, available_size), result_text)

    available_size = 12
    npt.assert_equal(show_ellipsis(text, text_size, available_size), text)


@set_random_number_generator()
def test_unpack_surface(rng):
    vertices = rng.random((100, 4))
    faces = rng.integers(0, 100, size=(100, 3))

    with npt.assert_raises(ValueError):
        unpack_surface((vertices, faces))

    vertices = rng.random((100, 3))
    faces = rng.integers(0, 100, size=(100, 4))

    with npt.assert_raises(ValueError):
        unpack_surface((vertices, faces, "/test/filename.pial"))

    vertices = rng.random((100, 3))
    faces = rng.integers(0, 100, size=(100, 3))
    v, f, fname = unpack_surface((vertices, faces, "/test/filename.pial"))
    npt.assert_equal(vertices, v)
    npt.assert_equal(faces, f)
    npt.assert_equal("/test/filename.pial", fname)


@set_random_number_generator()
def test_check_peak_size(rng):
    peak_dirs = rng.random((100, 100, 100, 10, 6))

    pam = PeaksAndMetrics()
    pam.peak_dirs = peak_dirs

    npt.assert_equal(True, check_peak_size([(pam, None)]))
    npt.assert_equal(True, check_peak_size([(pam, None), (pam, None)]))
    npt.assert_equal(
        False,
        check_peak_size([(pam, None)], ref_img_shape=(100, 100, 1), sync_imgs=True),
    )
    npt.assert_equal(
        False,
        check_peak_size([(pam, None)], ref_img_shape=(100, 100, 100), sync_imgs=False),
    )

    pam1 = PeaksAndMetrics()
    peak_dirs_1 = rng.random((100, 100, 50, 10, 6))
    pam1.peak_dirs = peak_dirs_1

    npt.assert_equal(False, check_peak_size([(pam, None), (pam1, None)]))
    npt.assert_equal(
        False,
        check_peak_size(
            [(pam, None), (pam1, None)], ref_img_shape=(100, 100, 100), sync_imgs=True
        ),
    )
    npt.assert_equal(
        False,
        check_peak_size(
            [(pam, None), (pam1, None)], ref_img_shape=(100, 100, 100), sync_imgs=False
        ),
    )


@set_random_number_generator()
def test_unpack_data(rng):
    # Test with a non-tuple object (single data array)
    pam_data = rng.random((10, 10))
    result = unpack_data(pam_data, return_size=2)
    npt.assert_equal(len(result), 2)
    npt.assert_equal(result[0], pam_data)
    npt.assert_equal(result[1], None)

    # Test with a tuple of length 1
    pam_tuple_1 = (pam_data,)
    result = unpack_data(pam_tuple_1, return_size=3)
    npt.assert_equal(len(result), 3)
    npt.assert_equal(result[0], pam_data)
    npt.assert_equal(result[1], None)
    npt.assert_equal(result[2], None)

    # Test with a complete tuple of length 2
    filename = "test_file.pam"
    pam_tuple_2 = (pam_data, filename)
    result = unpack_data(pam_tuple_2, return_size=2)
    npt.assert_equal(len(result), 2)
    npt.assert_equal(result[0], pam_data)
    npt.assert_equal(result[1], filename)

    # Test with a tuple of length greater than 2
    pam_tuple_3 = (pam_data, filename, "extra_item")
    result = unpack_data(pam_tuple_3, return_size=3)
    npt.assert_equal(result, pam_tuple_3)
    npt.assert_equal(len(result), 3)
