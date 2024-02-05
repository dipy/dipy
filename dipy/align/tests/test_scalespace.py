import numpy as np
import scipy as sp
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
from dipy.align import floating
from dipy.align.imwarp import get_direction_and_spacings
from dipy.align.scalespace import ScaleSpace, IsotropicScaleSpace
from dipy.align.tests.test_imwarp import get_synthetic_warped_circle
from dipy.testing.decorators import set_random_number_generator


def test_scale_space():
    num_levels = 3
    for test_class in [ScaleSpace, IsotropicScaleSpace]:
        for dim in [2, 3]:
            print(dim, test_class)
            if dim == 2:
                moving, static = get_synthetic_warped_circle(1)
            else:
                moving, static = get_synthetic_warped_circle(30)
            input_spacing = np.array([1.1, 1.2, 1.5])[:dim]
            grid2world = np.diag(tuple(input_spacing) + (1.0,))

            original = moving
            if test_class is ScaleSpace:
                ss = test_class(
                    original,
                    num_levels,
                    grid2world,
                    input_spacing)
            elif test_class is IsotropicScaleSpace:
                factors = [4, 2, 1]
                sigmas = [3.0, 1.0, 0.0]
                ss = test_class(
                    original,
                    factors,
                    sigmas,
                    grid2world,
                    input_spacing)
            for level in range(num_levels):
                # Verify sigmas and images are consistent
                sigmas = ss.get_sigmas(level)
                expected = sp.ndimage.gaussian_filter(original, sigmas)
                expected = ((expected - expected.min()) /
                            (expected.max() - expected.min()))
                actual = ss.get_image(level)
                assert_array_almost_equal(actual, expected)

                # Verify scalings and spacings are consistent
                spacings = ss.get_spacing(level)
                scalings = ss.get_scaling(level)
                expected = ss.get_spacing(0) * scalings
                actual = ss.get_spacing(level)
                assert_array_almost_equal(actual, expected)

                # Verify affine and affine_inv are consistent
                affine = ss.get_affine(level)
                affine_inv = ss.get_affine_inv(level)
                expected = np.eye(1 + dim)
                actual = affine.dot(affine_inv)
                assert_array_almost_equal(actual, expected)

                # Verify affine consistent with spacings
                exp_dir, expected_sp = get_direction_and_spacings(affine, dim)
                actual_sp = spacings
                assert_array_almost_equal(actual_sp, expected_sp)


@set_random_number_generator(2022966)
def test_scale_space_exceptions(rng):
    target_shape = (32, 32)
    # create a random image
    image = np.ndarray(target_shape, dtype=floating)
    ns = np.size(image)
    image[...] = rng.integers(0, 10, ns).reshape(tuple(target_shape))
    zeros = (image == 0).astype(np.int32)

    ss = ScaleSpace(image, 3)
    for invalid_level in [-1, 3, 4]:
        assert_raises(ValueError, ss.get_image, invalid_level)

    # Verify that the mask is correctly applied, when requested
    ss = ScaleSpace(image, 3, mask0=True)
    for level in range(3):
        img = ss.get_image(level)
        z = (img == 0).astype(np.int32)
        assert_array_equal(zeros, z)
