import numpy as np
import numpy.testing as npt

from dipy.core.geometry import euler_matrix
from dipy.reconst.dti import from_lower_triangular
from dipy.reconst.utils import _mask_from_roi, _roi_in_volume, convert_tensors
from dipy.testing.decorators import set_random_number_generator


def test_roi_in_volume():
    data_shape = (11, 11, 11, 64)
    roi_center = np.array([5, 5, 5])
    roi_radii = np.array([5, 5, 5])
    roi_radii_out = _roi_in_volume(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_radii_out, np.array([5, 5, 5]))

    roi_radii = np.array([6, 6, 6])
    roi_radii_out = _roi_in_volume(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_radii_out, np.array([5, 5, 5]))

    roi_center = np.array([4, 4, 4])
    roi_radii = np.array([5, 5, 5])
    roi_radii_out = _roi_in_volume(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_radii_out, np.array([4, 4, 4]))

    data_shape = (11, 11, 1, 64)
    roi_center = np.array([5, 5, 0])
    roi_radii = np.array([5, 5, 0])
    roi_radii_out = _roi_in_volume(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_radii_out, np.array([5, 5, 0]))

    roi_center = np.array([2, 5, 0])
    roi_radii = np.array([5, 10, 2])
    roi_radii_out = _roi_in_volume(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_radii_out, np.array([2, 5, 0]))


def test_mask_from_roi():
    data_shape = (5, 5, 5)
    roi_center = (2, 2, 2)
    roi_radii = (2, 2, 2)
    mask_gt = np.ones(data_shape)
    roi_mask = _mask_from_roi(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_mask, mask_gt)

    roi_radii = (1, 2, 2)
    mask_gt = np.zeros(data_shape)
    mask_gt[1:4, 0:5, 0:5] = 1
    roi_mask = _mask_from_roi(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_mask, mask_gt)

    roi_radii = (0, 2, 2)
    mask_gt = np.zeros(data_shape)
    mask_gt[2, 0:5, 0:5] = 1
    roi_mask = _mask_from_roi(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_mask, mask_gt)


def test_convert_tensor():
    # Test case 1: Convert from 'dipy' to 'mrtrix'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, "dipy", "mrtrix")
    expected_tensor = np.array([[[[1, 3, 6, 2, 4, 5]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 2: Convert from 'mrtrix' to 'ants'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, "mrtrix", "ants")
    expected_tensor = np.array([[[[[1, 4, 2, 5, 6, 3]]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 3: Convert from 'ants' to 'fsl'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, "ants", "fsl")
    expected_tensor = np.array([[[[1, 2, 4, 3, 5, 6]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 4: Convert from 'fsl' to 'dipy'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, "fsl", "dipy")
    expected_tensor = np.array([[[[1, 2, 4, 3, 5, 6]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 5: Convert from 'dipy' to 'ants'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, "dipy", "ants")
    expected_tensor = np.array([[[[[1, 2, 3, 4, 5, 6]]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 6: Convert from 'ants' to 'dipy'
    tensor = np.array([[[[[1, 2, 3, 4, 5, 6]]]]])
    converted_tensor = convert_tensors(tensor, "ants", "dipy")
    expected_tensor = np.array([1, 2, 3, 4, 5, 6])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 7: Convert from 'dipy' to 'dipy'
    tensor = np.array([[[[[1, 2, 3, 4, 5, 6]]]]])
    converted_tensor = convert_tensors(tensor, "dipy", "dipy")
    npt.assert_array_equal(converted_tensor, tensor)

    npt.assert_raises(ValueError, convert_tensors, tensor, "amico", "dipy")
    npt.assert_raises(ValueError, convert_tensors, tensor, "dipy", "amico")


@set_random_number_generator(1234)
def test_convert_tensor_affine(rng=None):
    rot = euler_matrix(*rng.uniform(-np.pi, np.pi, 3))
    tensor = rng.standard_normal((2, 3, 4, 6))
    v = rng.standard_normal((10, 3))  # directions in the voxel frame
    quad = "ni,...ij,nj->...n"
    expected = np.einsum(quad, v, from_lower_triangular(tensor), v)
    for sign in (-1.0, 1.0):  # oblique, stored in LAS (det < 0) or RAS order
        affine = rot @ np.diag([1.5 * sign, 1.5, 3.0, 1.0])
        to_world = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
        # The same directions, in the frame of each format. The diffusivity
        # along them must not depend on the format.
        frames = {"mrtrix": v @ to_world.T, "fsl": v * [-sign, 1, 1], "ants": v}
        out = {}
        for fmt, u in frames.items():
            out[fmt] = convert_tensors(tensor, "dipy", fmt, affine=affine)
            d_fmt = from_lower_triangular(convert_tensors(out[fmt], fmt, "dipy"))
            npt.assert_array_almost_equal(np.einsum(quad, u, d_fmt, u), expected)
            back = convert_tensors(out[fmt], fmt, "dipy", affine=affine)
            npt.assert_array_almost_equal(back, tensor)
        npt.assert_array_almost_equal(
            convert_tensors(out["fsl"], "fsl", "mrtrix", affine=affine), out["mrtrix"]
        )
