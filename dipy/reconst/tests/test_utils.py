import numpy as np
import numpy.testing as npt
from dipy.reconst.utils import _roi_in_volume, _mask_from_roi, convert_tensors


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
    converted_tensor = convert_tensors(tensor, 'dipy', 'mrtrix')
    expected_tensor = np.array([[[[1, 3, 6, 2, 4, 5]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 2: Convert from 'mrtrix' to 'ants'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, 'mrtrix', 'ants')
    expected_tensor = np.array([[[[[1, 4, 2, 5, 6, 3]]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 3: Convert from 'ants' to 'fsl'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, 'ants', 'fsl')
    expected_tensor = np.array([[[[1, 2, 4, 3, 5, 6]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 4: Convert from 'fsl' to 'dipy'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, 'fsl', 'dipy')
    expected_tensor = np.array([[[[1, 2, 4, 3, 5, 6]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 5: Convert from 'dipy' to 'ants'
    tensor = np.array([[[[1, 2, 3, 4, 5, 6]]]])
    converted_tensor = convert_tensors(tensor, 'dipy', 'ants')
    expected_tensor = np.array([[[[[1, 2, 3, 4, 5, 6]]]]])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 6: Convert from 'ants' to 'dipy'
    tensor = np.array([[[[[1, 2, 3, 4, 5, 6]]]]])
    converted_tensor = convert_tensors(tensor, 'ants', 'dipy')
    expected_tensor = np.array([1, 2, 3, 4, 5, 6])
    npt.assert_array_equal(converted_tensor, expected_tensor)

    # Test case 7: Convert from 'dipy' to 'dipy'
    tensor = np.array([[[[[1, 2, 3, 4, 5, 6]]]]])
    converted_tensor = convert_tensors(tensor, 'dipy', 'dipy')
    npt.assert_array_equal(converted_tensor, tensor)

    npt.assert_raises(ValueError, convert_tensors, tensor, 'amico', 'dipy')
    npt.assert_raises(ValueError, convert_tensors, tensor, 'dipy', 'amico')
