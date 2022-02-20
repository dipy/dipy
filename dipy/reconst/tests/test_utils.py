import numpy as np
import numpy.testing as npt
from dipy.reconst.utils import _roi_in_volume, _mask_from_roi


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
