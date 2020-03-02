import numpy as np
import numpy.testing as npt
from dipy.reconst.utils import _roi_in_volume


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
    roi_radii = np.array([5, 10, 0])
    roi_radii_out = _roi_in_volume(data_shape, roi_center, roi_radii)
    npt.assert_array_equal(roi_radii_out, np.array([2, 5, 0]))



if __name__ == "__main__":
    npt.run_module_suite()