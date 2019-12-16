import os

from dipy.data import fetch_gold_standard_io
from dipy.io.utils import (decfa, decfa_to_float,
                           get_reference_info,
                           create_nifti_header)
from nibabel import Nifti1Image
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

filepath_dix = {}
files, folder = fetch_gold_standard_io()
for filename in files:
    filepath_dix[filename] = os.path.join(folder, filename)


def test_decfa():
    data_orig = np.zeros((4, 4, 4, 3))
    data_orig[0, 0, 0] = np.array([1, 0, 0])
    img_orig = Nifti1Image(data_orig, np.eye(4))
    img_new = decfa(img_orig)
    data_new = img_new.get_data()
    assert data_new[0, 0, 0] == np.array((1, 0, 0),
                                         dtype=np.dtype([('R', 'uint8'),
                                                         ('G', 'uint8'),
                                                         ('B', 'uint8')]))
    assert data_new.dtype == np.dtype([('R', 'uint8'),
                                       ('G', 'uint8'),
                                       ('B', 'uint8')])

    round_trip = decfa_to_float(img_new)
    data_rt = round_trip.get_fdata()
    assert np.all(data_rt == data_orig)

    data_orig = np.zeros((4, 4, 4, 3))
    data_orig[0, 0, 0] = np.array([0.1, 0, 0])
    img_orig = Nifti1Image(data_orig, np.eye(4))
    img_new = decfa(img_orig, scale=True)
    data_new = img_new.get_data()
    assert data_new[0, 0, 0] == np.array((25, 0, 0),
                                         dtype=np.dtype([('R', 'uint8'),
                                                         ('G', 'uint8'),
                                                         ('B', 'uint8')]))
    assert data_new.dtype == np.dtype([('R', 'uint8'),
                                       ('G', 'uint8'),
                                       ('B', 'uint8')])

    round_trip = decfa_to_float(img_new)
    data_rt = round_trip.get_data()
    assert data_rt.shape == (4, 4, 4, 3)
    assert np.all(data_rt[0, 0, 0] == np.array([25, 0, 0]))


def test_reference_info_identical():
    tuple_1 = get_reference_info(filepath_dix['gs.trk'])
    tuple_2 = get_reference_info(filepath_dix['gs.nii'])
    affine_1, dimensions_1, voxel_sizes_1, voxel_order_1 = tuple_1
    affine_2, dimensions_2, voxel_sizes_2, voxel_order_2 = tuple_2

    assert_allclose(affine_1, affine_2)
    assert_array_equal(dimensions_1, dimensions_2)
    assert_allclose(voxel_sizes_1, voxel_sizes_2)
    assert voxel_order_1 == voxel_order_2


def test_all_zeros_affine():
    if reference_info_zero_affine():
        raise AssertionError()


def reference_info_zero_affine():
    header = create_nifti_header(np.zeros((4, 4)), [10, 10, 10], [1, 1, 1])
    try:
        get_reference_info(header)
        return True
    except ValueError:
        return False
