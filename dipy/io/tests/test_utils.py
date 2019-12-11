import os

from dipy.data import fetch_gold_standard_io
from dipy.io.utils import (create_nifti_header,
                           decfa, decfa_to_float,
                           get_reference_info,
                           is_reference_info_valid)
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


def is_affine_valid(affine):
    return is_reference_info_valid(affine, [1, 1, 1], [1.0, 1.0, 1.0],
                                   'RAS')


def is_dimensions_valid(dimensions):
    return is_reference_info_valid(np.eye(4), dimensions, [1.0, 1.0, 1.0],
                                   'RAS')


def is_voxel_sizes_valid(voxel_sizes):
    return is_reference_info_valid(np.eye(4), [1, 1, 1], voxel_sizes,
                                   'RAS')


def is_voxel_order_valid(voxel_order):
    return is_reference_info_valid(np.eye(4), [1, 1, 1], [1.0, 1.0, 1.0],
                                   voxel_order)


def test_reference_info_validity():
    if is_affine_valid(np.eye(3)):
        raise AssertionError()
    if is_affine_valid(np.zeros((4, 4))):
        raise AssertionError()
    if not is_affine_valid(np.eye(4)):
        raise AssertionError()

    if is_dimensions_valid([0, 0]):
        raise AssertionError()
    if is_dimensions_valid([1, 1.0, 1]):
        raise AssertionError()
    if is_dimensions_valid([1, -1.0, 1]):
        raise AssertionError()
    if not is_dimensions_valid([1, 1, 1]):
        raise AssertionError()

    if is_voxel_sizes_valid([0, 0]):
        raise AssertionError()
    if is_voxel_sizes_valid([1, -1.0, 1]):
        raise AssertionError()
    if not is_voxel_sizes_valid([1.0, 1.0, 1.0]):
        raise AssertionError()

    if is_voxel_order_valid('RA'):
        raise AssertionError()
    if is_voxel_order_valid(['RAS']):
        raise AssertionError()
    if is_voxel_order_valid(['R', 'A', 'Z']):
        raise AssertionError()
    if is_voxel_order_valid('RAZ'):
        raise AssertionError()
    if not is_voxel_order_valid('RAS'):
        raise AssertionError()
    if not is_voxel_order_valid(['R', 'A', 'S']):
        raise AssertionError()


def reference_info_zero_affine():
    header = create_nifti_header(np.zeros((4, 4)), [10, 10, 10], [1, 1, 1])
    try:
        get_reference_info(header)
        return True
    except ValueError:
        return False


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
