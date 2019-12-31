import os

from dipy.data import fetch_gold_standard_io
from dipy.io.utils import (create_nifti_header,
                           decfa, decfa_to_float,
                           get_reference_info,
                           is_reference_info_valid)
from nibabel import Nifti1Image
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_

filepath_dix = {}
files, folder = fetch_gold_standard_io()
for filename in files:
    filepath_dix[filename] = os.path.join(folder, filename)


def test_decfa():
    data_orig = np.zeros((4, 4, 4, 3))
    data_orig[0, 0, 0] = np.array([1, 0, 0])
    img_orig = Nifti1Image(data_orig, np.eye(4))
    img_new = decfa(img_orig)
    data_new = np.asanyarray(img_new.dataobj)
    assert data_new[0, 0, 0] == np.array((1, 0, 0),
                                         dtype=np.dtype([('R', 'uint8'),
                                                         ('G', 'uint8'),
                                                         ('B', 'uint8')]))
    assert data_new.dtype == np.dtype([('R', 'uint8'),
                                       ('G', 'uint8'),
                                       ('B', 'uint8')])

    round_trip = decfa_to_float(img_new)
    data_rt = np.asanyarray(round_trip.dataobj)
    assert np.all(data_rt == data_orig)

    data_orig = np.zeros((4, 4, 4, 3))
    data_orig[0, 0, 0] = np.array([0.1, 0, 0])
    img_orig = Nifti1Image(data_orig, np.eye(4))
    img_new = decfa(img_orig, scale=True)
    data_new = np.asanyarray(img_new.dataobj)
    assert data_new[0, 0, 0] == np.array((25, 0, 0),
                                         dtype=np.dtype([('R', 'uint8'),
                                                         ('G', 'uint8'),
                                                         ('B', 'uint8')]))
    assert data_new.dtype == np.dtype([('R', 'uint8'),
                                       ('G', 'uint8'),
                                       ('B', 'uint8')])

    round_trip = decfa_to_float(img_new)
    data_rt = np.asanyarray(round_trip.dataobj)
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
    assert_(not is_affine_valid(np.eye(3)),
            msg='3x3 affine is invalid')
    assert_(not is_affine_valid(np.zeros((4, 4))),
            msg='All zeroes affine is invalid')
    assert_(is_affine_valid(np.eye(4)),
            msg='Identity should be valid')

    assert_(not is_dimensions_valid([0, 0]),
            msg='Dimensions of the wrong length')
    assert_(not is_dimensions_valid([1, 1.0, 1]),
            msg='Dimensions cannot be float')
    assert_(not is_dimensions_valid([1, -1, 1]),
            msg='Dimensions cannot be negative')
    assert_(is_dimensions_valid([1, 1, 1]),
            msg='Dimensions of [1,1,1] should be valid')

    assert_(not is_voxel_sizes_valid([0, 0]),
            msg='Voxel sizes of the wrong length')
    assert_(not is_voxel_sizes_valid([1, -1, 1]),
            msg='Voxel sizes cannot be negative')
    assert_(is_voxel_sizes_valid([1.0, 1.0, 1.0]),
            msg='Voxel sizes of [1.0,1.0,1.0] should be valid')

    assert_(not is_voxel_order_valid('RA'),
            msg='Voxel order of the wrong length')
    assert_(not is_voxel_order_valid(['RAS']),
            msg='List of string is not a valid voxel order')
    assert_(not is_voxel_order_valid(['R', 'A', 'Z']),
            msg='Invalid value for voxel order (Z)')
    assert_(not is_voxel_order_valid('RAZ'),
            msg='Invalid value for voxel order (Z)')
    assert_(is_voxel_order_valid('RAS'),
            msg='RAS should be a valid voxel order')
    assert_(is_voxel_order_valid(['R', 'A', 'S']),
            msg='RAS should be a valid voxel order')


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
    assert_(not reference_info_zero_affine(),
            msg='An all zeros affine should not be valid')
