import tempfile
from urllib.error import URLError, HTTPError

import pytest

import nibabel as nib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_
import trx.trx_file_memmap as tmm

from dipy.data import get_fnames
from dipy.io.streamline import load_tractogram
from dipy.io.utils import (create_nifti_header,
                           decfa, decfa_to_float,
                           get_reference_info,
                           is_reference_info_valid,
                           read_img_arr_or_path)
from dipy.testing.decorators import set_random_number_generator

FILEPATH_DIX = None


def setup_module():
    global FILEPATH_DIX
    try:
        FILEPATH_DIX, _, _ = get_fnames('gold_standard_tracks')
    except (HTTPError, URLError) as e:
        FILEPATH_DIX = None
        error_msg = f'"Tests Data failed to download." Reason: {e}'
        pytest.skip(error_msg, allow_module_level=True)
        return


def teardown_module():
    global FILEPATH_DIX
    FILEPATH_DIX = None,


def test_decfa():
    data_orig = np.zeros((4, 4, 4, 3))
    data_orig[0, 0, 0] = np.array([1, 0, 0])
    img_orig = nib.Nifti1Image(data_orig, np.eye(4))
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
    img_orig = nib.Nifti1Image(data_orig, np.eye(4))
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


def test_reference_trk_file_info_identical():
    tuple_1 = get_reference_info(FILEPATH_DIX['gs.trk'])
    tuple_2 = get_reference_info(FILEPATH_DIX['gs.nii'])
    affine_1, dimensions_1, voxel_sizes_1, voxel_order_1 = tuple_1
    affine_2, dimensions_2, voxel_sizes_2, voxel_order_2 = tuple_2

    assert_allclose(affine_1, affine_2)
    assert_array_equal(dimensions_1, dimensions_2)
    assert_allclose(voxel_sizes_1, voxel_sizes_2)
    assert voxel_order_1 == voxel_order_2


def test_reference_trx_file_info_identical():
    tuple_1 = get_reference_info(FILEPATH_DIX['gs.trx'])
    tuple_2 = get_reference_info(FILEPATH_DIX['gs.nii'])
    affine_1, dimensions_1, voxel_sizes_1, voxel_order_1 = tuple_1
    affine_2, dimensions_2, voxel_sizes_2, voxel_order_2 = tuple_2

    assert_allclose(affine_1, affine_2)
    assert_array_equal(dimensions_1, dimensions_2)
    assert_allclose(voxel_sizes_1, voxel_sizes_2)
    assert voxel_order_1 == voxel_order_2


def test_reference_obj_info_identical():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], 'same')
    trx = tmm.load(FILEPATH_DIX['gs.trx'])
    img = nib.load(FILEPATH_DIX['gs.nii'])

    tuple_1 = get_reference_info(sft)
    tuple_2 = get_reference_info(trx)
    tuple_3 = get_reference_info(img)
    affine_1, dimensions_1, voxel_sizes_1, voxel_order_1 = tuple_1
    affine_2, dimensions_2, voxel_sizes_2, voxel_order_2 = tuple_2
    affine_3, dimensions_3, voxel_sizes_3, voxel_order_3 = tuple_3

    assert_allclose(affine_1, affine_2)
    assert_array_equal(dimensions_1, dimensions_2)
    assert_allclose(voxel_sizes_1, voxel_sizes_2)
    assert voxel_order_1 == voxel_order_2

    assert_allclose(affine_1, affine_3)
    assert_array_equal(dimensions_1, dimensions_3)
    assert_allclose(voxel_sizes_1, voxel_sizes_3)
    assert voxel_order_1 == voxel_order_3


def test_reference_header_info_identical():
    trk = nib.streamlines.load(FILEPATH_DIX['gs.trk'])
    trx = tmm.load(FILEPATH_DIX['gs.trx'])
    img = nib.load(FILEPATH_DIX['gs.nii'])

    tuple_1 = get_reference_info(trk.header)
    tuple_2 = get_reference_info(trx.header)
    tuple_3 = get_reference_info(img.header)
    affine_1, dimensions_1, voxel_sizes_1, voxel_order_1 = tuple_1
    affine_2, dimensions_2, voxel_sizes_2, voxel_order_2 = tuple_2
    affine_3, dimensions_3, voxel_sizes_3, voxel_order_3 = tuple_3

    assert_allclose(affine_1, affine_2)
    assert_array_equal(dimensions_1, dimensions_2)
    assert_allclose(voxel_sizes_1, voxel_sizes_2)
    assert voxel_order_1 == voxel_order_2

    assert_allclose(affine_1, affine_3)
    assert_array_equal(dimensions_1, dimensions_3)
    assert_allclose(voxel_sizes_1, voxel_sizes_3)
    assert voxel_order_1 == voxel_order_3


def test_all_zeros_affine():
    assert_(not reference_info_zero_affine(),
            msg='An all zeros affine should not be valid')


@set_random_number_generator()
def test_read_img_arr_or_path(rng):
    data = rng.random((4, 4, 4, 3))
    aff = np.eye(4)
    aff[:3, :] = rng.standard_normal((3, 4))
    img = nib.Nifti1Image(data, aff)
    path = tempfile.NamedTemporaryFile().name + '.nii.gz'
    nib.save(img, path)
    for this in [data, img, path]:
        dd, aa = read_img_arr_or_path(this, affine=aff)
        assert np.allclose(dd, data)
        assert np.allclose(aa, aff)

    # Tests that if an array is provided, but no affine, an error is raised:
    with pytest.raises(ValueError):
        read_img_arr_or_path(data)

    # Tests that the affine is recovered correctly from path:
    dd, aa = read_img_arr_or_path(path)
    assert np.allclose(dd, data)
    assert np.allclose(aa, aff)
