import json
import os
from copy import deepcopy

from nibabel.tmpdirs import InTemporaryDirectory
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_array_equal, assert_
import pytest

from dipy.data import fetch_gold_standard_io
from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram

from dipy.utils.optpkg import optional_package
fury, have_fury, setup_module = optional_package('fury')

filepath_dix = {}
files, folder = fetch_gold_standard_io()
for filename in files:
    filepath_dix[filename] = os.path.join(folder, filename)

with open(filepath_dix['points_data.json']) as json_file:
    points_data = dict(json.load(json_file))

with open(filepath_dix['streamlines_data.json']) as json_file:
    streamlines_data = dict(json.load(json_file))


# UNIT TESTS
def trk_equal_in_vox_space():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.VOX)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_vox_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def tck_equal_in_vox_space():
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'],
                          to_space=Space.VOX)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_vox_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def fib_equal_in_vox_space():
    if not have_fury:
        return
    sft = load_tractogram(filepath_dix['gs.fib'], filepath_dix['gs.nii'],
                          to_space=Space.VOX)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_vox_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def dpy_equal_in_vox_space():
    sft = load_tractogram(filepath_dix['gs.dpy'], filepath_dix['gs.nii'],
                          to_space=Space.VOX)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_vox_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def trk_equal_in_rasmm_space():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def tck_equal_in_rasmm_space():
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def fib_equal_in_rasmm_space():
    if not have_fury:
        return
    sft = load_tractogram(filepath_dix['gs.fib'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def dpy_equal_in_rasmm_space():
    sft = load_tractogram(filepath_dix['gs.dpy'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def trk_equal_in_voxmm_space():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.VOXMM)
    tmp_points_voxmm = np.loadtxt(filepath_dix['gs_voxmm_space.txt'])
    assert_allclose(tmp_points_voxmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def tck_equal_in_voxmm_space():
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'],
                          to_space=Space.VOXMM)
    tmp_points_voxmm = np.loadtxt(filepath_dix['gs_voxmm_space.txt'])
    assert_allclose(tmp_points_voxmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def fib_equal_in_voxmm_space():
    if not have_fury:
        return
    sft = load_tractogram(filepath_dix['gs.fib'], filepath_dix['gs.nii'],
                          to_space=Space.VOXMM)
    tmp_points_voxmm = np.loadtxt(filepath_dix['gs_voxmm_space.txt'])
    assert_allclose(tmp_points_voxmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def dpy_equal_in_voxmm_space():
    sft = load_tractogram(filepath_dix['gs.dpy'], filepath_dix['gs.nii'],
                          to_space=Space.VOXMM)
    tmp_points_voxmm = np.loadtxt(filepath_dix['gs_voxmm_space.txt'])
    assert_allclose(tmp_points_voxmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def switch_voxel_sizes_from_rasmm():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    sft_switch = StatefulTractogram(sft.streamlines,
                                    filepath_dix['gs_3mm.nii'],
                                    Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    tmp_points_voxmm = np.loadtxt(filepath_dix['gs_voxmm_space.txt'])

    sft_switch.to_rasmm()
    assert_allclose(tmp_points_rasmm,
                    sft_switch.streamlines.get_data(), atol=1e-3, rtol=1e-6)

    sft_switch.to_voxmm()
    assert_allclose(tmp_points_voxmm,
                    sft_switch.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def switch_voxel_sizes_from_voxmm():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.VOXMM)
    sft_switch = StatefulTractogram(sft.streamlines,
                                    filepath_dix['gs_3mm.nii'],
                                    Space.VOXMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    tmp_points_voxmm = np.loadtxt(filepath_dix['gs_voxmm_space.txt'])

    sft_switch.to_rasmm()
    assert_allclose(tmp_points_rasmm,
                    sft_switch.streamlines.get_data(), atol=1e-3, rtol=1e-6)

    sft_switch.to_voxmm()
    assert_allclose(tmp_points_voxmm,
                    sft_switch.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def to_rasmm_equivalence():
    sft_1 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOX)
    sft_2 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOX)

    sft_1.to_rasmm()
    sft_2.to_space(Space.RASMM)
    assert_allclose(sft_1.streamlines.get_data(),
                    sft_2.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def to_voxmm_equivalence():
    sft_1 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOX)
    sft_2 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOX)

    sft_1.to_voxmm()
    sft_2.to_space(Space.VOXMM)
    assert_allclose(sft_1.streamlines.get_data(),
                    sft_2.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def to_vox_equivalence():
    sft_1 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.RASMM)
    sft_2 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.RASMM)

    sft_1.to_vox()
    sft_2.to_space(Space.VOX)
    assert_allclose(sft_1.streamlines.get_data(),
                    sft_2.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def to_corner_equivalence():
    sft_1 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOX)
    sft_2 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOX)

    sft_1.to_corner()
    sft_2.to_origin(Origin.TRACKVIS)
    assert_allclose(sft_1.streamlines.get_data(),
                    sft_2.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def to_center_equivalence():
    sft_1 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOX)
    sft_2 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOX)

    sft_1.to_center()
    sft_2.to_origin(Origin.NIFTI)
    assert_allclose(sft_1.streamlines.get_data(),
                    sft_2.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def trk_iterative_saving_loading():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    with InTemporaryDirectory():
        save_tractogram(sft, 'gs_iter.trk')
        tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])

        for _ in range(100):
            sft_iter = load_tractogram('gs_iter.trk', filepath_dix['gs.nii'],
                                       to_space=Space.RASMM)
            assert_allclose(tmp_points_rasmm,
                            sft_iter.streamlines.get_data(),
                            atol=1e-3, rtol=1e-6)
            save_tractogram(sft_iter, 'gs_iter.trk')


def tck_iterative_saving_loading():
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    with InTemporaryDirectory():
        save_tractogram(sft, 'gs_iter.tck')
        tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])

        for _ in range(100):
            sft_iter = load_tractogram('gs_iter.tck', filepath_dix['gs.nii'],
                                       to_space=Space.RASMM)
            assert_allclose(tmp_points_rasmm,
                            sft_iter.streamlines.get_data(),
                            atol=1e-3, rtol=1e-6)
            save_tractogram(sft_iter, 'gs_iter.tck')


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def fib_iterative_saving_loading():
    if not have_fury:
        return
    sft = load_tractogram(filepath_dix['gs.fib'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    with InTemporaryDirectory():
        save_tractogram(sft, 'gs_iter.fib')
        tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])

        for _ in range(100):
            sft_iter = load_tractogram('gs_iter.fib', filepath_dix['gs.nii'],
                                       to_space=Space.RASMM)
            assert_allclose(tmp_points_rasmm,
                            sft_iter.streamlines.get_data(),
                            atol=1e-3, rtol=1e-6)
            save_tractogram(sft_iter, 'gs_iter.fib')


def dpy_iterative_saving_loading():
    sft = load_tractogram(filepath_dix['gs.dpy'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    with InTemporaryDirectory():
        save_tractogram(sft, 'gs_iter.dpy')
        tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])

        for _ in range(100):
            sft_iter = load_tractogram('gs_iter.dpy', filepath_dix['gs.nii'],
                                       to_space=Space.RASMM)
            assert_allclose(tmp_points_rasmm,
                            sft_iter.streamlines.get_data(),
                            atol=1e-3, rtol=1e-6)
            save_tractogram(sft_iter, 'gs_iter.dpy')


def iterative_to_vox_transformation():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    for _ in range(1000):
        sft.to_vox()
        sft.to_rasmm()
        assert_allclose(tmp_points_rasmm,
                        sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def iterative_to_voxmm_transformation():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(filepath_dix['gs_rasmm_space.txt'])
    for _ in range(1000):
        sft.to_voxmm()
        sft.to_rasmm()
        assert_allclose(tmp_points_rasmm,
                        sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def empty_space_change():
    sft = StatefulTractogram([], filepath_dix['gs.nii'], Space.VOX)
    sft.to_vox()
    sft.to_voxmm()
    sft.to_rasmm()
    assert_array_equal([], sft.streamlines.get_data())


def empty_shift_change():
    sft = StatefulTractogram([], filepath_dix['gs.nii'], Space.VOX)
    sft.to_corner()
    sft.to_center()
    assert_array_equal([], sft.streamlines.get_data())


def empty_remove_invalid():
    sft = StatefulTractogram([], filepath_dix['gs.nii'], Space.VOX)
    sft.remove_invalid_streamlines()
    assert_array_equal([], sft.streamlines.get_data())


def shift_corner_from_rasmm():
    sft_1 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOX)
    sft_1.to_corner()
    bbox_1 = sft_1.compute_bounding_box()

    sft_2 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.RASMM)
    sft_2.to_corner()
    sft_2.to_vox()
    bbox_2 = sft_2.compute_bounding_box()

    assert_allclose(bbox_1, bbox_2, atol=1e-3, rtol=1e-6)


def shift_corner_from_voxmm():
    sft_1 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOX)
    sft_1.to_corner()
    bbox_1 = sft_1.compute_bounding_box()

    sft_2 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                            to_space=Space.VOXMM)
    sft_2.to_corner()
    sft_2.to_vox()
    bbox_2 = sft_2.compute_bounding_box()

    assert_allclose(bbox_1, bbox_2, atol=1e-3, rtol=1e-6)


def iterative_shift_corner():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_streamlines = sft.get_streamlines_copy()

    for _ in range(1000):
        sft._shift_voxel_origin()

    assert_allclose(sft.get_streamlines_copy(),
                    tmp_streamlines, atol=1e-3, rtol=1e-6)


def replace_streamlines():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_streamlines = sft.get_streamlines_copy()[::-1]

    try:
        sft.streamlines = tmp_streamlines
        return True
    except (TypeError, ValueError):
        return False


def subsample_streamlines():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)
    tmp_streamlines = sft.get_streamlines_copy()[0:8]

    try:
        sft.streamlines = tmp_streamlines
        return False
    except (TypeError, ValueError):
        return True


def reassign_both_data_sep_to_empty():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)

    try:
        sft.data_per_point = {}
        sft.data_per_streamline = {}
    except (TypeError, ValueError):
        return False

    return sft.data_per_point == {} and \
        sft.data_per_streamline == {}


def reassign_both_data_sep():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_space=Space.RASMM)

    try:
        sft.data_per_point = points_data
        sft.data_per_streamline = streamlines_data
    except (TypeError, ValueError):
        return False

    return True


def bounding_bbox_valid(standard):
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'],
                          to_origin=standard, bbox_valid_check=False)

    return sft.is_bbox_in_vox_valid()


def random_point_color():
    np.random.seed(0)
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'])

    random_colors = np.random.randint(0, 255, (13, 8, 3))
    coloring_dict = {}
    coloring_dict['colors'] = random_colors

    try:
        sft.data_per_point = coloring_dict
        with InTemporaryDirectory():
            save_tractogram(sft, 'random_points_color.trk')
        return True
    except (TypeError, ValueError):
        return False


def random_point_gray():
    np.random.seed(0)
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'])

    random_colors = np.random.randint(0, 255, (13, 8, 1))
    coloring_dict = {}
    coloring_dict['color_x'] = random_colors
    coloring_dict['color_y'] = random_colors
    coloring_dict['color_z'] = random_colors

    try:
        sft.data_per_point = coloring_dict
        with InTemporaryDirectory():
            save_tractogram(sft, 'random_points_gray.trk')
        return True
    except ValueError:
        return False


def random_streamline_color():
    np.random.seed(0)
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'])

    uniform_colors_x = np.random.randint(0, 255, (13, 1))
    uniform_colors_y = np.random.randint(0, 255, (13, 1))
    uniform_colors_z = np.random.randint(0, 255, (13, 1))
    uniform_colors_x = np.expand_dims(
        np.repeat(uniform_colors_x, 8, axis=1), axis=-1)
    uniform_colors_y = np.expand_dims(
        np.repeat(uniform_colors_y, 8, axis=1), axis=-1)
    uniform_colors_z = np.expand_dims(
        np.repeat(uniform_colors_z, 8, axis=1), axis=-1)

    coloring_dict = {}
    coloring_dict['color_x'] = uniform_colors_x
    coloring_dict['color_y'] = uniform_colors_y
    coloring_dict['color_z'] = uniform_colors_z

    try:
        sft.data_per_point = coloring_dict
        with InTemporaryDirectory():
            save_tractogram(sft, 'random_streamlines_color.trk')
        return True
    except (TypeError, ValueError):
        return False


def out_of_grid(value):
    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'])
    sft.to_vox()
    tmp_streamlines = list(sft.get_streamlines_copy())
    tmp_streamlines[0] += value

    try:
        sft.streamlines = tmp_streamlines
        return sft.is_bbox_in_vox_valid()
    except (TypeError, ValueError):
        return True

    sft = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'])
    sft.to_vox()
    tmp_streamlines = list(sft.get_streamlines_copy())
    tmp_streamlines[0] += value

    try:
        sft.streamlines = tmp_streamlines
        return sft.is_bbox_in_vox_valid()
    except (TypeError, ValueError):
        return True


def data_per_point_consistency_addition():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    sft_first_half = sft[0:7]
    sft_last_half = sft[7:13]

    sft_first_half.data_per_point = {}
    try:
        _ = sft_first_half + sft_last_half
        return True
    except ValueError:
        return False


def data_per_streamline_consistency_addition():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    sft_first_half = sft[0:7]
    sft_last_half = sft[7:13]

    sft_first_half.data_per_streamline = {}
    try:
        _ = sft_first_half + sft_last_half
        return True
    except ValueError:
        return False


def space_consistency_addition():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    sft_first_half = sft[0:7]
    sft_last_half = sft[7:13]

    sft_first_half.to_vox()
    try:
        _ = sft_first_half + sft_last_half
        return True
    except ValueError:
        return False


def origin_consistency_addition():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    sft_first_half = sft[0:7]
    sft_last_half = sft[7:13]

    sft_first_half.to_corner()
    try:
        _ = sft_first_half + sft_last_half
        return True
    except ValueError:
        return False


def space_attributes_consistency_addition():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    sft_switch = StatefulTractogram(sft.streamlines,
                                    filepath_dix['gs_3mm.nii'],
                                    Space.RASMM)

    try:
        _ = sft + sft_switch
        return True
    except ValueError:
        return False


def test_equality():
    sft_1 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    sft_2 = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])

    assert_(sft_1 == sft_2,
            msg='Identical sft should be equal (==)')


def test_basic_slicing():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    first_streamline_sft = sft[0]

    npt.assert_allclose(first_streamline_sft.streamlines[0][0],
                        [11.149319, 21.579943, 37.600685],
                        err_msg='streamlines were not sliced correctly')
    rgb = np.array([first_streamline_sft.data_per_point['color_x'][0][0],
                    first_streamline_sft.data_per_point['color_y'][0][0],
                    first_streamline_sft.data_per_point['color_z'][0][0]])
    npt.assert_allclose(np.squeeze(rgb), [220., 20., 60.],
                        err_msg='data_per_point were not sliced correctly')
    rand_coord = first_streamline_sft.data_per_streamline['random_coord']
    npt.assert_allclose(np.squeeze(rand_coord), [7., 1., 5.],
                        err_msg='data_per_streamline were not sliced correctly')


def test_space_side_effect_slicing():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    first_streamline = deepcopy(sft.streamlines[0])

    first_streamline_sft = sft[0]
    sft.to_vox()
    npt.assert_allclose(first_streamline_sft.streamlines[0], first_streamline,
                        err_msg='Side effect, modifying a StatefulTractogram '
                                'after slicing should not modify the slice')
    # Testing it both ways
    sft.to_rasmm()
    first_streamline_sft.to_vox()
    npt.assert_allclose(sft.streamlines[0], first_streamline,
                        err_msg='Side effect, modifying a StatefulTractogram '
                                'after slicing should not modify the slice')


def test_origin_side_effect_slicing():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    first_streamline = deepcopy(sft.streamlines[0])

    first_streamline_sft = sft[0]
    sft.to_corner()
    npt.assert_allclose(first_streamline_sft.streamlines[0], first_streamline,
                        err_msg='Side effect, modifying a StatefulTractogram '
                                'after slicing should not modify the slice')
    # Testing it both ways
    sft.to_center()
    first_streamline_sft.to_corner()
    npt.assert_allclose(sft.streamlines[0], first_streamline,
                        err_msg='Side effect, modifying a StatefulTractogram '
                                'after slicing should not modify the slice')


def test_advanced_slicing():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    last_streamline_sft = sft[::-1][0]

    npt.assert_allclose(last_streamline_sft.streamlines[0][0],
                        [14.389803, 27.857153, 39.3602],
                        err_msg='streamlines were not sliced correctly')
    rgb = np.array([last_streamline_sft.data_per_point['color_x'][0][0],
                    last_streamline_sft.data_per_point['color_y'][0][0],
                    last_streamline_sft.data_per_point['color_z'][0][0]])
    npt.assert_allclose(np.squeeze(rgb), [0., 255., 0.],
                        err_msg='data_per_point were not sliced correctly')
    rand_coord = last_streamline_sft.data_per_streamline['random_coord']
    npt.assert_allclose(np.squeeze(rand_coord), [7., 9., 8.],
                        err_msg='data_per_streamline were not sliced correctly')


def test_basic_addition():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    sft_first_half = sft[0:7]
    sft_last_half = sft[7:13]

    concatenate_sft = sft_first_half + sft_last_half
    assert_(concatenate_sft == sft,
            msg='sft were not added correctly')


def test_space_side_effect_addition():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    sft_first_half = sft[0:7]
    sft_last_half = sft[7:13]

    concatenate_sft = sft_first_half + sft_last_half
    sft.to_vox()
    assert_(concatenate_sft != sft,
            msg='Side effect, modifying a StatefulTractogram '
            'after an addition should not modify the result')

    # Testing it both ways
    sft.to_rasmm()
    concatenate_sft.to_vox()
    assert_(concatenate_sft != sft,
            msg='Side effect, modifying a StatefulTractogram '
            'after an addition should not modify the result')


def test_origin_side_effect_addition():
    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    sft_first_half = sft[0:7]
    sft_last_half = sft[7:13]

    concatenate_sft = sft_first_half + sft_last_half
    sft.to_corner()
    assert_(concatenate_sft != sft,
            msg='Side effect, modifying a StatefulTractogram '
            'after an addition should not modify the result')

    # Testing it both ways
    sft.to_center()
    concatenate_sft.to_corner()
    assert_(concatenate_sft != sft,
            msg='Side effect, modifying a StatefulTractogram '
            'after an addition should not modify the result')


def test_addition_consistency():
    assert_(not space_attributes_consistency_addition(),
            msg='Adding sft with different space attributes should fail')
    assert_(not data_per_point_consistency_addition(),
            msg='Adding sft with different data_per_point keys should fail')
    assert_(not data_per_streamline_consistency_addition(),
            msg='Adding sft with different data_per_streamline keys should fail')
    assert_(not space_consistency_addition(),
            msg='Adding sft with different Space should fail')
    assert_(not origin_consistency_addition(),
            msg='Adding sft with different Origin should fail')


def test_iterative_transformation():
    iterative_to_vox_transformation()
    iterative_to_voxmm_transformation()


def test_iterative_saving_loading():
    trk_iterative_saving_loading()
    tck_iterative_saving_loading()
    fib_iterative_saving_loading()
    dpy_iterative_saving_loading()


def test_equal_in_vox_space():
    trk_equal_in_vox_space()
    tck_equal_in_vox_space()
    fib_equal_in_vox_space()
    dpy_equal_in_vox_space()


def test_equal_in_rasmm_space():
    trk_equal_in_rasmm_space()
    tck_equal_in_rasmm_space()
    fib_equal_in_rasmm_space()
    dpy_equal_in_rasmm_space()


def test_equal_in_voxmm_space():
    trk_equal_in_voxmm_space()
    tck_equal_in_voxmm_space()
    fib_equal_in_voxmm_space()
    dpy_equal_in_voxmm_space()


def test_switch_reference():
    switch_voxel_sizes_from_rasmm()
    switch_voxel_sizes_from_voxmm()


def test_to_space():
    to_rasmm_equivalence()
    to_voxmm_equivalence()
    to_vox_equivalence()


def test_to_origin():
    to_center_equivalence()
    to_corner_equivalence()


def test_empty_sft():
    empty_space_change()
    empty_shift_change()
    empty_remove_invalid()


def test_shifting_corner():
    shift_corner_from_rasmm()
    shift_corner_from_voxmm()
    iterative_shift_corner()


def test_replace_streamlines():
    # First two is expected to fail
    assert_(subsample_streamlines(),
            msg='Subsampling streamlines should not fail')
    assert_(replace_streamlines(),
            msg='Replacing streamlines should not fail')
    assert_(reassign_both_data_sep(),
            msg='Reassigning streamline/point data should not fail')
    assert_(reassign_both_data_sep_to_empty(),
            msg='Emptying streamline/point data should not fail')


def test_bounding_box():
    assert_(bounding_bbox_valid(Origin.NIFTI),
            msg='Bounding box should be valid with proper declaration')
    assert_(bounding_bbox_valid(Origin.TRACKVIS),
            msg='Bounding box should be valid with proper declaration')
    assert_(not out_of_grid(100),
            msg='Positive translation should make the bbox check fail')
    assert_(not out_of_grid(-100),
            msg='Negative translation should make the bbox check fail')


def test_invalid_streamlines():

    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    src_strml_count = len(sft)

    obtained_idx_to_remove, obtained_idx_to_keep = \
        sft.remove_invalid_streamlines()

    expected_idx_to_keep = list(range(src_strml_count))

    assert len(obtained_idx_to_remove) == 0
    assert expected_idx_to_keep == obtained_idx_to_keep
    assert_(
        len(sft) == src_strml_count,
        msg='An unshifted gold standard should have {} invalid streamlines'.
            format(src_strml_count - src_strml_count))

    # Change the dimensions so that a few streamlines become invalid
    sft.dimensions[2] = 5

    obtained_idx_to_remove, obtained_idx_to_keep = \
        sft.remove_invalid_streamlines()

    expected_idx_to_remove = [1, 3, 5, 7, 8, 9, 10, 11]
    expected_idx_to_keep = [0, 2, 4, 6, 12]
    expected_len_sft = 5

    assert obtained_idx_to_remove == expected_idx_to_remove
    assert obtained_idx_to_keep == expected_idx_to_keep
    assert_(
        len(sft) == expected_len_sft,
        msg='The shifted gold standard should have {} invalid streamlines'.
            format(src_strml_count - expected_len_sft))


def test_invalid_streamlines_epsilon():

    sft = load_tractogram(filepath_dix['gs.trk'], filepath_dix['gs.nii'])
    src_strml_count = len(sft)

    epsilon = 1e-6
    obtained_idx_to_remove, obtained_idx_to_keep = \
        sft.remove_invalid_streamlines(epsilon)

    expected_idx_to_keep = list(range(src_strml_count))

    assert len(obtained_idx_to_remove) == 0
    assert expected_idx_to_keep == obtained_idx_to_keep
    assert_(len(sft) == src_strml_count,
            msg='A small epsilon should not remove any streamlines')

    epsilon = 1.0
    obtained_idx_to_remove, obtained_idx_to_keep = \
        sft.remove_invalid_streamlines(epsilon)

    expected_idx_to_remove = [0, 1, 2, 3, 4, 5, 6, 7]
    expected_idx_to_keep = [8, 9, 10, 11, 12]
    expected_len_sft = 5

    expected_removed_strml_count = src_strml_count - expected_len_sft

    assert obtained_idx_to_remove == expected_idx_to_remove
    assert obtained_idx_to_keep == expected_idx_to_keep
    assert_(
        len(sft) == expected_len_sft,
        msg='Too big of an epsilon ({} mm) should have removed {} streamlines '
            '({} corners)'.format(
            epsilon,
            expected_removed_strml_count,
            expected_removed_strml_count)
    )


def test_trk_coloring():
    assert_(random_streamline_color(),
            msg='Streamlines color assignement failed')
    assert_(random_point_gray(),
            msg='Streamlines points gray assignement failed')
    assert_(random_point_color(),
            msg='Streamlines points color assignement failed')


def test_create_from_sft():
    sft_1 = load_tractogram(filepath_dix['gs.tck'], filepath_dix['gs.nii'])
    sft_2 = StatefulTractogram.from_sft(
        sft_1.streamlines, sft_1,
        data_per_point=sft_1.data_per_point,
        data_per_streamline=sft_1.data_per_streamline)

    if not (np.array_equal(sft_1.streamlines, sft_2.streamlines)
            and sft_1.space_attributes == sft_2.space_attributes
            and sft_1.space == sft_2.space
            and sft_1.origin == sft_2.origin
            and sft_1.data_per_point == sft_2.data_per_point
            and sft_1.data_per_streamline == sft_2.data_per_streamline):
        raise AssertionError()

    # Side effect testing
    sft_1.streamlines = np.arange(6000).reshape((100, 20, 3))
    if np.array_equal(sft_1.streamlines, sft_2.streamlines):
        raise AssertionError()
