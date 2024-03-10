import os
from os.path import join as pjoin
from copy import deepcopy
from tempfile import TemporaryDirectory
from urllib.error import URLError, HTTPError

import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_array_equal, assert_
import pytest

from dipy.data import get_fnames
from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.utils import is_header_compatible
from dipy.testing.decorators import set_random_number_generator

import trx.trx_file_memmap as tmm

from dipy.utils.optpkg import optional_package
fury, have_fury, setup_module = optional_package('fury', min_version="0.10.0")


FILEPATH_DIX, POINTS_DATA, STREAMLINES_DATA = None, None, None


def setup_module():
    global FILEPATH_DIX, POINTS_DATA, STREAMLINES_DATA
    try:
        FILEPATH_DIX, POINTS_DATA, STREAMLINES_DATA = get_fnames(
            'gold_standard_tracks')
    except (HTTPError, URLError) as e:
        FILEPATH_DIX, POINTS_DATA, STREAMLINES_DATA = None, None, None
        error_msg = f'"Tests Data failed to download." Reason: {e}'
        pytest.skip(error_msg, allow_module_level=True)
        return


def teardown_module():
    global FILEPATH_DIX, POINTS_DATA, STREAMLINES_DATA
    FILEPATH_DIX, POINTS_DATA, STREAMLINES_DATA = None, None, None


def test_direct_trx_loading():
    trx = tmm.load(FILEPATH_DIX['gs.trx'])
    tmp_dir = deepcopy(trx._uncompressed_folder_handle.name)
    assert os.path.isdir(tmp_dir)
    sft = trx.to_sft()

    tmp_points_vox = np.loadtxt(FILEPATH_DIX['gs_vox_space.txt'])
    tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])

    trx.close()
    assert not os.path.isdir(tmp_dir)

    assert_allclose(sft.streamlines._data, tmp_points_rasmm,
                    rtol=1e-04, atol=1e-06)
    sft.to_vox()
    assert_allclose(sft.streamlines._data, tmp_points_vox,
                    rtol=1e-04, atol=1e-06)


def test_trk_equal_in_vox_space():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.VOX)
    tmp_points_vox = np.loadtxt(FILEPATH_DIX['gs_vox_space.txt'])
    assert_allclose(tmp_points_vox,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_tck_equal_in_vox_space():
    sft = load_tractogram(FILEPATH_DIX['gs.tck'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.VOX)
    tmp_points_vox = np.loadtxt(FILEPATH_DIX['gs_vox_space.txt'])
    assert_allclose(tmp_points_vox,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_trx_equal_in_vox_space():
    sft = load_tractogram(FILEPATH_DIX['gs.trx'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.VOX)
    tmp_points_vox = np.loadtxt(FILEPATH_DIX['gs_vox_space.txt'])
    assert_allclose(tmp_points_vox,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_fib_equal_in_vox_space():
    if not have_fury:
        return
    sft = load_tractogram(FILEPATH_DIX['gs.fib'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.VOX)
    tmp_points_vox = np.loadtxt(FILEPATH_DIX['gs_vox_space.txt'])
    assert_allclose(tmp_points_vox,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_dpy_equal_in_vox_space():
    sft = load_tractogram(FILEPATH_DIX['gs.dpy'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.VOX)
    tmp_points_vox = np.loadtxt(FILEPATH_DIX['gs_vox_space.txt'])
    assert_allclose(tmp_points_vox,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_trk_equal_in_rasmm_space():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_tck_equal_in_rasmm_space():
    sft = load_tractogram(FILEPATH_DIX['gs.tck'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_trx_equal_in_rasmm_space():
    sft = load_tractogram(FILEPATH_DIX['gs.trx'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_fib_equal_in_rasmm_space():
    if not have_fury:
        return
    sft = load_tractogram(FILEPATH_DIX['gs.fib'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_dpy_equal_in_rasmm_space():
    sft = load_tractogram(FILEPATH_DIX['gs.dpy'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])
    assert_allclose(tmp_points_rasmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_trk_equal_in_voxmm_space():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.VOXMM)
    tmp_points_voxmm = np.loadtxt(FILEPATH_DIX['gs_voxmm_space.txt'])
    assert_allclose(tmp_points_voxmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_tck_equal_in_voxmm_space():
    sft = load_tractogram(FILEPATH_DIX['gs.tck'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.VOXMM)
    tmp_points_voxmm = np.loadtxt(FILEPATH_DIX['gs_voxmm_space.txt'])
    assert_allclose(tmp_points_voxmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_trx_equal_in_voxmm_space():
    sft = load_tractogram(FILEPATH_DIX['gs.trx'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.VOXMM)
    tmp_points_voxmm = np.loadtxt(FILEPATH_DIX['gs_voxmm_space.txt'])
    assert_allclose(tmp_points_voxmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_fib_equal_in_voxmm_space():
    if not have_fury:
        return
    sft = load_tractogram(FILEPATH_DIX['gs.fib'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.VOXMM)
    tmp_points_voxmm = np.loadtxt(FILEPATH_DIX['gs_voxmm_space.txt'])
    assert_allclose(tmp_points_voxmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_dpy_equal_in_voxmm_space():
    sft = load_tractogram(FILEPATH_DIX['gs.dpy'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.VOXMM)
    tmp_points_voxmm = np.loadtxt(FILEPATH_DIX['gs_voxmm_space.txt'])
    assert_allclose(tmp_points_voxmm,
                    sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_switch_voxel_sizes_from_rasmm():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    sft_switch = StatefulTractogram(sft.streamlines,
                                    FILEPATH_DIX['gs_3mm.nii'],
                                    Space.RASMM)
    tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])
    tmp_points_voxmm = np.loadtxt(FILEPATH_DIX['gs_voxmm_space.txt'])

    sft_switch.to_rasmm()
    assert_allclose(tmp_points_rasmm,
                    sft_switch.streamlines.get_data(), atol=1e-3, rtol=1e-6)

    sft_switch.to_voxmm()
    assert_allclose(tmp_points_voxmm,
                    sft_switch.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_switch_voxel_sizes_from_voxmm():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.VOXMM)
    sft_switch = StatefulTractogram(sft.streamlines,
                                    FILEPATH_DIX['gs_3mm.nii'],
                                    Space.VOXMM)
    tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])
    tmp_points_voxmm = np.loadtxt(FILEPATH_DIX['gs_voxmm_space.txt'])

    sft_switch.to_rasmm()
    assert_allclose(tmp_points_rasmm,
                    sft_switch.streamlines.get_data(), atol=1e-3, rtol=1e-6)

    sft_switch.to_voxmm()
    assert_allclose(tmp_points_voxmm,
                    sft_switch.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_to_rasmm_equivalence():
    sft_1 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.VOX)
    sft_2 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.VOX)

    sft_1.to_rasmm()
    sft_2.to_space(Space.RASMM)
    assert_allclose(sft_1.streamlines.get_data(),
                    sft_2.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_to_voxmm_equivalence():
    sft_1 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.VOX)
    sft_2 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.VOX)

    sft_1.to_voxmm()
    sft_2.to_space(Space.VOXMM)
    assert_allclose(sft_1.streamlines.get_data(),
                    sft_2.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_to_vox_equivalence():
    sft_1 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.RASMM)
    sft_2 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.RASMM)

    sft_1.to_vox()
    sft_2.to_space(Space.VOX)
    assert_allclose(sft_1.streamlines.get_data(),
                    sft_2.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_to_corner_equivalence():
    sft_1 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.VOX)
    sft_2 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.VOX)

    sft_1.to_corner()
    sft_2.to_origin(Origin.TRACKVIS)
    assert_allclose(sft_1.streamlines.get_data(),
                    sft_2.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_to_center_equivalence():
    sft_1 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.VOX)
    sft_2 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.VOX)

    sft_1.to_center()
    sft_2.to_origin(Origin.NIFTI)
    assert_allclose(sft_1.streamlines.get_data(),
                    sft_2.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_empty_sft_case():
    sft_1 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.VOX, to_origin=Origin('corner'))
    # Removing data_per_point
    sft_1 = sft_1.from_sft(sft_1.streamlines, sft_1)

    # Creating an empty set with the same spatial attributes.
    sft_2 = sft_1.from_sft([], sft_1)

    # Loaded in Vox, Corner. Modifying and checking.
    sft_1.to_rasmm()
    sft_2.to_rasmm()
    sft_1.to_center()
    sft_2.to_center()
    assert StatefulTractogram.are_compatible(sft_1, sft_2)
    assert is_header_compatible(sft_1, sft_2)


def test_trk_iterative_saving_loading():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    with TemporaryDirectory() as tmp_dir:
        save_tractogram(sft, pjoin(tmp_dir, 'gs_iter.trk'))
        tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])

        for _ in range(100):
            sft_iter = load_tractogram(pjoin(tmp_dir, 'gs_iter.trk'),
                                       FILEPATH_DIX['gs.nii'],
                                       to_space=Space.RASMM)
            assert_allclose(tmp_points_rasmm,
                            sft_iter.streamlines.get_data(),
                            atol=1e-3, rtol=1e-6)
            save_tractogram(sft_iter, 'gs_iter.trk')


def test_tck_iterative_saving_loading():
    sft = load_tractogram(FILEPATH_DIX['gs.tck'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    with TemporaryDirectory() as tmp_dir:
        save_tractogram(sft, pjoin(tmp_dir, 'gs_iter.tck'))
        tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])

        for _ in range(100):
            sft_iter = load_tractogram(pjoin(tmp_dir, 'gs_iter.tck'),
                                       FILEPATH_DIX['gs.nii'],
                                       to_space=Space.RASMM)
            assert_allclose(tmp_points_rasmm,
                            sft_iter.streamlines.get_data(),
                            atol=1e-3, rtol=1e-6)
            save_tractogram(sft_iter, pjoin(tmp_dir, 'gs_iter.tck'))


def test_trx_iterative_saving_loading():
    sft = load_tractogram(FILEPATH_DIX['gs.trx'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    with TemporaryDirectory() as tmp_dir:
        save_tractogram(sft, pjoin(tmp_dir, 'gs_iter.trx'))
        tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])

        for _ in range(100):
            sft_iter = load_tractogram(pjoin(tmp_dir, 'gs_iter.trx'),
                                       FILEPATH_DIX['gs.nii'],
                                       to_space=Space.RASMM)
            assert_allclose(tmp_points_rasmm,
                            sft_iter.streamlines.get_data(),
                            atol=1e-3, rtol=1e-6)
            save_tractogram(sft_iter, pjoin(tmp_dir, 'gs_iter.trx'))


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_fib_iterative_saving_loading():
    if not have_fury:
        return
    sft = load_tractogram(FILEPATH_DIX['gs.fib'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    with TemporaryDirectory() as tmp_dir:
        save_tractogram(sft, pjoin(tmp_dir, 'gs_iter.fib'))
        tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])

        for _ in range(100):
            sft_iter = load_tractogram(pjoin(tmp_dir, 'gs_iter.fib'),
                                       FILEPATH_DIX['gs.nii'],
                                       to_space=Space.RASMM)
            assert_allclose(tmp_points_rasmm,
                            sft_iter.streamlines.get_data(),
                            atol=1e-3, rtol=1e-6)
            save_tractogram(sft_iter, pjoin(tmp_dir, 'gs_iter.fib'))


def test_dpy_iterative_saving_loading():
    sft = load_tractogram(FILEPATH_DIX['gs.dpy'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    with TemporaryDirectory() as tmp_dir:
        save_tractogram(sft, pjoin(tmp_dir, 'gs_iter.dpy'))
        tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])

        for _ in range(100):
            sft_iter = load_tractogram(pjoin(tmp_dir, 'gs_iter.dpy'),
                                       FILEPATH_DIX['gs.nii'],
                                       to_space=Space.RASMM)
            assert_allclose(tmp_points_rasmm,
                            sft_iter.streamlines.get_data(),
                            atol=1e-3, rtol=1e-6)
            save_tractogram(sft_iter, pjoin(tmp_dir, 'gs_iter.dpy'))


def test_iterative_to_vox_transformation():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])
    for _ in range(1000):
        sft.to_vox()
        sft.to_rasmm()
        assert_allclose(tmp_points_rasmm,
                        sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_iterative_to_voxmm_transformation():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    tmp_points_rasmm = np.loadtxt(FILEPATH_DIX['gs_rasmm_space.txt'])
    for _ in range(1000):
        sft.to_voxmm()
        sft.to_rasmm()
        assert_allclose(tmp_points_rasmm,
                        sft.streamlines.get_data(), atol=1e-3, rtol=1e-6)


def test_empty_space_change():
    sft = StatefulTractogram([], FILEPATH_DIX['gs.nii'], Space.VOX)
    sft.to_vox()
    sft.to_voxmm()
    sft.to_rasmm()
    assert_array_equal([], sft.streamlines.get_data())


def test_empty_shift_change():
    sft = StatefulTractogram([], FILEPATH_DIX['gs.nii'], Space.VOX)
    sft.to_corner()
    sft.to_center()
    assert_array_equal([], sft.streamlines.get_data())


def test_empty_remove_invalid():
    sft = StatefulTractogram([], FILEPATH_DIX['gs.nii'], Space.VOX)
    sft.remove_invalid_streamlines()
    assert_array_equal([], sft.streamlines.get_data())


def test_shift_corner_from_rasmm():
    sft_1 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.VOX)
    sft_1.to_corner()
    bbox_1 = sft_1.compute_bounding_box()

    sft_2 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.RASMM)
    sft_2.to_corner()
    sft_2.to_vox()
    bbox_2 = sft_2.compute_bounding_box()

    assert_allclose(bbox_1, bbox_2, atol=1e-3, rtol=1e-6)


def test_shift_corner_from_voxmm():
    sft_1 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.VOX)
    sft_1.to_corner()
    bbox_1 = sft_1.compute_bounding_box()

    sft_2 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                            to_space=Space.VOXMM)
    sft_2.to_corner()
    sft_2.to_vox()
    bbox_2 = sft_2.compute_bounding_box()

    assert_allclose(bbox_1, bbox_2, atol=1e-3, rtol=1e-6)


def test_iterative_shift_corner():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    tmp_streamlines = sft.get_streamlines_copy()

    for _ in range(1000):
        sft._shift_voxel_origin()

    assert_allclose(sft.get_streamlines_copy(),
                    tmp_streamlines, atol=1e-3, rtol=1e-6)


def test_replace_streamlines():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    tmp_streamlines = sft.get_streamlines_copy()[::-1]

    try:
        sft.streamlines = tmp_streamlines
        assert_(True)
    except (TypeError, ValueError):
        assert_(False)


def test_subsample_streamlines():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)
    tmp_streamlines = sft.get_streamlines_copy()[0:8]

    try:
        sft.streamlines = tmp_streamlines
        assert_(False)
    except (TypeError, ValueError):
        assert_(True)


def test_reassign_both_data_sep_to_empty():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)

    try:
        sft.data_per_point = {}
        sft.data_per_streamline = {}
    except (TypeError, ValueError):
        assert_(False)

    assert_(sft.data_per_point == {} and
            sft.data_per_streamline == {})


def test_reassign_both_data_sep():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_space=Space.RASMM)

    try:
        sft.data_per_point = POINTS_DATA
        sft.data_per_streamline = STREAMLINES_DATA
        assert_(True)
    except (TypeError, ValueError):
        assert_(False)


@pytest.mark.parametrize('standard', [Origin.NIFTI, Origin.TRACKVIS])
def test_bounding_bbox_valid(standard):
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'],
                          to_origin=standard, bbox_valid_check=False)

    assert_(sft.is_bbox_in_vox_valid())


@set_random_number_generator(0)
def test_random_point_color(rng):
    sft = load_tractogram(FILEPATH_DIX['gs.tck'], FILEPATH_DIX['gs.nii'])

    random_colors = rng.integers(0, 255, (13, 8, 3))
    coloring_dict = {'colors': random_colors}

    try:
        sft.data_per_point = coloring_dict
        with TemporaryDirectory() as tmp_dir:
            save_tractogram(sft, pjoin(tmp_dir, 'random_points_color.trk'))
        assert_(True)
    except (TypeError, ValueError):
        assert_(False)


@set_random_number_generator(0)
def test_random_point_gray(rng):
    sft = load_tractogram(FILEPATH_DIX['gs.tck'], FILEPATH_DIX['gs.nii'])

    random_colors = rng.integers(0, 255, (13, 8, 1))
    coloring_dict = {
        'color_x': random_colors,
        'color_y': random_colors,
        'color_z': random_colors
    }

    try:
        sft.data_per_point = coloring_dict
        with TemporaryDirectory() as tmp_dir:
            save_tractogram(sft, pjoin(tmp_dir, 'random_points_gray.trk'))
        assert_(True)
    except (ValueError):
        assert_(False)


@set_random_number_generator(0)
def test_random_streamline_color(rng):
    sft = load_tractogram(FILEPATH_DIX['gs.tck'], FILEPATH_DIX['gs.nii'])

    uniform_colors_x = rng.integers(0, 255, (13, 1))
    uniform_colors_y = rng.integers(0, 255, (13, 1))
    uniform_colors_z = rng.integers(0, 255, (13, 1))
    uniform_colors_x = np.expand_dims(
        np.repeat(uniform_colors_x, 8, axis=1), axis=-1)
    uniform_colors_y = np.expand_dims(
        np.repeat(uniform_colors_y, 8, axis=1), axis=-1)
    uniform_colors_z = np.expand_dims(
        np.repeat(uniform_colors_z, 8, axis=1), axis=-1)

    coloring_dict = {
        'color_x': uniform_colors_x,
        'color_y': uniform_colors_y,
        'color_z': uniform_colors_z
    }

    try:
        sft.data_per_point = coloring_dict
        with TemporaryDirectory() as tmp_dir:
            save_tractogram(sft, pjoin(
                tmp_dir, 'random_streamlines_color.trk'))
        assert_(True)
    except (TypeError, ValueError):
        assert_(False)


@pytest.mark.parametrize('value, is_out_of_grid',
                         [(100, True), (-100, True), (0, False)])
def test_out_of_grid(value, is_out_of_grid):
    sft = load_tractogram(FILEPATH_DIX['gs.tck'], FILEPATH_DIX['gs.nii'])
    sft.to_vox()
    tmp_streamlines = list(sft.get_streamlines_copy())
    tmp_streamlines[0] += value

    try:
        sft.streamlines = tmp_streamlines
        assert_(sft.is_bbox_in_vox_valid() != is_out_of_grid)
    except (TypeError, ValueError):
        assert_(False)


def test_data_per_point_consistency_addition():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    sft_first_half = sft[0:7]
    sft_last_half = sft[7:13]

    sft_first_half.data_per_point = {}
    try:
        _ = sft_first_half + sft_last_half
        assert_(False)
    except ValueError:
        assert_(True)


def test_data_per_streamline_consistency_addition():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    sft_first_half = sft[0:7]
    sft_last_half = sft[7:13]

    sft_first_half.data_per_streamline = {}
    try:
        _ = sft_first_half + sft_last_half
        assert_(False)
    except ValueError:
        assert_(True)


def test_space_consistency_addition():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    sft_first_half = sft[0:7]
    sft_last_half = sft[7:13]

    sft_first_half.to_vox()
    try:
        _ = sft_first_half + sft_last_half
        assert_(False)
    except ValueError:
        assert_(True)


def test_origin_consistency_addition():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    sft_first_half = sft[0:7]
    sft_last_half = sft[7:13]

    sft_first_half.to_corner()
    try:
        _ = sft_first_half + sft_last_half
        assert_(False)
    except ValueError:
        assert_(True)


def test_space_attributes_consistency_addition():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    sft_switch = StatefulTractogram(sft.streamlines,
                                    FILEPATH_DIX['gs_3mm.nii'],
                                    Space.RASMM)

    try:
        _ = sft + sft_switch
        assert_(False)
    except ValueError:
        assert_(True)


def test_equality():
    sft_1 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    sft_2 = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])

    assert_(sft_1 == sft_2,
            msg='Identical sft should be equal (==)')


def test_basic_slicing():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    first_streamline_sft = sft[0]

    npt.assert_allclose(first_streamline_sft.streamlines[0][0],
                        [11.149319, 21.579943, 37.600685], rtol=1e-6,
                        err_msg='streamlines were not sliced correctly')
    rgb = np.array([first_streamline_sft.data_per_point['color_x'][0][0],
                    first_streamline_sft.data_per_point['color_y'][0][0],
                    first_streamline_sft.data_per_point['color_z'][0][0]])
    npt.assert_allclose(np.squeeze(rgb), [220., 20., 60.],
                        err_msg='data_per_point were not sliced correctly')
    rand_coord = first_streamline_sft.data_per_streamline['random_coord']
    npt.assert_allclose(np.squeeze(rand_coord), [7., 1., 5.], rtol=1e-3,
                        err_msg='data_per_streamline were not sliced correctly')


def test_space_side_effect_slicing():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    first_streamline = deepcopy(sft.streamlines[0])

    first_streamline_sft = sft[0]
    sft.to_vox()
    npt.assert_allclose(first_streamline_sft.streamlines[0], first_streamline,
                        rtol=1e-6,
                        err_msg='Side effect, modifying a StatefulTractogram '
                                'after slicing should not modify the slice')
    # Testing it both ways
    sft.to_rasmm()
    first_streamline_sft.to_vox()
    npt.assert_allclose(sft.streamlines[0], first_streamline, rtol=1e-6,
                        err_msg='Side effect, modifying a StatefulTractogram '
                                'after slicing should not modify the slice')


def test_origin_side_effect_slicing():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    first_streamline = deepcopy(sft.streamlines[0])

    first_streamline_sft = sft[0]
    sft.to_corner()
    npt.assert_allclose(first_streamline_sft.streamlines[0], first_streamline,
                        rtol=1e-6,
                        err_msg='Side effect, modifying a StatefulTractogram '
                                'after slicing should not modify the slice')
    # Testing it both ways
    sft.to_center()
    first_streamline_sft.to_corner()
    npt.assert_allclose(sft.streamlines[0], first_streamline, rtol=1e-6,
                        err_msg='Side effect, modifying a StatefulTractogram '
                                'after slicing should not modify the slice')


def test_advanced_slicing():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    last_streamline_sft = sft[::-1][0]

    npt.assert_allclose(last_streamline_sft.streamlines[0][0],
                        [14.389803, 27.857153, 39.3602], rtol=1e-6,
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
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    sft_first_half = sft[0:7]
    sft_last_half = sft[7:13]

    concatenate_sft = sft_first_half + sft_last_half
    assert_(concatenate_sft == sft,
            msg='sft were not added correctly')


def test_space_side_effect_addition():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
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
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
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


def test_invalid_streamlines():

    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
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
    assert_(len(sft) == expected_len_sft,
            msg='The shifted gold standard should have {} invalid streamlines'.
            format(src_strml_count - expected_len_sft))


def test_invalid_streamlines_epsilon():

    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
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


def test_create_from_sft():
    sft_1 = load_tractogram(FILEPATH_DIX['gs.tck'], FILEPATH_DIX['gs.nii'])
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
        assert_(True, msg='Streamlines, space attributes, space, origin, '
                          'data_per_point and data_per_streamline should '
                          'be identical')

    # Side effect testing
    sft_1.streamlines = np.arange(6000).reshape((100, 20, 3))
    if np.array_equal(sft_1.streamlines, sft_2.streamlines):
        assert_(True, msg='Side effect, modifying the original '
                          'StatefulTractogram after creating a new one '
                          'should not modify the new one')


def test_init_dtype_dict_attributes():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    dtype_dict = {'positions': np.float32,
                  'offsets': np.int_,
                  'dpp': {'color_x': np.float32,
                          'color_y': np.float32,
                          'color_z': np.float32},
                  'dps': {'random_coord': np.float32}}

    try:
        recursive_compare(dtype_dict, sft.dtype_dict)
    except ValueError as e:
        print(e)
        assert_(False, msg=e)


def test_set_dtype_dict_attributes():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    dtype_dict = {'positions': np.float16,
                  'offsets': np.int32,
                  'dpp': {'color_x': np.uint8,
                          'color_y': np.uint8,
                          'color_z': np.uint8},
                  'dps': {'random_coord': np.float64}}

    sft.dtype_dict = dtype_dict
    try:
        recursive_compare(dtype_dict, sft.dtype_dict)
    except ValueError:
        assert_(False, msg='dtype_dict should be identical after set.')


def test_set_partial_dtype_dict_attributes():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    dtype_dict = {'positions': np.float16,
                  'offsets': np.int32}
    dpp_dtype_dict = {'dpp': {'color_x': np.float32,
                              'color_y': np.float32,
                              'color_z': np.float32}}
    dps_dtype_dict = {'dps': {'random_coord': np.float32}}

    # Set only positions and offsets
    sft.dtype_dict = dtype_dict

    try:
        recursive_compare(dtype_dict['positions'], sft.dtype_dict['positions'])
        recursive_compare(dtype_dict['offsets'], sft.dtype_dict['offsets'])
        recursive_compare(dpp_dtype_dict['dpp'], sft.dtype_dict['dpp'])
        recursive_compare(dps_dtype_dict['dps'], sft.dtype_dict['dps'])
    except ValueError:
        assert_(False, msg='Partial use of dtype_dict should apply only to the '
                'relevant portions.')


def test_non_existing_dtype_dict_attributes():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    dtype_dict = {'dpp': {'color_a': np.uint8,   # Fake
                          'color_b': np.uint8,   # Fake
                          'color_c': np.uint8},  # Fake
                  'dps': {'random_coordinates': np.float64},  # Fake
                  'dps2': {'random_coord': np.float64}}       # Fake

    sft.dtype_dict = dtype_dict
    try:
        recursive_compare(sft.dtype_dict, dtype_dict)
        assert_(False, msg='Fake entries in dtype_dict should not work.')
    except ValueError:
        assert_(True)


def test_from_sft_dtype_dict_attributes():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    dtype_dict = {'positions': np.float16,
                  'offsets': np.int32,
                  'dpp': {'color_x': np.uint8,
                          'color_y': np.uint8,
                          'color_z': np.uint8},
                  'dps': {'random_coord': np.float64}}

    sft.dtype_dict = dtype_dict
    new_sft = StatefulTractogram.from_sft(sft.streamlines, sft,
                                          data_per_point=sft.data_per_point,
                                          data_per_streamline=sft.data_per_streamline)
    try:
        recursive_compare(new_sft.dtype_dict, dtype_dict)
        recursive_compare(sft.dtype_dict, dtype_dict)
    except ValueError:
        assert_(False, msg='from_sft() should not modify the dtype_dict.')


def test_slicing_dtype_dict_attributes():
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])
    dtype_dict = {'positions': np.float16,
                  'offsets': np.int32,
                  'dpp': {'color_x': np.uint8,
                          'color_y': np.uint8,
                          'color_z': np.uint8},
                  'dps': {'random_coord': np.float64}}

    sft.dtype_dict = dtype_dict
    new_sft = sft[::2]

    try:
        recursive_compare(new_sft.dtype_dict, dtype_dict)
        recursive_compare(sft.dtype_dict, dtype_dict)
    except ValueError:
        assert_(False, msg='Slicing should not modify the dtype_dict.')


def recursive_compare(d1, d2, level='root'):
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            s1 = set(d1.keys())
            s2 = set(d2.keys())
            common_keys = s1 & s2
            if s1 - s2:
                raise ValueError('Keys {} in d1 but not in d2'.format(s1 - s2))
        else:
            common_keys = set(d1.keys())

        for k in common_keys:
            recursive_compare(d1[k], d2[k], level='{}.{}'.format(level, k))

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            raise ValueError('Lists do not have the same length at level {}'.format(
                level))
        common_len = min(len(d1), len(d2))

        for i in range(common_len):
            recursive_compare(d1[i], d2[i], level='{}[{}]'.format(level, i))

    else:
        if np.dtype(d1).itemsize != np.dtype(d2).itemsize:
            raise ValueError(
                'Values {}, {} do not match at level {}'.format(d1, d2, level))
