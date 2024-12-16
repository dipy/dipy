import itertools
import os
import pytest
from tempfile import TemporaryDirectory

import numpy as np
import numpy.testing as npt

from dipy.io.stateful_tractogram import StatefulTractogram as _  # fake import
from dipy.io.stateful_surface import StatefulSurface
from dipy.io.surface import load_surface, save_surface
from dipy.io.utils import Space, Origin
from dipy.utils.optpkg import optional_package


fury, have_fury, setup_module = optional_package("fury", min_version="0.8.0")

CWD = "/home/rhef1902/Datasets/stateful_surface/"


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_empty_change_space():
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))
    sfs = StatefulSurface(([], []), 'mni_masked.nii.gz',
                          space=Space.RASMM, origin=Origin.NIFTI)

    # Test all space combinations
    sfs.to_vox()
    sfs.to_voxmm()
    sfs.to_rasmm()
    sfs.to_lpsmm()

    npt.assert_equal([], sfs.vertices)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_empty_change_origin():
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))
    sfs = StatefulSurface(([], []), 'mni_masked.nii.gz',
                          space=Space.RASMM, origin=Origin.NIFTI)

    # Test all origin combinations
    sfs.to_center()
    sfs.to_corner()

    npt.assert_equal([], sfs.vertices)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
@pytest.mark.parametrize("space", [Space.LPSMM, Space.RASMM, Space.VOXMM, Space.VOX])
def test_to_space_equivalent_to_rasmm(space):
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))

    # Load initial surface and convert to rasmm directly
    sfs = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz')
    sfs.to_rasmm()
    ref_vertices = sfs.vertices.copy()

    # Load surface again and use to_space
    sfs = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz')
    sfs.to_space(Space.RASMM)

    npt.assert_allclose(ref_vertices, sfs.vertices, atol=1e-3, rtol=1e-6)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
@pytest.mark.parametrize("origin", [Origin.NIFTI, Origin.TRACKVIS])
def test_to_origin_equivalent_to_center(origin):
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))

    # Load initial surface and convert to center directly
    sfs = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz')
    sfs.to_center()
    ref_vertices = sfs.vertices.copy()

    # Load surface again and use to_origin
    sfs = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz')
    sfs.to_origin(Origin.NIFTI)

    npt.assert_allclose(ref_vertices, sfs.vertices, atol=1e-3, rtol=1e-6)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
@pytest.mark.parametrize("space", [Space.LPSMM, Space.RASMM, Space.VOXMM, Space.VOX])
def test_change_origin_from_space(space):
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))
    sfs = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz')
    sfs.to_space(space)
    sfs.to_center()
    ref_vertices = sfs.vertices.copy()

    sfs.to_corner()
    sfs.to_center()

    npt.assert_allclose(ref_vertices, sfs.vertices, atol=1e-3, rtol=1e-6)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
@pytest.mark.parametrize("space", [Space.LPSMM, Space.RASMM, Space.VOXMM, Space.VOX])
def test_change_space_many_times(space):
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))
    sfs = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz')
    sfs.to_space(space)
    ref_vertices = sfs.vertices.copy()

    # Call it twice, should not do anything
    sfs.to_space(space)
    sfs.to_space(space)

    npt.assert_allclose(ref_vertices, sfs.vertices, atol=1e-3, rtol=1e-6)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
@pytest.mark.parametrize("origin", [Origin.NIFTI, Origin.TRACKVIS])
def test_change_space_many_times_with_origin(origin):
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))
    sfs = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz')
    sfs.to_origin(origin)
    ref_vertices = sfs.vertices.copy()

    # Call it twice, should not do anything
    sfs.to_origin(origin)
    sfs.to_origin(origin)

    npt.assert_allclose(ref_vertices, sfs.vertices, atol=1e-3, rtol=1e-6)


# Test out of grid
@pytest.mark.skipif(not have_fury, reason="Requires FURY")
@pytest.mark.parametrize(
    "value, is_out_of_grid", [(1000, True), (-1000, True), (0, False)])
def test_out_of_grid(value, is_out_of_grid):
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))
    sfs = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz')
    sfs.to_vox()
    sfs.to_corner()

    tmp_vertices = sfs.vertices.copy()
    tmp_vertices[0] += value

    try:
        sfs.vertices = tmp_vertices
        npt.assert_(sfs.is_bbox_in_vox_valid() != is_out_of_grid)
    except (TypeError, ValueError):
        npt.assert_(False)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_invalid_empty():
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))
    sfs = StatefulSurface(([], []), 'mni_masked.nii.gz',
                          space=Space.RASMM, origin=Origin.NIFTI)
    sfs.to_vox()
    sfs.to_corner()

    try:
        sfs.is_bbox_in_vox_valid()
    except (TypeError, ValueError):
        npt.assert_(True)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_equality():
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))
    sfs_1 = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz')
    sfs_2 = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz',
                         from_space=Space.LPSMM, from_origin=Origin.NIFTI)
    sfs_1.to_rasmm()
    sfs_1.to_center()
    sfs_2.to_rasmm()
    sfs_2.to_center()

    npt.assert_(sfs_1 == sfs_2)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_random_space_transformations():
    """Test multiple random space transformations maintain vertex positions"""
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))
    sfs = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz')

    # Store initial state
    sfs.to_rasmm()
    sfs.to_center()
    initial_vertices = sfs.vertices.copy()

    # List of possible spaces
    spaces = [Space.LPSMM, Space.RASMM, Space.VOXMM, Space.VOX]
    origins = [Origin.NIFTI, Origin.TRACKVIS]

    # Apply 100 random transformations
    for _ in range(100):
        space = np.random.choice(spaces, 1, replace=False)
        origin = np.random.choice(origins, 1, replace=False)
        sfs.to_space(space)
        sfs.to_origin(origin)

    # Return to initial space and compare
    sfs.to_rasmm()
    sfs.to_center()
    npt.assert_almost_equal(initial_vertices, sfs.vertices, decimal=5)


spaces = [Space.LPSMM, Space.RASMM, Space.VOXMM, Space.VOX]
origins = [Origin.NIFTI, Origin.TRACKVIS]


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
@pytest.mark.parametrize("space, origin", list(itertools.product(spaces, origins)))
def test_space_origin_gold_standard(space, origin):
    os.chdir(os.path.join(os.path.expanduser(CWD), 'toy_data'))
    sfs = load_surface(f'gs_{space.value.lower()}_{origin.value.lower()}.ply',
                       'gs.nii', from_space=space, from_origin=origin,
                       to_space=space, to_origin=origin)

    # Test in the space it was loaded from
    vertices = np.loadtxt(
        f'gs_{space.value.lower()}_{origin.value.lower()}.txt')
    faces = np.loadtxt(f'faces.txt')
    npt.assert_allclose(vertices, sfs.vertices, atol=1e-3, rtol=1e-6)
    npt.assert_allclose(faces, sfs.faces, atol=1e-3, rtol=1e-6)

    # Test in a standard space
    sfs.to_rasmm()
    sfs.to_center()
    vertices = np.loadtxt(f'gs_rasmm_center.txt')
    npt.assert_allclose(vertices, sfs.vertices, atol=1e-3, rtol=1e-6)


@pytest.mark.parametrize("extension", [".vtk", ".gii", ".pial"])
@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_save_load_many_times(extension):
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))

    # Load initial surface
    sfs = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz')
    ref_vertices = sfs.vertices.copy()

    with TemporaryDirectory() as tmpdir:
        # Save and load 10 times
        for i in range(10):
            save_surface(os.path.join(tmpdir, f'test_{i}.{extension}'), sfs)
            sfs = load_surface(os.path.join(
                tmpdir, f'test_{i}.{extension}'), 'mni_masked.nii.gz')

        # Final vertices should match original
        npt.assert_almost_equal(ref_vertices, sfs.vertices, decimal=5)
