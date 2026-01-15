import itertools
from os.path import join as pjoin
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.io.stateful_surface import StatefulSurface
from dipy.io.surface import load_surface, save_surface
from dipy.io.utils import Origin, Space, recursive_compare
from dipy.utils.optpkg import optional_package

vtk, have_vtk, setup_module = optional_package(
    "vtk", min_version="9.0.0", max_version="9.1.0"
)
SPACES = [Space.LPSMM, Space.RASMM, Space.VOXMM, Space.VOX]
ORIGINS = [Origin.NIFTI, Origin.TRACKVIS]

FILEPATH_DIX = None


def setup_module():
    global FILEPATH_DIX
    try:
        FILEPATH_DIX, _, _ = get_fnames(name="gold_standard_io")
        FILEPATH_DIX.update(get_fnames(name="real_data_io"))
    except (HTTPError, URLError) as e:
        FILEPATH_DIX = None
        error_msg = f'"Tests Data failed to download." Reason: {e}'
        pytest.skip(error_msg, allow_module_level=True)
        return


def test_empty_change_space():
    sfs = StatefulSurface(
        [],
        [],
        FILEPATH_DIX["naf_mni_masked.nii.gz"],
        space=Space.RASMM,
        origin=Origin.NIFTI,
    )

    # Test all space combinations
    sfs.to_vox()
    sfs.to_voxmm()
    sfs.to_rasmm()
    sfs.to_lpsmm()

    npt.assert_equal([], sfs.vertices)


def test_empty_change_origin():
    sfs = StatefulSurface(
        [],
        [],
        FILEPATH_DIX["naf_mni_masked.nii.gz"],
        space=Space.RASMM,
        origin=Origin.NIFTI,
    )

    # Test all origin combinations
    sfs.to_center()
    sfs.to_corner()

    npt.assert_equal([], sfs.vertices)


@pytest.mark.parametrize("space", [Space.LPSMM, Space.RASMM, Space.VOXMM, Space.VOX])
def test_to_space_equivalent_to_rasmm(space):
    # Load initial surface and convert to rasmm directly
    sfs = load_surface(
        FILEPATH_DIX["naf_lh.pial"],
        FILEPATH_DIX["naf_mni_masked.nii.gz"],
    )
    if space == Space.RASMM:
        sfs.to_rasmm()
    elif space == Space.LPSMM:
        sfs.to_lpsmm()
    elif space == Space.VOXMM:
        sfs.to_voxmm()
    elif space == Space.VOX:
        sfs.to_vox()
    ref_vertices = sfs.vertices.copy()

    # Load surface again and use to_space
    sfs = load_surface(
        FILEPATH_DIX["naf_lh.pial"], FILEPATH_DIX["naf_mni_masked.nii.gz"]
    )
    sfs.to_space(space)

    npt.assert_allclose(ref_vertices, sfs.vertices, atol=1e-3, rtol=1e-6)


@pytest.mark.parametrize("origin", [Origin.NIFTI, Origin.TRACKVIS])
def test_to_origin_equivalent_to_center(origin):
    # Load initial surface and convert to center directly
    sfs = load_surface(
        FILEPATH_DIX["naf_lh.pial"], FILEPATH_DIX["naf_mni_masked.nii.gz"]
    )

    if origin == Origin.NIFTI:
        sfs.to_center()
    elif origin == Origin.TRACKVIS:
        sfs.to_corner()
    ref_vertices = sfs.vertices.copy()

    # Load surface again and use to_origin
    sfs = load_surface(
        FILEPATH_DIX["naf_lh.pial"], FILEPATH_DIX["naf_mni_masked.nii.gz"]
    )
    sfs.to_origin(origin)

    npt.assert_allclose(ref_vertices, sfs.vertices, atol=1e-3, rtol=1e-6)


@pytest.mark.parametrize("space", [Space.LPSMM, Space.RASMM, Space.VOXMM, Space.VOX])
def test_change_origin_from_space(space):
    sfs = load_surface(
        FILEPATH_DIX["naf_lh.pial"], FILEPATH_DIX["naf_mni_masked.nii.gz"]
    )
    sfs.to_space(space)
    sfs.to_center()
    ref_vertices = sfs.vertices.copy()

    sfs.to_corner()
    sfs.to_center()

    npt.assert_allclose(ref_vertices, sfs.vertices, atol=1e-3, rtol=1e-6)


@pytest.mark.parametrize("space", [Space.LPSMM, Space.RASMM, Space.VOXMM, Space.VOX])
def test_change_space_many_times(space):
    sfs = load_surface(
        FILEPATH_DIX["naf_lh.pial"], FILEPATH_DIX["naf_mni_masked.nii.gz"]
    )
    sfs.to_space(space)
    ref_vertices = sfs.vertices.copy()

    # Call it twice, should not do anything
    sfs.to_space(space)
    sfs.to_space(space)

    npt.assert_allclose(ref_vertices, sfs.vertices, atol=1e-3, rtol=1e-6)


@pytest.mark.parametrize("origin", [Origin.NIFTI, Origin.TRACKVIS])
def test_change_space_many_times_with_origin(origin):
    sfs = load_surface(
        FILEPATH_DIX["naf_lh.pial"], FILEPATH_DIX["naf_mni_masked.nii.gz"]
    )
    sfs.to_origin(origin)
    ref_vertices = sfs.vertices.copy()

    # Call it twice, should not do anything
    sfs.to_origin(origin)
    sfs.to_origin(origin)

    npt.assert_allclose(ref_vertices, sfs.vertices, atol=1e-3, rtol=1e-6)


# Test out of grid
@pytest.mark.parametrize(
    "value, is_out_of_grid", [(1000, True), (-1000, True), (0, False)]
)
def test_out_of_grid(value, is_out_of_grid):
    sfs = load_surface(
        FILEPATH_DIX["naf_lh.pial"], FILEPATH_DIX["naf_mni_masked.nii.gz"]
    )
    sfs.to_vox()
    sfs.to_corner()

    tmp_vertices = sfs.vertices.copy()
    tmp_vertices[0] += value

    try:
        sfs.vertices = tmp_vertices
        npt.assert_(sfs.is_bbox_in_vox_valid() != is_out_of_grid)
    except (TypeError, ValueError):
        npt.assert_(False)


def test_invalid_empty():
    sfs = StatefulSurface(
        [],
        [],
        FILEPATH_DIX["naf_mni_masked.nii.gz"],
        space=Space.RASMM,
        origin=Origin.NIFTI,
    )
    sfs.to_vox()
    sfs.to_corner()

    try:
        sfs.is_bbox_in_vox_valid()
    except (TypeError, ValueError):
        npt.assert_(True)


def test_equality():
    sfs_1 = load_surface(
        FILEPATH_DIX["naf_lh.pial"], FILEPATH_DIX["naf_mni_masked.nii.gz"]
    )
    sfs_2 = load_surface(
        FILEPATH_DIX["naf_lh.pial"],
        FILEPATH_DIX["naf_mni_masked.nii.gz"],
        from_space=Space.LPSMM,
        from_origin=Origin.NIFTI,
    )
    sfs_1.to_rasmm()
    sfs_1.to_center()
    sfs_2.to_rasmm()
    sfs_2.to_center()

    npt.assert_(sfs_1 == sfs_2)


def test_random_space_transformations():
    sfs = load_surface(
        FILEPATH_DIX["naf_lh.pial"], FILEPATH_DIX["naf_mni_masked.nii.gz"]
    )

    # Store initial state
    sfs.to_rasmm()
    sfs.to_center()
    initial_vertices = sfs.vertices.copy()

    # Apply 100 random transformations
    for _ in range(100):
        space = np.random.choice(SPACES, 1, replace=False)
        origin = np.random.choice(ORIGINS, 1, replace=False)
        sfs.to_space(space)
        sfs.to_origin(origin)

    # Return to initial space and compare
    sfs.to_rasmm()
    sfs.to_center()
    npt.assert_almost_equal(initial_vertices, sfs.vertices, decimal=5)


@pytest.mark.skipif(not have_vtk, reason="Requires VTK")
@pytest.mark.parametrize("space, origin", itertools.product(SPACES, ORIGINS))
def test_space_origin_gold_standard(space, origin):
    fname = FILEPATH_DIX[f"gs_mesh_{space.value.lower()}_{origin.value.lower()}.ply"]
    sfs = load_surface(
        fname,
        FILEPATH_DIX["gs_volume.nii"],
        from_space=space,
        from_origin=origin,
        to_space=space,
        to_origin=origin,
    )

    # Test in the space it was loaded from
    vertices = np.loadtxt(fname.with_suffix(".txt"))
    faces = np.loadtxt(FILEPATH_DIX["gs_mesh_faces.txt"])
    npt.assert_allclose(vertices, sfs.vertices, atol=1e-3, rtol=1e-6)
    npt.assert_allclose(faces, sfs.faces, atol=1e-3, rtol=1e-6)

    # Test in a standard space
    sfs.to_rasmm()
    sfs.to_center()
    vertices = np.loadtxt(FILEPATH_DIX["gs_mesh_rasmm_center.txt"])
    npt.assert_allclose(vertices, sfs.vertices, atol=1e-3, rtol=1e-6)


def test_equivalent_gii():
    fname = FILEPATH_DIX["gs_mesh.gii"]
    sfs = load_surface(fname, FILEPATH_DIX["gs_volume.nii"])

    # Test in the space it was loaded from
    faces = np.loadtxt(FILEPATH_DIX["gs_mesh_faces.txt"])
    npt.assert_allclose(faces, sfs.faces, atol=1e-3, rtol=1e-6)

    # Test in a standard space
    sfs.to_rasmm()
    sfs.to_center()
    vertices = np.loadtxt(FILEPATH_DIX["gs_mesh_rasmm_center.txt"])
    npt.assert_allclose(vertices, sfs.vertices, atol=1e-3, rtol=1e-6)


@pytest.mark.skipif(not have_vtk, reason="Requires VTK")
def test_create_from_sfs():
    sfs_1 = load_surface(
        FILEPATH_DIX["gs_mesh_rasmm_center.ply"], FILEPATH_DIX["gs_volume.nii"]
    )
    sfs_2 = StatefulSurface.from_sfs(
        sfs_1.vertices, sfs_1, data_per_vertex=sfs_1.data_per_vertex
    )

    if not (sfs_1 == sfs_2):
        npt.assert_(
            True,
            msg="vertices, faces, space attributes, space, origin, "
            "and data_per_vertex should be identical",
        )

    nb_pts = sfs_1.vertices.shape[0]
    sfs_1.vertices = np.arange(nb_pts * 3).reshape((nb_pts, 3))
    if np.array_equal(sfs_1.vertices, sfs_2.vertices):
        npt.assert_(
            True,
            msg="Side effect, modifying the original "
            "StatefulTractogram after creating a new one "
            "should not modify the new one",
        )


@pytest.mark.skipif(not have_vtk, reason="Requires VTK")
def test_init_dtype_dict_attributes():
    sfs = load_surface(
        FILEPATH_DIX["gs_mesh_rasmm_center.ply"], FILEPATH_DIX["gs_volume.nii"]
    )
    dtype_dict = {
        "vertices": np.float64,
        "faces": np.uint32,
    }

    try:
        recursive_compare(dtype_dict, sfs.dtype_dict)
    except ValueError as e:
        print(e)
        npt.assert_(False, msg=e)


@pytest.mark.skipif(not have_vtk, reason="Requires VTK")
def test_set_dtype_dict_attributes():
    sfs = load_surface(
        FILEPATH_DIX["gs_mesh_rasmm_center.ply"], FILEPATH_DIX["gs_volume.nii"]
    )
    sfs.data_per_vertex = {
        "normal": np.zeros((sfs.vertices.shape[0], 3), dtype=np.float16),
    }
    dtype_dict = {
        "vertices": np.float16,
        "faces": np.int32,
        "dpp": {"normal": np.float16},
    }

    sfs.dtype_dict = dtype_dict
    try:
        recursive_compare(dtype_dict, sfs.dtype_dict)
    except ValueError:
        npt.assert_(False, msg="dtype_dict should be identical after set.")


@pytest.mark.skipif(not have_vtk, reason="Requires VTK")
def test_set_partial_dtype_dict_attributes():
    sfs = load_surface(
        FILEPATH_DIX["gs_mesh_rasmm_center.ply"], FILEPATH_DIX["gs_volume.nii"]
    )
    sfs.data_per_vertex = {
        "normal": np.zeros((sfs.vertices.shape[0], 3), dtype=np.float16),
    }
    dtype_dict = {"vertices": np.float16, "faces": np.int32}
    dpp_dtype_dict = {
        "normal": np.float16,
    }

    # Set only vertices and faces
    sfs.dtype_dict = dtype_dict

    try:
        recursive_compare(dtype_dict["vertices"], sfs.dtype_dict["vertices"])
        recursive_compare(dtype_dict["faces"], sfs.dtype_dict["faces"])
        recursive_compare(dpp_dtype_dict, sfs.dtype_dict["dpp"])
    except ValueError:
        npt.assert_(
            False,
            msg="Partial use of dtype_dict should apply only to the relevant portions.",
        )


@pytest.mark.skipif(not have_vtk, reason="Requires VTK")
def test_non_existing_dtype_dict_attributes():
    sfs = load_surface(
        FILEPATH_DIX["gs_mesh_rasmm_center.ply"], FILEPATH_DIX["gs_volume.nii"]
    )
    dtype_dict = {
        "dpp": {
            "color_x": np.uint8,  # Fake
            "color_y": np.uint8,  # Fake
            "color_z": np.uint8,
        },  # Fake
        "fake_attr": {"random_value": np.float64},  # Fake
    }

    sfs.dtype_dict = dtype_dict
    try:
        recursive_compare(sfs.dtype_dict, dtype_dict)
        npt.assert_(False, msg="Fake entries in dtype_dict should not work.")
    except ValueError:
        npt.assert_(True)


@pytest.mark.skipif(not have_vtk, reason="Requires VTK")
def test_from_sfs_dtype_dict_attributes():
    sfs = load_surface(
        FILEPATH_DIX["gs_mesh_rasmm_center.ply"], FILEPATH_DIX["gs_volume.nii"]
    )
    sfs.data_per_vertex = {
        "color_r": np.zeros((sfs.vertices.shape[0], 3), dtype=np.uint16),
        "color_g": np.zeros((sfs.vertices.shape[0], 3), dtype=np.uint16),
        "color_b": np.zeros((sfs.vertices.shape[0], 3), dtype=np.uint16),
    }
    dtype_dict = {
        "vertices": np.float16,
        "faces": np.int32,
        "dpp": {"color_r": np.uint8, "color_g": np.uint8, "color_b": np.uint8},
    }

    sfs.dtype_dict = dtype_dict
    new_sfs = StatefulSurface.from_sfs(
        sfs.vertices, sfs, data_per_vertex=sfs.data_per_vertex
    )
    try:
        recursive_compare(new_sfs.dtype_dict, dtype_dict)
        recursive_compare(sfs.dtype_dict, dtype_dict)
    except ValueError:
        npt.assert_(False, msg="from_sfs() should not modify the dtype_dict.")


@pytest.mark.skipif(not have_vtk, reason="Requires VTK")
@pytest.mark.parametrize("extension", ["vtk", "gii", "pial"])
def test_save_load_many_times(extension):
    # Load initial surface
    sfs = load_surface(
        FILEPATH_DIX["naf_lh.pial"], FILEPATH_DIX["naf_mni_masked.nii.gz"]
    )
    ref_vertices = sfs.vertices.copy()

    with TemporaryDirectory() as tmpdir:
        # Save and load 10 times
        for i in range(10):
            save_surface(sfs, pjoin(tmpdir, f"test_{i}.{extension}"))
            sfs = load_surface(
                pjoin(tmpdir, f"test_{i}.{extension}"),
                FILEPATH_DIX["naf_mni_masked.nii.gz"],
            )

        # Final vertices should match original
        npt.assert_almost_equal(ref_vertices, sfs.vertices, decimal=5)
