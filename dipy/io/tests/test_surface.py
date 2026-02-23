import itertools
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.io.surface import load_surface, save_surface
from dipy.io.utils import Origin, Space
from dipy.utils.optpkg import optional_package

vtk, have_vtk, setup_module = optional_package(
    "vtk", min_version="9.0.0", max_version="9.1.0"
)

FOLDERS_GII = ["ascii", "base64", "gzip_base64"]
FILENAMES_GII = ["pial.L.surf.gii", "smoothwm.L.surf.gii"]

FOLDERS = ["big_affine_freesurfer", "small_affine_freesurfer"]
HEMISPHERES = ["lh", "rh"]
TYPES = ["pial", "smoothwm", "orig"]

SPACES = [Space.RASMM, Space.LPSMM, Space.VOXMM, Space.VOX]
ORIGINS = [Origin.NIFTI, Origin.TRACKVIS]


def setup_module():
    global FILEPATH_DIX, POINTS_DATA, STREAMLINES_DATA
    try:
        FILEPATH_DIX = get_fnames(name="real_data_io")
    except (HTTPError, URLError) as e:
        FILEPATH_DIX, POINTS_DATA, STREAMLINES_DATA = None, None, None
        error_msg = f'"Tests Data failed to download." Reason: {e}'
        pytest.skip(error_msg, allow_module_level=True)
        return


@pytest.mark.skipif(not have_vtk, reason="Requires VTK")
def test_pial_load_save():
    data_raw = nib.freesurfer.read_geometry(FILEPATH_DIX["naf_lh.pial"])

    sfs = load_surface(
        FILEPATH_DIX["naf_lh.pial"], FILEPATH_DIX["naf_mni_masked.nii.gz"]
    )

    # Change the space/origin, should not affect the saved pial
    sfs.to_vox()
    sfs.to_corner()

    with TemporaryDirectory() as tmpdir:
        save_surface(
            sfs, Path(tmpdir) / "lh.pial", ref_pial=FILEPATH_DIX["naf_lh.pial"]
        )
        data_save = nib.freesurfer.read_geometry(Path(tmpdir) / "lh.pial")
    npt.assert_almost_equal(data_raw[0], data_save[0], decimal=5)


@pytest.mark.skipif(not have_vtk, reason="Requires VTK")
@pytest.mark.parametrize("space,origin", list(itertools.product(SPACES, ORIGINS)))
def test_vtk_matching_space(space, origin):
    sfs = load_surface(
        FILEPATH_DIX["naf_lh.pial"], FILEPATH_DIX["naf_mni_masked.nii.gz"]
    )
    sfs.to_rasmm()
    sfs.to_center()
    ref_vertices = sfs.vertices.copy()

    with TemporaryDirectory() as tmpdir:
        save_surface(sfs, Path(tmpdir) / "tmp.vtk", to_space=space, to_origin=origin)
        sfs = load_surface(
            Path(tmpdir) / "tmp.vtk",
            FILEPATH_DIX["naf_mni_masked.nii.gz"],
            from_space=space,
            from_origin=origin,
        )

        sfs.to_rasmm()
        sfs.to_center()
        save_vertices = sfs.vertices.copy()
        npt.assert_almost_equal(ref_vertices, save_vertices, decimal=5)


@pytest.mark.skipif(not have_vtk, reason="Requires VTK")
@pytest.mark.parametrize(
    "type,fname,space,origin",
    list(itertools.product(FOLDERS_GII, FILENAMES_GII, SPACES, ORIGINS)),
)
def test_gifti_matching_space(type, fname, space, origin):
    if type == "gzip_base64":
        fname += ".gz"
    sfs = load_surface(FILEPATH_DIX[fname], FILEPATH_DIX["anat.nii.gz"])
    sfs.to_rasmm()
    sfs.to_center()
    ref_vertices = sfs.vertices.copy()

    with TemporaryDirectory() as tmpdir:
        save_surface(sfs, Path(tmpdir) / "tmp.gii", to_space=space, to_origin=origin)
        sfs = load_surface(
            Path(tmpdir) / "tmp.gii",
            FILEPATH_DIX["anat.nii.gz"],
            from_space=space,
            from_origin=origin,
        )

        sfs.to_rasmm()
        sfs.to_center()
        save_vertices = sfs.vertices.copy()
        npt.assert_almost_equal(ref_vertices, save_vertices, decimal=5)


@pytest.mark.skipif(not have_vtk, reason="Requires VTK")
@pytest.mark.parametrize(
    "dataset,hemisphere,type", list(itertools.product(FOLDERS, HEMISPHERES, TYPES))
)
def test_freesurfer_density_operation(dataset, hemisphere, type):
    prefix = "baf" if dataset == "big_affine_freesurfer" else "saf"
    fname = f"{prefix}_{hemisphere}.{type}"
    sfs = load_surface(FILEPATH_DIX[fname], FILEPATH_DIX[f"{prefix}_t1.nii.gz"])

    assert sfs.is_bbox_in_vox_valid()

    data = np.zeros(sfs.dimensions, dtype=np.uint32)

    sfs.to_vox()
    sfs.to_corner()

    # Compute density map in the numpy grid
    for vertice in sfs.vertices:
        coord = tuple(vertice.astype(np.int32))
        data[coord] += 1

    with TemporaryDirectory() as tmpdir:
        nib.save(
            nib.Nifti1Image(data, sfs.affine),
            Path(tmpdir) / f"{hemisphere}_{type}.nii.gz",
        )

    # Compute the barycenter of the density map and compare it to the
    # approximate barycenter
    barycenter = np.mean(np.argwhere(data), axis=0)
    if dataset == "small_affine_freesurfer":
        approx_barycenter = [141, 100, 82] if hemisphere == "lh" else [80, 101, 83]
    elif dataset == "big_affine_freesurfer":
        approx_barycenter = [139, 96, 80] if hemisphere == "lh" else [79, 97, 78]

    npt.assert_(np.linalg.norm(barycenter - approx_barycenter) < 2.0)
