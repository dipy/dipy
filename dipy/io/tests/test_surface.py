from dipy.io.stateful_surface import StatefulSurface
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.utils import Space, Origin
from dipy.io.streamline import load_tractogram, save_tractogram
from matplotlib.pyplot import get_cmap
import sys
from dipy.io.surface import save_surface, load_surface, load_pial
from scipy.spatial import KDTree
import os
import itertools

import nibabel as nib
import numpy as np
import logging
from dipy.utils.optpkg import optional_package
import numpy.testing as npt
import pytest
from tempfile import TemporaryDirectory

fury, have_fury, setup_module = optional_package("fury", min_version="0.8.0")

CWD = "/home/rhef1902/Datasets/stateful_surface/"


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_pial_load_save():
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))
    data_raw = nib.freesurfer.read_geometry(os.path.join("surf", 'lh.pial'))

    sfs = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz')

    # Change the space/origin, should not affect the saved pial
    sfs.to_vox()
    sfs.to_corner()

    with TemporaryDirectory() as tmpdir:
        save_surface(os.path.join(tmpdir, 'lh.pial'), sfs,
                     ref_pial=os.path.join("surf", 'lh.pial'))
        data_save = nib.freesurfer.read_geometry(
            os.path.join(tmpdir, 'lh.pial'))
    npt.assert_almost_equal(data_raw[0], data_save[0], decimal=5)


spaces = [Space.RASMM, Space.LPSMM, Space.VOXMM, Space.VOX]
origins = [Origin.NIFTI, Origin.TRACKVIS]


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
@pytest.mark.parametrize("space,origin", list(itertools.product(spaces, origins)))
def test_vtk_matching_space(space, origin):
    os.chdir(os.path.join(os.path.expanduser(CWD), 'mni_freesurfer'))

    sfs = load_surface(os.path.join("surf", 'lh.pial'), 'mni_masked.nii.gz')
    sfs.to_rasmm()
    sfs.to_center()
    ref_vertices = sfs.vertices.copy()

    with TemporaryDirectory() as tmpdir:
        save_surface(os.path.join(tmpdir, 'tmp.vtk'), sfs,
                     to_space=space, to_origin=origin)
        sfs = load_surface(os.path.join(tmpdir, 'tmp.vtk'), 'mni_masked.nii.gz',
                           from_space=space, from_origin=origin)

        sfs.to_rasmm()
        sfs.to_center()
        save_vertices = sfs.vertices.copy()
        npt.assert_almost_equal(ref_vertices, save_vertices, decimal=5)


folders = ["ascii", "base64", "gzip_base64"]
filenames = ["gifti.case1.pial.L.surf.gii", "gifti.case1.smoothwm.L.surf.gii"]


@pytest.mark.parametrize("type,fname,space,origin", list(itertools.product(folders, filenames, spaces, origins)))
def test_gifti_matching_space(type, fname, space, origin):
    os.chdir(os.path.join(os.path.expanduser(CWD), 'gifti'))
    if type == "gzip_base64":
        fname += ".gz"
    sfs = load_surface(os.path.join(type, fname), 'anat.nii.gz')
    sfs.to_rasmm()
    sfs.to_center()
    ref_vertices = sfs.vertices.copy()

    with TemporaryDirectory() as tmpdir:
        save_surface(os.path.join(tmpdir, 'tmp.gii'), sfs,
                     to_space=space, to_origin=origin)
        sfs = load_surface(os.path.join(tmpdir, 'tmp.gii'), 'anat.nii.gz',
                           from_space=space, from_origin=origin)

        sfs.to_rasmm()
        sfs.to_center()
        save_vertices = sfs.vertices.copy()
        npt.assert_almost_equal(ref_vertices, save_vertices, decimal=5)


folders = ["big_affine_freesurfer", "small_affine_freesurfer"]
hemispheres = ["lh", "rh"]
types = ["pial", "smoothwm", "orig"]


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
@pytest.mark.parametrize("dataset,hemisphere,type", list(itertools.product(folders, hemispheres, types)))
def test_freesurfer_density_operation(dataset, hemisphere, type):
    os.chdir(os.path.join(os.path.expanduser(CWD), dataset))
    fname = os.path.join("surf", f"{hemisphere}.{type}")
    sfs = load_surface(fname, 't1.nii.gz')

    assert sfs.is_bbox_in_vox_valid()

    data = np.zeros(sfs.dimensions, dtype=np.uint32)

    sfs.to_vox()
    sfs.to_corner()

    # Compute density map in the numpy grid
    for vertice in sfs.vertices:
        coord = tuple(vertice.astype(np.int32))
        data[coord] += 1

    with TemporaryDirectory() as tmpdir:
        nib.save(nib.Nifti1Image(data, sfs.affine),
                 os.path.join(tmpdir, f"{hemisphere}_{type}.nii.gz"))

    # Compute the barycenter of the density map and compare it to the approximate barycenter
    barycenter = np.mean(np.argwhere(data), axis=0)
    if dataset == "small_affine_freesurfer":
        approx_barycenter = [
            141, 100, 82] if hemisphere == "lh" else [80, 101, 83]
    elif dataset == "big_affine_freesurfer":
        approx_barycenter = [
            139, 96, 80] if hemisphere == "lh" else [79, 97, 78]

    print(barycenter, approx_barycenter)
    npt.assert_(np.linalg.norm(barycenter - approx_barycenter) < 2.0)
