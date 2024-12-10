from dipy.io.stateful_surface import StatefulSurface
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.utils import Space, Origin
from dipy.io.streamline import load_tractogram, save_tractogram
from matplotlib.pyplot import get_cmap
import sys
from dipy.io.surface import save_surface, load_surface, load_pial
from scipy.spatial import KDTree
import os

import nibabel as nib
import numpy as np
import logging
from dipy.utils.optpkg import optional_package
import numpy.testing as npt
import pytest
from tempfile import TemporaryDirectory

fury, have_fury, setup_module = optional_package("fury", min_version="0.8.0")

CWD = "/home/local/USHERBROOKE/rhef1902/Datasets/stateful_surface/"

@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_pial_load_save():
    os.chdir(os.path.expanduser(CWD))
    data_raw = nib.freesurfer.read_geometry('lh.pial')

    sfs = load_surface('lh.pial', 'mni_masked.nii.gz')
    sfs.to_rasmm()
    sfs.to_center()

    with TemporaryDirectory() as tmpdir:
        save_surface(os.path.join(tmpdir, 'lh.pial'), sfs, ref_pial='lh.pial')
        data_save = nib.freesurfer.read_geometry(os.path.join(tmpdir, 'lh.pial'))
    npt.assert_almost_equal(data_raw[0], data_save[0], decimal=5)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
@pytest.mark.parametrize("space,origin", [(Space.RASMM, Origin.NIFTI),
                                            (Space.LPSMM, Origin.TRACKVIS),
                                            (Space.LPSMM, Origin.NIFTI),
                                            (Space.RASMM, Origin.TRACKVIS),
                                            (Space.VOXMM, Origin.TRACKVIS),
                                            (Space.VOX, Origin.TRACKVIS),
                                            (Space.VOXMM, Origin.NIFTI),
                                            (Space.VOX, Origin.NIFTI)])
                                            
def test_vtk_matching_space(space, origin):
    os.chdir(os.path.expanduser(CWD))

    sfs = load_surface('lh.pial', 'mni_masked.nii.gz')
    sfs.to_rasmm()
    sfs.to_center()
    ref_vertices = sfs.vertices.copy()

    with TemporaryDirectory() as tmpdir:
        save_surface(os.path.join(tmpdir, 'lh.vtk'), sfs,
                     to_space=space, to_origin=origin)
        sfs = load_surface(os.path.join(tmpdir, 'lh.vtk'), 'mni_masked.nii.gz',
                           from_space=space, from_origin=origin)

        sfs.to_rasmm()
        sfs.to_center()
        save_vertices = sfs.vertices.copy()
        npt.assert_almost_equal(ref_vertices, save_vertices, decimal=5)
