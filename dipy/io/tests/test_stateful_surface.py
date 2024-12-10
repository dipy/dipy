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
from numpy.testing import assert_, assert_allclose, assert_array_equal
import pytest
from tempfile import TemporaryDirectory

fury, have_fury, setup_module = optional_package("fury", min_version="0.8.0")

CWD = "/home/local/USHERBROOKE/rhef1902/Datasets/stateful_surface/"

@pytest.mark.skipif(not have_fury, reason="Requires FURY")
# Test if empty

# Test if save 100 time

# To space equivalent to to_rasmm()

# to origin equivalent to to_center()

# Test change origin from different space

# shift of change space 1000 time

# Test out of grid
@pytest.mark.skipif(not have_fury, reason="Requires FURY")
@pytest.mark.parametrize(
    "value, is_out_of_grid", [(1000, True), (-1000, True), (0, False)])
def test_out_of_grid(value, is_out_of_grid):
    os.chdir(os.path.expanduser(CWD))
    sfs = load_surface('lh.pial', 'mni_masked.nii.gz')
    sfs.to_vox()
    sfs.to_corner()

    tmp_vertices = sfs.vertices.copy()
    tmp_vertices[0] += value

    try:
        sfs.vertices = tmp_vertices
        assert_(sfs.is_bbox_in_vox_valid() != is_out_of_grid)
    except (TypeError, ValueError):
        assert_(False)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_invalid_empty():
    os.chdir(os.path.expanduser(CWD))
    sfs = StatefulSurface(([], []), 'mni_masked.nii.gz',
                          space=Space.RASMM, origin=Origin.NIFTI)
    sfs.to_vox()
    sfs.to_corner()

    try:
        sfs.is_bbox_in_vox_valid()
    except (TypeError, ValueError):
        assert_(True)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_equality():
    os.chdir(os.path.expanduser(CWD))
    sfs_1 = load_surface('lh.pial', 'mni_masked.nii.gz')
    sfs_2 = load_surface('lh.pial.ply', 'mni_masked.nii.gz',
                         from_space=Space.LPSMM, from_origin=Origin.NIFTI)
    sfs_1.to_rasmm()
    sfs_1.to_center()
    sfs_2.to_rasmm()
    sfs_2.to_center()

    assert_(sfs_1 == sfs_2)


