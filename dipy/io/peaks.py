from __future__ import division, print_function, absolute_import

import numpy as np

from dipy.core.sphere import Sphere
from dipy.direction.peaks import (PeaksAndMetrics,
                                  reshape_peaks_for_visualization)
from distutils.version import LooseVersion
from dipy.reconst.peaks import PeaksAndMetrics

# Conditional testing machinery for pytables
from dipy.testing import doctest_skip_parser

# Conditional import machinery for pytables
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests, if we don't have pytables
tables, have_tables, _ = optional_package('tables')

# Useful variable for backward compatibility.
if have_tables:
    TABLES_LESS_3_0 = LooseVersion(tables.__version__) < "3.0"

from dipy.data import get_sphere
from dipy.core.sphere import Sphere


def _safe_save(f, group, array, name):
    """ Safe saving of arrays with specific names

    Parameters
    ----------
    f : HDF5 file handle
    group : HDF5 group
    array : array
    name : string
    """

    if TABLES_LESS_3_0:
        func_create_carray = f.createCArray
    else:
        func_create_carray = f.create_carray

    if array is not None:
        atom = tables.Atom.from_dtype(array.dtype)
        ds = func_create_carray(group, name, atom, array.shape)
        ds[:] = array


def load_peaks(fname, verbose=False):
    """ Load PeaksAndMetrics PAM5 file (HDF5)

    Parameters
    ----------
    fname : string
        Filename of PAM5 file.
    verbose : bool
        Print summary information about the loaded file.
    """

    if TABLES_LESS_3_0:
        func_open_file = tables.openFile
    else:
        func_open_file = tables.open_file

    f = func_open_file(fname, 'r')

    pam = PeaksAndMetrics()

    pamh = f.root.pam

    try:
        affine = pamh.affine[:]
    except tables.NoSuchNodeError:
        affine = None

    try:
        peak_dirs = pamh.peak_dirs[:]
    except tables.NoSuchNodeError:
        peak_dirs = None

    try:
        peak_values = pamh.peak_values[:]
    except tables.NoSuchNodeError:
        peak_values = None

    try:
        peak_indices = pamh.peak_indices[:]
    except tables.NoSuchNodeError:
        peak_indices = None

    try:
        shm_coeff = pamh.shm_coeff[:]
    except tables.NoSuchNodeError:
        shm_coeff = None

    try:
        sphere_vertices = pamh.sphere_vertices[:]
    except tables.NoSuchNodeError:
        sphere_vertices = None

    try:
        odf = pamh.odf[:]
    except tables.NoSuchNodeError:
        odf = None


    pam.affine = affine
    pam.peak_dirs = peak_dirs
    pam.peak_values = peak_values
    pam.peak_indices = peak_indices
    pam.shm_coeff = shm_coeff
    pam.sphere = Sphere(xyz=pamh.sphere_vertices)
    pam.B = pamh.B[:]
    pam.total_weight = pamh.total_weight[:][0]
    pam.ang_thr = pamh.ang_thr[:][0]
    pam.gfa = pamh.gfa[:]
    pam.qa = pamh.qa[:]
    pam.odf = odf

    f.close()

    if verbose:
        print('Affine')
        print(pam.affine)
        print('Dirs shape')
        print(pam.peak_dirs.shape)
        print('SH shape')
        print(pam.shm_coeff.shape)
        print('ODF shape')
        print(pam.odf.shape)
        print('Total weight')
        print(pam.total_weight)
        print('Angular threshold')
        print(pam.ang_thr)
        print('Sphere vertices shape')
        print(pam.sphere.vertices.shape)

    return pam


def save_peaks(fname, pam):
    """ Save PAM5 file (HDF5) with all important attributes of object
    PeaksAndMetrics

    Parameters
    ----------
    fname : string
        Filenam of PAM5 file
    pam : PeakAndMetrics
        Object holding peak_dirs, shm_coeffs and other attributes
    """

    if TABLES_LESS_3_0:
        func_open_file = tables.openFile
    else:
        func_open_file = tables.open_file

    f = func_open_file(fname, 'w')

    if TABLES_LESS_3_0:
        func_create_group = f.createGroup
    else:
        func_create_group = f.create_group

    group = func_create_group(f.root, 'pam')

    _safe_save(f, group, pam.affine, 'affine')
    _safe_save(f, group, pam.peak_dirs, 'peak_dirs')
    _safe_save(f, group, pam.peak_values, 'peak_values')
    _safe_save(f, group, pam.peak_indices, 'peak_indices')
    _safe_save(f, group, pam.shm_coeff, 'shm_coeff')
    _safe_save(f, group, pam.sphere.vertices, 'sphere_vertices')
    _safe_save(f, group, pam.B, 'B')
    _safe_save(f, group, np.array([pam.total_weight]), 'total_weight')
    _safe_save(f, group, np.array([pam.ang_thr]), 'ang_thr')
    _safe_save(f, group, pam.B, 'gfa')
    _safe_save(f, group, pam.B, 'qa')
    _safe_save(f, group, pam.B, 'odf')

    f.close()

