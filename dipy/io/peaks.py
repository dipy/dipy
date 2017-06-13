from __future__ import division, print_function, absolute_import

import os
import numpy as np

from dipy.core.sphere import Sphere
from dipy.direction.peaks import (PeaksAndMetrics,
                                  reshape_peaks_for_visualization)
from distutils.version import LooseVersion

# Conditional import machinery for pytables
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests, if we don't have pytables
tables, have_tables, _ = optional_package('tables')

# Useful variable for backward compatibility.
if have_tables:
    TABLES_LESS_3_0 = LooseVersion(tables.__version__) < "3.0"

from dipy.data import get_sphere
from dipy.core.sphere import Sphere
from dipy.io.image import save_nifti


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
    """ Load a PeaksAndMetrics HDF5 file (PAM5)

    Parameters
    ----------
    fname : string
        Filename of PAM5 file.
    verbose : bool
        Print summary information about the loaded file.

    Returns
    -------
    pam : PeaksAndMetrics object
    """

    if os.path.splitext(fname)[1] != '.pam5':
        raise IOError('This function supports only PAM5 (HDF5) files')

    if TABLES_LESS_3_0:
        func_open_file = tables.openFile
    else:
        func_open_file = tables.open_file

    f = func_open_file(fname, 'r')

    pam = PeaksAndMetrics()

    pamh = f.root.pam

    version = f.root.version[0].decode()

    if version != '0.0.1':
        raise IOError('Incorrect PAM5 file version')

    try:
        affine = pamh.affine[:]
    except tables.NoSuchNodeError:
        affine = None

    peak_dirs = pamh.peak_dirs[:]
    peak_values = pamh.peak_values[:]
    peak_indices = pamh.peak_indices[:]

    try:
        shm_coeff = pamh.shm_coeff[:]
    except tables.NoSuchNodeError:
        shm_coeff = None

    sphere_vertices = pamh.sphere_vertices[:]

    try:
        odf = pamh.odf[:]
    except tables.NoSuchNodeError:
        odf = None

    pam.affine = affine
    pam.peak_dirs = peak_dirs
    pam.peak_values = peak_values
    pam.peak_indices = peak_indices
    pam.shm_coeff = shm_coeff
    pam.sphere = Sphere(xyz=sphere_vertices)
    pam.B = pamh.B[:]
    pam.total_weight = pamh.total_weight[:][0]
    pam.ang_thr = pamh.ang_thr[:][0]
    pam.gfa = pamh.gfa[:]
    pam.qa = pamh.qa[:]
    pam.odf = odf

    f.close()

    if verbose:
        print('PAM5 version')
        print(version)
        print('Affine')
        print(pam.affine)
        print('Dirs shape')
        print(pam.peak_dirs.shape)
        print('SH shape')
        if pam.shm_coeff is not None:
            print(pam.shm_coeff.shape)
        else:
            print('None')
        print('ODF shape')
        if pam.odf is not None:
            print(pam.odf.shape)
        else:
            print('None')
        print('Total weight')
        print(pam.total_weight)
        print('Angular threshold')
        print(pam.ang_thr)
        print('Sphere vertices shape')
        print(pam.sphere.vertices.shape)

    return pam


def save_peaks(fname, pam, affine=None, verbose=False):
    """ Save all important attributes of object PeaksAndMetrics in a PAM5 file
    (HDF5).

    Parameters
    ----------
    fname : string
        Filename of PAM5 file
    pam : PeaksAndMetrics
        Object holding peak_dirs, shm_coeffs and other attributes
    affine : array
        The 4x4 matrix transforming the date from native to world coordinates.
        PeaksAndMetrics should have that attribute but if not it can be
        provided here. Default None.
    verbose : bool
        Print summary information about the saved file.
    """

    if os.path.splitext(fname)[1] != '.pam5':
        raise IOError('This function saves only PAM5 (HDF5) files')

    if not (hasattr(pam, 'peak_dirs') and hasattr(pam, 'peak_values') and
            hasattr(pam, 'peak_indices')):

        msg = 'Cannot save object without peak_dirs, peak_values'
        msg += ' and peak_indices'
        raise ValueError(msg)

    if TABLES_LESS_3_0:
        func_open_file = tables.openFile
    else:
        func_open_file = tables.open_file

    f = func_open_file(fname, 'w')

    if TABLES_LESS_3_0:
        func_create_group = f.createGroup
        func_create_array = f.createArray
    else:
        func_create_group = f.create_group
        func_create_array = f.create_array

    group = func_create_group(f.root, 'pam')
    version = func_create_array(f.root, 'version',
                                [b"0.0.1"], 'PAM5 version number')
    version_string = f.root.version[0].decode()

    try:
        affine = pam.affine
    except AttributeError:
        pass

    try:
        shm_coeff = pam.shm_coeff
    except AttributeError:
        shm_coeff = None

    try:
        odf = pam.odf
    except AttributeError:
        odf = None

    _safe_save(f, group, affine, 'affine')
    _safe_save(f, group, pam.peak_dirs, 'peak_dirs')
    _safe_save(f, group, pam.peak_values, 'peak_values')
    _safe_save(f, group, pam.peak_indices, 'peak_indices')
    _safe_save(f, group, shm_coeff, 'shm_coeff')
    _safe_save(f, group, pam.sphere.vertices, 'sphere_vertices')
    _safe_save(f, group, pam.B, 'B')
    _safe_save(f, group, np.array([pam.total_weight]), 'total_weight')
    _safe_save(f, group, np.array([pam.ang_thr]), 'ang_thr')
    _safe_save(f, group, pam.gfa, 'gfa')
    _safe_save(f, group, pam.qa, 'qa')
    _safe_save(f, group, odf, 'odf')

    f.close()

    if verbose:
        print('PAM5 version')
        print(version_string)
        print('Affine')
        print(affine)
        print('Dirs shape')
        print(pam.peak_dirs.shape)
        print('SH shape')
        if shm_coeff is not None:
            print(shm_coeff.shape)
        else:
            print('None')
        print('ODF shape')
        if odf is not None:
            print(pam.odf.shape)
        else:
            print('None')
        print('Total weight')
        print(pam.total_weight)
        print('Angular threshold')
        print(pam.ang_thr)
        print('Sphere vertices shape')
        print(pam.sphere.vertices.shape)

    return pam


def peaks_to_niftis(pam,
                    fname_shm,
                    fname_dirs,
                    fname_values,
                    fname_indices,
                    fname_gfa,
                    reshape_dirs=False):
        """ Save SH, directions, indices and values of peaks to Nifti.
        """

        save_nifti(fname_shm, pam.shm_coeff.astype(np.float32), pam.affine)

        if reshape_dirs:
            pam_dirs = reshape_peaks_for_visualization(pam)
        else:
            pam_dirs = pam.peak_dirs.astype(np.float32)

        save_nifti(fname_dirs, pam_dirs, pam.affine)

        save_nifti(fname_values, pam.peak_values.astype(np.float32),
                   pam.affine)

        save_nifti(fname_indices, pam.peak_indices, pam.affine)

        save_nifti(fname_gfa, pam.gfa, pam.affine)

