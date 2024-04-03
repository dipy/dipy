import os
import numpy as np

from dipy.direction.peaks import (PeaksAndMetrics,
                                  reshape_peaks_for_visualization)
from dipy.core.sphere import Sphere
from dipy.io.image import save_nifti
import h5py


def _safe_save(group, array, name):
    """ Safe saving of arrays with specific names

    Parameters
    ----------
    group : HDF5 group
    array : array
    name : string
    """

    if array is not None:
        ds = group.create_dataset(name, shape=array.shape,
                                  dtype=array.dtype, chunks=True)
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

    if os.path.splitext(fname)[1].lower() != '.pam5':
        raise OSError('This function supports only PAM5 (HDF5) files')

    f = h5py.File(fname, 'r')

    pam = PeaksAndMetrics()

    pamh = f['pam']

    version = f.attrs['version']

    if version != '0.0.1':
        raise OSError('Incorrect PAM5 file version {0}'.format(version,))

    try:
        affine = pamh['affine'][:]
    except KeyError:
        affine = None

    peak_dirs = pamh['peak_dirs'][:]
    peak_values = pamh['peak_values'][:]
    peak_indices = pamh['peak_indices'][:]

    try:
        shm_coeff = pamh['shm_coeff'][:]
    except KeyError:
        shm_coeff = None

    sphere_vertices = pamh['sphere_vertices'][:]

    try:
        odf = pamh['odf'][:]
    except KeyError:
        odf = None

    pam.affine = affine
    pam.peak_dirs = peak_dirs
    pam.peak_values = peak_values
    pam.peak_indices = peak_indices
    pam.shm_coeff = shm_coeff
    pam.sphere = Sphere(xyz=sphere_vertices)
    pam.B = pamh['B'][:]
    pam.total_weight = pamh['total_weight'][:][0]
    pam.ang_thr = pamh['ang_thr'][:][0]
    pam.gfa = pamh['gfa'][:]
    pam.qa = pamh['qa'][:]
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
        raise OSError('This function saves only PAM5 (HDF5) files')

    if not (hasattr(pam, 'peak_dirs') and hasattr(pam, 'peak_values') and
            hasattr(pam, 'peak_indices')):

        msg = 'Cannot save object without peak_dirs, peak_values'
        msg += ' and peak_indices'
        raise ValueError(msg)

    f = h5py.File(fname, 'w')

    group = f.create_group('pam')
    f.attrs['version'] = '0.0.1'

    version_string = f.attrs['version']

    affine = pam.affine if hasattr(pam, 'affine') else affine
    shm_coeff = pam.shm_coeff if hasattr(pam, 'shm_coeff') else None
    odf = pam.odf if hasattr(pam, 'odf') else None

    _safe_save(group, affine, 'affine')
    _safe_save(group, pam.peak_dirs, 'peak_dirs')
    _safe_save(group, pam.peak_values, 'peak_values')
    _safe_save(group, pam.peak_indices, 'peak_indices')
    _safe_save(group, shm_coeff, 'shm_coeff')
    _safe_save(group, pam.sphere.vertices, 'sphere_vertices')
    _safe_save(group, pam.B, 'B')
    _safe_save(group, np.array([pam.total_weight]), 'total_weight')
    _safe_save(group, np.array([pam.ang_thr]), 'ang_thr')
    _safe_save(group, pam.gfa, 'gfa')
    _safe_save(group, pam.qa, 'qa')
    _safe_save(group, odf, 'odf')

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
                    fname_gfa=None,
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

        if fname_gfa is not None and hasattr(pam, 'gfa'):
            save_nifti(fname_gfa, pam.gfa, pam.affine)
