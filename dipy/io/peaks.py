
import os
import numpy as np

from dipy.direction.peaks import (PeaksAndMetrics,
                                  reshape_peaks_for_visualization)
from dipy.core.sphere import Sphere
from dipy.io.image import save_nifti
from dipy.reconst.dti import quantize_evecs
import h5py


def _safe_save(group, array, name):
    """Safe saving of arrays with specific names.

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
    """Load a PeaksAndMetrics HDF5 file (PAM5).

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
    # TODO: Deprecate
    return load_pam(fname=fname, verbose=verbose)


def load_pam(fname, verbose=False):
    """Load a PeaksAndMetrics HDF5 file (PAM5).

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
        raise IOError('This function supports only PAM5 (HDF5) files')

    f = h5py.File(fname, 'r')

    pam = PeaksAndMetrics()

    pamh = f['pam']

    version = f.attrs['version']

    if version != '0.0.1':
        raise IOError('Incorrect PAM5 file version {0}'.format(version,))

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
    pam.B = pamh['B'][:] if 'B' in pamh else None
    pam.total_weight = pamh['total_weight'][:][0] if 'total_weight' in pamh \
        else None
    pam.ang_thr = pamh['ang_thr'][:][0] if 'ang_thr' in pamh else None
    pam.gfa = pamh['gfa'][:] if 'gfa' in pamh else None
    pam.qa = pamh['qa'][:] if 'qa' in pamh else None
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
    """Save all important attributes of object PeaksAndMetrics in a PAM5 file
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
    # Todo, ADD deprecation warning
    return save_pam(fname=fname, pam=pam, affine=affine, verbose=verbose)


def save_pam(fname, pam, affine=None, verbose=False):
    """Save all important attributes of object PeaksAndMetrics in a PAM5 file
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

    f = h5py.File(fname, 'w')

    group = f.create_group('pam')
    f.attrs['version'] = u'0.0.1'

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


def peaks_to_niftis(pam, fname_shm, fname_dirs, fname_values, fname_indices,
                    fname_gfa, reshape_dirs=False):
    """Save SH, directions, indices and values of peaks to Nifti.

    Parameters
    ----------
    pam : PeaksAndMetrics
        Object holding peak_dirs, shm_coeffs and other attributes
    fname_shm : str
        Spherical Harmonics coefficients filename
    fname_dirs : str
        peaks direction filename
    fname_values : str
        peaks values filename
    fname_indices : str
        peaks indices filename
    fname_gfa : str
        Generalized FA filename
    reshape_dirs : bool, optional
        If True, reshape peaks for visualization
        (default False)

    """
    # Todo, ADD deprecation warning
    save_nifti(fname_shm, pam.shm_coeff.astype(np.float32), pam.affine)

    if reshape_dirs:
        pam_dirs = reshape_peaks_for_visualization(pam)
    else:
        pam_dirs = pam.peak_dirs.astype(np.float32)

    save_nifti(fname_dirs, pam_dirs, pam.affine)
    save_nifti(fname_values, pam.peak_values.astype(np.float32), pam.affine)
    save_nifti(fname_indices, pam.peak_indices, pam.affine)
    save_nifti(fname_gfa, pam.gfa, pam.affine)


def pam_to_niftis(pam, prefix_fname, reshape_dirs=False):
    """Save SH, directions, indices and values of peaks to Nifti.

    Parameters
    ----------
    pam : PeaksAndMetrics
        Object holding peak_dirs, shm_coeffs and other attributes
    prefix_fname : str
        prefix that will be added to all filenames
    reshape_dirs : bool, optional
        If True, Reshape and convert to float32 a set of peaks for
        visualisation with mrtrix or the fibernavigator.
        (default False)

    """
    if reshape_dirs:
        pam_dirs = reshape_peaks_for_visualization(pam)
    else:
        pam_dirs = pam.peak_dirs.astype(np.float32)

    save_nifti(prefix_fname + '_peaks_dirs.nii.gz', pam_dirs, pam.affine)
    save_nifti(prefix_fname + '_peaks_values.nii.gz',
               pam.peak_values.astype(np.float32), pam.affine)
    save_nifti(prefix_fname + '_peaks_indices.nii.gz', pam.peak_indices,
               pam.affine)
    if hasattr(pam, 'shm_coeff'):
        save_nifti(prefix_fname + '_shm.nii.gz',
                   pam.shm_coeff.astype(np.float32), pam.affine)
    if hasattr(pam, 'gfa'):
        save_nifti(prefix_fname + '_gfa.nii.gz', pam.gfa, pam.affine)
    if hasattr(pam, 'sphere'):
        np.savetxt(prefix_fname + '_sphere.txt', pam.sphere.vertices)
    if hasattr(pam, 'B'):
        save_nifti(prefix_fname + '_B.nii.gz', pam.B, pam.affine)
    if hasattr(pam, 'qa'):
        save_nifti(prefix_fname + '_qa.nii.gz', pam.qa, pam.affine)
    if hasattr(pam, 'odf'):
        save_nifti(prefix_fname + '_odf.nii.gz', pam.odf, pam.affine)

    # Float and int value, maybe on int headers
    # save_nifti(prefix_fname, pam.total_weight), pam.affine)
    # save_nifti(prefix_fname, pam.ang_thr), pam.affine)


def niftis_to_pam(affine, peak_dirs, peak_values, peak_indices,
                  shm_coeff=None, sphere=None, gfa=None, B=None,
                  qa=None, odf=None, total_weight=None, ang_thr=None,
                  pam_file=None):
    """Return SH, directions, indices and values of peaks to pam5.

    Parameters
    ----------
    affine : array, (4, 4)
        the matrix defining the affine transform
    peak_dirs : ndarray
        The direction of each peak.
    peak_values : ndarray
        The value of the peaks.
    peak_indices : ndarray
        Indices (in sphere vertices) of the peaks in each voxel.
    shm_coeff : array, optional
        Spherical harmonics coefficients
    sphere : `Sphere` class instance, optional
        The Sphere providing discrete directions for evaluation.
    gfa : ndarray, optional
        generalized FA volume
    B : ndarray, optional
        Matrix that transforms spherical harmonics to spherical function
    qa : array, optional
        Quantitative Anisotropy in each voxel.
    odf : ndarray, optional
        SH coefficients for the ODF spherical function
    total_weight : float, optional
    ang_thr : float, optional
    pam_file : str, optional
        Filename of the desired pam file

    Returns
    -------
    pam : PeaksAndMetrics
        Object holding peak_dirs, shm_coeffs and other attributes

    """
    pam = PeaksAndMetrics()
    pam.affine = affine
    pam.peak_dirs = peak_dirs
    pam.peak_values = peak_values
    pam.peak_indices = peak_indices

    for name, value in [('shm_coeff', shm_coeff), ('sphere', sphere), ('B', B),
                        ('total_weight', total_weight), ('ang_thr', ang_thr),
                        ('gfa', gfa), ('qa', qa), ('odf', odf)]:
        if value is not None:
            setattr(pam, name, value)

    if pam_file:
        save_pam(pam_file, pam)
    return pam


def tensor_to_pam(evals, evecs, affine, shm_coeff=None, sphere=None, gfa=None,
                  B=None, qa=None, odf=None, total_weight=None, ang_thr=None,
                  pam_file=None, npeaks=5, generate_peaks_indices=True):
    """Convert diffusion tensor to pam5.

    Parameters
    ----------
    evals : ndarray
        Eigenvalues of a diffusion tensor. shape should be (...,3).
    evecs : ndarray
        Eigen vectors from the tensor model
    affine : array, (4, 4)
        the matrix defining the affine transform
    shm_coeff : array, optional
        Spherical harmonics coefficients
    sphere : `Sphere` class instance, optional
        The Sphere providing discrete directions for evaluation.
    gfa : ndarray, optional
        generalized FA volume
    B : ndarray, optional
        Matrix that transforms spherical harmonics to spherical function
    qa : array, optional
        Quantitative Anisotropy in each voxel.
    odf : ndarray, optional
        SH coefficients for the ODF spherical function
    pam_file : str, optional
        Filename of the desired pam file
    npeaks : int
        Maximum number of peaks found (default 5 peaks).
    generate_peaks_indices : bool, optional
    total_weight : float, optional
    ang_thr : float, optional

    Returns
    -------
    pam : PeaksAndMetrics
        Object holding peak_dirs, shm_coeffs and other attributes

    """
    shape = evals.shape[:3]
    peaks_dirs = np.zeros((shape + (npeaks, 3)))
    peaks_dirs[..., :3, :] = evecs
    peaks_values = np.zeros((shape + (npeaks,)))
    peaks_values[..., :3] = evals

    if generate_peaks_indices:
        vertices = sphere.vertices if sphere else None
        peaks_indices = quantize_evecs(evecs, vertices)
    else:
        peaks_indices = np.zeros((shape + (npeaks,)), dtype='int')
        peaks_indices.fill(-1)

    return niftis_to_pam(affine=affine, peak_dirs=peaks_dirs,
                         peak_values=peaks_values, peak_indices=peaks_indices,
                         shm_coeff=shm_coeff, sphere=sphere, gfa=gfa, B=B,
                         qa=qa, odf=odf, total_weight=total_weight,
                         ang_thr=ang_thr, pam_file=pam_file)
