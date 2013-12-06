"""
Read test or example data
"""

from __future__ import division, print_function, absolute_import


import sys

from nibabel import load
from dipy.io.bvectxt import read_bvec_file
from os.path import join as pjoin, dirname

if sys.version_info[0] < 3:
    import cPickle

    def loads_compat(bytes):
        return cPickle.loads(bytes)
else:  # Python 3
    import pickle
    # Need to load pickles saved in Python 2

    def loads_compat(bytes):
        return pickle.loads(bytes, encoding='latin1')

import gzip
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.core.sphere import Sphere
from dipy.sims.voxel import SticksAndBall
import numpy as np
from dipy.data.fetcher import (fetch_scil_b0,
                               read_scil_b0,
                               fetch_stanford_hardi,
                               read_stanford_hardi,
                               fetch_taiwan_ntu_dsi,
                               read_taiwan_ntu_dsi,
                               fetch_sherbrooke_3shell,
                               read_sherbrooke_3shell,
                               fetch_isbi2013_2shell,
                               read_isbi2013_2shell)

from ..utils.arrfuncs import as_native_array

THIS_DIR = dirname(__file__)
SPHERE_FILES = {
    'symmetric362': pjoin(THIS_DIR, 'evenly_distributed_sphere_362.npz'),
    'symmetric642': pjoin(THIS_DIR, 'evenly_distributed_sphere_642.npz'),
    'symmetric724': pjoin(THIS_DIR, 'evenly_distributed_sphere_724.npz')
}


class DataError(Exception):
    pass


def get_sim_voxels(name='fib1'):
    """ provide some simulated voxel data

    Parameters
    ------------
    name : str, which file?
        'fib0', 'fib1' or 'fib2'

    Returns
    ---------
    dix : dictionary, where dix['data'] returns a 2d array
        where every row is a simulated voxel with different orientation

    Examples
    ----------
    >>> from dipy.data import get_sim_voxels
    >>> sv=get_sim_voxels('fib1')
    >>> sv['data'].shape
    (100, 102)
    >>> sv['fibres']
    '1'
    >>> sv['gradients'].shape
    (102, 3)
    >>> sv['bvals'].shape
    (102,)
    >>> sv['snr']
    '60'
    >>> sv2=get_sim_voxels('fib2')
    >>> sv2['fibres']
    '2'
    >>> sv2['snr']
    '80'

    Notes
    -------
    These sim voxels were provided by M.M. Correia using Rician noise.
    """
    if name == 'fib0':
        fname = pjoin(THIS_DIR, 'fib0.pkl.gz')
    if name == 'fib1':
        fname = pjoin(THIS_DIR, 'fib1.pkl.gz')
    if name == 'fib2':
        fname = pjoin(THIS_DIR, 'fib2.pkl.gz')
    return loads_compat(gzip.open(fname, 'rb').read())


def get_skeleton(name='C1'):
    """ provide skeletons generated from Local Skeleton Clustering (LSC)

    Parameters
    -----------
    name : str, 'C1' or 'C3'

    Returns
    -------
    dix : dictionary

    Examples
    ---------
    >>> from dipy.data import get_skeleton
    >>> C=get_skeleton('C1')
    >>> len(C.keys())
    117
    >>> for c in C: break
    >>> sorted(C[c].keys())
    ['N', 'hidden', 'indices', 'most']
    """
    if name == 'C1':
        fname = pjoin(THIS_DIR, 'C1.pkl.gz')
    if name == 'C3':
        fname = pjoin(THIS_DIR, 'C3.pkl.gz')
    return loads_compat(gzip.open(fname, 'rb').read())


def get_sphere(name='symmetric362'):
    ''' provide triangulated spheres

    Parameters
    ------------
    name : str
        which sphere - one of:
        * 'symmetric362'
        * 'symmetric642'
        * 'symmetric724'

    Returns
    -------
    sphere : a dipy.core.sphere.Sphere class instance

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.data import get_sphere
    >>> sphere = get_sphere('symmetric362')
    >>> verts, faces = sphere.vertices, sphere.faces
    >>> verts.shape
    (362, 3)
    >>> faces.shape
    (720, 3)
    >>> verts, faces = get_sphere('not a sphere name') #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    DataError: No sphere called "not a sphere name"
    '''
    fname = SPHERE_FILES.get(name)
    if fname is None:
        raise DataError('No sphere called "%s"' % name)
    res = np.load(fname)
    # Set to native byte order to avoid errors in compiled routines for
    # big-endian platforms, when using these spheres.
    return Sphere(xyz=as_native_array(res['vertices']),
                  faces=as_native_array(res['faces']))


def get_data(name='small_64D'):
    """ provides filenames of some test datasets or other useful parametrisations

    Parameters
    ----------
    name : str
        the filename/s of which dataset to return, one of:
        'small_64D' small region of interest nifti,bvecs,bvals 64 directions
        'small_101D' small region of interest nifti,bvecs,bvals 101 directions
        'aniso_vox' volume with anisotropic voxel size as Nifti
        'fornix' 300 tracks in Trackvis format (from Pittsburgh Brain Competition)
        'gqi_vectors' the scanner wave vectors needed for a GQI acquisitions of 101 directions tested on Siemens 3T Trio
        'small_25' small ROI (10x8x2) DTI data (b value 2000, 25 directions)

    Returns
    -------
    fnames : tuple
        filenames for dataset

    Examples
    ----------
    >>> import numpy as np
    >>> from dipy.data import get_data
    >>> fimg,fbvals,fbvecs=get_data('small_101D')
    >>> bvals=np.loadtxt(fbvals)
    >>> bvecs=np.loadtxt(fbvecs).T
    >>> import nibabel as nib
    >>> img=nib.load(fimg)
    >>> data=img.get_data()
    >>> data.shape
    (6, 10, 10, 102)
    >>> bvals.shape
    (102,)
    >>> bvecs.shape
    (102, 3)
    """

    if name == 'small_64D':
        fbvals = pjoin(THIS_DIR, 'small_64D.bvals.npy')
        fbvecs = pjoin(THIS_DIR, 'small_64D.gradients.npy')
        fimg = pjoin(THIS_DIR, 'small_64D.nii')
        return fimg, fbvals, fbvecs
    if name == '55dir_grad.bvec':
        return pjoin(THIS_DIR, '55dir_grad.bvec')
    if name == 'small_101D':
        fbvals = pjoin(THIS_DIR, 'small_101D.bval')
        fbvecs = pjoin(THIS_DIR, 'small_101D.bvec')
        fimg = pjoin(THIS_DIR, 'small_101D.nii.gz')
        return fimg, fbvals, fbvecs
    if name == 'aniso_vox':
        return pjoin(THIS_DIR, 'aniso_vox.nii.gz')
    if name == 'fornix':
        return pjoin(THIS_DIR, 'tracks300.trk')
    if name == 'gqi_vectors':
        return pjoin(THIS_DIR, 'ScannerVectors_GQI101.txt')
    if name == 'dsi515btable':
        return pjoin(THIS_DIR, 'dsi515_b_table.txt')
    if name == 'dsi4169btable':
        return pjoin(THIS_DIR, 'dsi4169_b_table.txt')
    if name == 'grad514':
        return pjoin(THIS_DIR, 'grad_514.txt')
    if name == "small_25":
        fbvals = pjoin(THIS_DIR, 'small_25.bval')
        fbvecs = pjoin(THIS_DIR, 'small_25.bvec')
        fimg = pjoin(THIS_DIR, 'small_25.nii.gz')
        return fimg, fbvals, fbvecs
    if name == "S0_10":
        fimg = pjoin(THIS_DIR, 'S0_10slices.nii.gz')
        return fimg
    if name == 'ISBI_testing_2shells_table':
        fbvals = pjoin(THIS_DIR, '2shells-1500-2500-N64.bval')
        fbvecs = pjoin(THIS_DIR, '2shells-1500-2500-N64.bvec')
        fimg = pjoin(THIS_DIR, 'MS-SNR-30.nii.gz')
        return fimg, fbvals, fbvecs
    if name == '3shells_data':
        fbvals = pjoin(THIS_DIR, '3shells-1000-2000-3500-N193.bval')
        fbvecs = pjoin(THIS_DIR, '3shells-1000-2000-3500-N193.bvec')
        fimg = pjoin(THIS_DIR, '3shells-1000-2000-3500-N193.nii.gz')
        return fimg, fbvals, fbvecs

def dsi_voxels():
    fimg, fbvals, fbvecs = get_data('small_101D')
    bvals = np.loadtxt(fbvals)
    bvecs = np.loadtxt(fbvecs).T
    img = load(fimg)
    data = img.get_data()
    gtab = gradient_table(bvals, bvecs)
    return data, gtab


def dsi_deconv_voxels():
    gtab = gradient_table(np.loadtxt(get_data('dsi515btable')))
    data = np.zeros((2, 2, 2, 515))
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                data[ix, iy, iz], dirs = SticksAndBall(gtab,
                                                       d=0.0015,
                                                       S0=100,
                                                       angles=[(0, 0), (90, 0)],
                                                       fractions=[50, 50],
                                                       snr=None)
    return data, gtab


def mrtrix_spherical_functions():
    """Spherical functions represented by spherical harmonic coefficients and
    evaluated on a discrete sphere.

    Returns
    -------
    func_coef : array (2, 3, 4, 45)
        Functions represented by the coefficients associated with the
        mxtrix spherical harmonic basis of order 8.
    func_discrete : array (2, 3, 4, 81)
        Functions evaluated on `sphere`.
    sphere : Sphere
        The discrete sphere, points on the surface of a unit sphere, used to
        evaluate the functions.

    Notes
    -----
    These coefficients were obtained by using the dwi2SH command of mrtrix.

    """
    func_discrete = load(pjoin(THIS_DIR, "func_discrete.nii.gz")).get_data()
    func_coef = load(pjoin(THIS_DIR, "func_coef.nii.gz")).get_data()
    gradients = np.loadtxt(pjoin(THIS_DIR, "sphere_grad.txt"))
    # gradients[0] and the first volume of func_discrete,
    # func_discrete[..., 0], are associated with the b=0 signal.
    # gradients[:, 3] are the b-values for each gradient/volume.
    sphere = Sphere(xyz=gradients[1:, :3])
    return func_coef, func_discrete[..., 1:], sphere


def two_shells_voxels(xmin,xmax,ymin,ymax,zmin,zmax):
    fimg, fbvals, fbvecs = get_data('ISBI_testing_2shells_table')
    bvals = np.loadtxt(fbvals)
    bvecs = np.loadtxt(fbvecs).T
    gtab = gradient_table(bvals[1:], bvecs[1:,:])
    img = load(fimg)
    data = img.get_data()
    b0 = data[:,:,:,0]
    data = data[xmin:xmax,ymin:ymax,zmin:zmax,1:]/b0[xmin:xmax,ymin:ymax,zmin:zmax,None]
    affine = img.get_affine()
    return data, affine, gtab


def three_shells_voxels(xmin,xmax,ymin,ymax,zmin,zmax):
    fimg, fbvals, fbvecs = get_data('3shells_data')
    bvals = np.loadtxt(fbvals)
    bvecs = np.loadtxt(fbvecs).T
    bvecs[:,0] = -bvecs[:,0]
    bvecs[:,1] = bvecs[:,1]
    bvecs[:,2] = bvecs[:,2]
    gtab = gradient_table(bvals[1:], bvecs[1:,:])
    img = load(fimg)
    data = np.double(img.get_data())
    b0 = np.double(data[:,:,:,0])
    data = data[xmin:xmax,ymin:ymax,zmin:zmax,1:]/b0[xmin:xmax,ymin:ymax,zmin:zmax,None]
    affine = img.get_affine()
    return data, affine, gtab
