"""
Read test or example data
"""
from __future__ import division, print_function, absolute_import

import sys
import json
import warnings

from nibabel import load
from os.path import join as pjoin, dirname

import gzip
import numpy as np
from dipy.core.gradients import GradientTable, gradient_table
from dipy.core.sphere import Sphere, HemiSphere
from dipy.sims.voxel import SticksAndBall
from dipy.data.fetcher import (fetch_scil_b0,
                               read_scil_b0,
                               fetch_stanford_hardi,
                               read_stanford_hardi,
                               fetch_taiwan_ntu_dsi,
                               read_taiwan_ntu_dsi,
                               fetch_sherbrooke_3shell,
                               read_sherbrooke_3shell,
                               fetch_isbi2013_2shell,
                               read_isbi2013_2shell,
                               read_stanford_labels,
                               fetch_syn_data,
                               read_syn_data,
                               fetch_stanford_t1,
                               read_stanford_t1,
                               fetch_stanford_pve_maps,
                               read_stanford_pve_maps,
                               fetch_viz_icons,
                               read_viz_icons,
                               fetch_bundles_2_subjects,
                               read_bundles_2_subjects,
                               fetch_cenir_multib,
                               read_cenir_multib,
                               fetch_mni_template,
                               read_mni_template,
                               fetch_ivim,
                               read_ivim,
                               fetch_tissue_data,
                               read_tissue_data,
                               fetch_cfin_multib,
                               read_cfin_dwi,
                               read_cfin_t1)

from ..utils.arrfuncs import as_native_array
from dipy.tracking.streamline import relist_streamlines

if sys.version_info[0] < 3:
    import cPickle

    def loads_compat(bytes):
        return cPickle.loads(bytes)
else:  # Python 3
    import pickle
    # Need to load pickles saved in Python 2

    def loads_compat(bytes):
        return pickle.loads(bytes, encoding='latin1')


DATA_DIR = pjoin(dirname(__file__), 'files')
SPHERE_FILES = {
    'symmetric362': pjoin(DATA_DIR, 'evenly_distributed_sphere_362.npz'),
    'symmetric642': pjoin(DATA_DIR, 'evenly_distributed_sphere_642.npz'),
    'symmetric724': pjoin(DATA_DIR, 'evenly_distributed_sphere_724.npz'),
    'repulsion724': pjoin(DATA_DIR, 'repulsion724.npz'),
    'repulsion100': pjoin(DATA_DIR, 'repulsion100.npz'),
    'repulsion200': pjoin(DATA_DIR, 'repulsion200.npz')
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
    >>> sv['data'].shape == (100, 102)
    True
    >>> sv['fibres']
    '1'
    >>> sv['gradients'].shape == (102, 3)
    True
    >>> sv['bvals'].shape == (102,)
    True
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
        fname = pjoin(DATA_DIR, 'fib0.pkl.gz')
    if name == 'fib1':
        fname = pjoin(DATA_DIR, 'fib1.pkl.gz')
    if name == 'fib2':
        fname = pjoin(DATA_DIR, 'fib2.pkl.gz')
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
        fname = pjoin(DATA_DIR, 'C1.pkl.gz')
    if name == 'C3':
        fname = pjoin(DATA_DIR, 'C3.pkl.gz')
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
        * 'repulsion724'
        * 'repulsion100'
        * 'repulsion200'

    Returns
    -------
    sphere : a dipy.core.sphere.Sphere class instance

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.data import get_sphere
    >>> sphere = get_sphere('symmetric362')
    >>> verts, faces = sphere.vertices, sphere.faces
    >>> verts.shape == (362, 3)
    True
    >>> faces.shape == (720, 3)
    True
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


default_sphere = HemiSphere.from_sphere(get_sphere('symmetric724'))
small_sphere = HemiSphere.from_sphere(get_sphere('symmetric362'))


def get_data(name='small_64D'):
    """ provides filenames of some test datasets or other useful parametrisations

    Parameters
    ----------
    name : str
        the filename/s of which dataset to return, one of:
        'small_64D' small region of interest nifti,bvecs,bvals 64 directions
        'small_101D' small region of interest nifti,bvecs,bvals 101 directions
        'aniso_vox' volume with anisotropic voxel size as Nifti
        'fornix' 300 tracks in Trackvis format (from Pittsburgh
            Brain Competition)
        'gqi_vectors' the scanner wave vectors needed for a GQI acquisitions
            of 101 directions tested on Siemens 3T Trio
        'small_25' small ROI (10x8x2) DTI data (b value 2000, 25 directions)
        'test_piesno' slice of N=8, K=14 diffusion data
        'reg_c' small 2D image used for validating registration
        'reg_o' small 2D image used for validation registration
        'cb_2' two vectorized cingulum bundles

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
    >>> data.shape == (6, 10, 10, 102)
    True
    >>> bvals.shape == (102,)
    True
    >>> bvecs.shape == (102, 3)
    True
    """

    if name == 'small_64D':
        fbvals = pjoin(DATA_DIR, 'small_64D.bvals.npy')
        fbvecs = pjoin(DATA_DIR, 'small_64D.gradients.npy')
        fimg = pjoin(DATA_DIR, 'small_64D.nii')
        return fimg, fbvals, fbvecs
    if name == '55dir_grad.bvec':
        return pjoin(DATA_DIR, '55dir_grad.bvec')
    if name == 'small_101D':
        fbvals = pjoin(DATA_DIR, 'small_101D.bval')
        fbvecs = pjoin(DATA_DIR, 'small_101D.bvec')
        fimg = pjoin(DATA_DIR, 'small_101D.nii.gz')
        return fimg, fbvals, fbvecs
    if name == 'aniso_vox':
        return pjoin(DATA_DIR, 'aniso_vox.nii.gz')
    if name == 'ascm_test':
        return pjoin(DATA_DIR, 'ascm_out_test.nii.gz')
    if name == 'fornix':
        return pjoin(DATA_DIR, 'tracks300.trk')
    if name == 'gqi_vectors':
        return pjoin(DATA_DIR, 'ScannerVectors_GQI101.txt')
    if name == 'dsi515btable':
        return pjoin(DATA_DIR, 'dsi515_b_table.txt')
    if name == 'dsi4169btable':
        return pjoin(DATA_DIR, 'dsi4169_b_table.txt')
    if name == 'grad514':
        return pjoin(DATA_DIR, 'grad_514.txt')
    if name == "small_25":
        fbvals = pjoin(DATA_DIR, 'small_25.bval')
        fbvecs = pjoin(DATA_DIR, 'small_25.bvec')
        fimg = pjoin(DATA_DIR, 'small_25.nii.gz')
        return fimg, fbvals, fbvecs
    if name == "S0_10":
        fimg = pjoin(DATA_DIR, 'S0_10slices.nii.gz')
        return fimg
    if name == "test_piesno":
        fimg = pjoin(DATA_DIR, 'test_piesno.nii.gz')
        return fimg
    if name == "reg_c":
        return pjoin(DATA_DIR, 'C.npy')
    if name == "reg_o":
        return pjoin(DATA_DIR, 'circle.npy')
    if name == 'cb_2':
        return pjoin(DATA_DIR, 'cb_2.npz')
    if name == "t1_coronal_slice":
        return pjoin(DATA_DIR, 't1_coronal_slice.npy')


def _gradient_from_file(filename):
    """Reads a gradient file saved as a text file compatible with np.loadtxt
    and saved in the dipy data directory"""
    def gtab_getter():
        gradfile = pjoin(DATA_DIR, filename)
        grad = np.loadtxt(gradfile, delimiter=',')
        gtab = GradientTable(grad)
        return gtab
    return gtab_getter


get_3shell_gtab = _gradient_from_file("gtab_3shell.txt")
get_isbi2013_2shell_gtab = _gradient_from_file("gtab_isbi2013_2shell.txt")
get_gtab_taiwan_dsi = _gradient_from_file("gtab_taiwan_dsi.txt")


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
                                                       S0=1.,
                                                       angles=[(0, 0),
                                                               (90, 0)],
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
    func_discrete = load(pjoin(DATA_DIR, "func_discrete.nii.gz")).get_data()
    func_coef = load(pjoin(DATA_DIR, "func_coef.nii.gz")).get_data()
    gradients = np.loadtxt(pjoin(DATA_DIR, "sphere_grad.txt"))
    # gradients[0] and the first volume of func_discrete,
    # func_discrete[..., 0], are associated with the b=0 signal.
    # gradients[:, 3] are the b-values for each gradient/volume.
    sphere = Sphere(xyz=gradients[1:, :3])
    return func_coef, func_discrete[..., 1:], sphere


dipy_cmaps = None


def get_cmap(name):
    """Makes a callable, similar to maptlotlib.pyplot.get_cmap"""
    if name.lower() == "accent":
        warnings.warn("The `Accent` colormap is deprecated as of version" +
                      " 0.12 of Dipy and will be removed in a future " +
                      "version. Please use another colormap",
                      DeprecationWarning)
    global dipy_cmaps
    if dipy_cmaps is None:
        filename = pjoin(DATA_DIR, "dipy_colormaps.json")
        with open(filename) as f:
            dipy_cmaps = json.load(f)

    desc = dipy_cmaps.get(name)
    if desc is None:
        return None

    def simple_cmap(v):
        """Emulates matplotlib colormap callable"""
        rgba = np.ones((len(v), 4))
        for i, color in enumerate(('red', 'green', 'blue')):
            x, y0, y1 = zip(*desc[color])
            # Matplotlib allows more complex colormaps, but for users who do
            # not have Matplotlib dipy makes a few simple colormaps available.
            # These colormaps are simple because y0 == y1, and therefor we
            # ignore y1 here.
            rgba[:, i] = np.interp(v, x, y0)
        return rgba

    return simple_cmap


def two_cingulum_bundles():
    fname = get_data('cb_2')
    res = np.load(fname)
    cb1 = relist_streamlines(res['points'], res['offsets'])
    cb2 = relist_streamlines(res['points2'], res['offsets2'])
    return cb1, cb2


def matlab_life_results():
    matlab_rmse = np.load(pjoin(DATA_DIR, 'life_matlab_rmse.npy'))
    matlab_weights = np.load(pjoin(DATA_DIR, 'life_matlab_weights.npy'))
    return matlab_rmse, matlab_weights
