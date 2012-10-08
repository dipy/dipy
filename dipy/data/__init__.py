""" Read test or example data
"""
from nibabel import load
from dipy.io.bvectxt import read_bvec_file
from os.path import join as pjoin, dirname
import cPickle
import gzip
from dipy.core.gradients import gradient_table
from dipy.core.sphere import Sphere
import numpy as np

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
    if name=='fib0':
        fname=pjoin(THIS_DIR,'fib0.pkl.gz')
    if name=='fib1':
        fname=pjoin(THIS_DIR,'fib1.pkl.gz')
    if name=='fib2':
        fname=pjoin(THIS_DIR,'fib2.pkl.gz')
    return cPickle.loads(gzip.open(fname,'rb').read())


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
    >>> C[c].keys()
    ['indices', 'most', 'hidden', 'N']
    """
    if name=='C1':
        fname=pjoin(THIS_DIR,'C1.pkl.gz')
    if name=='C3':
        fname=pjoin(THIS_DIR,'C3.pkl.gz')
    return cPickle.loads(gzip.open(fname,'rb').read())


def get_sphere(name='symmetric362'):
    ''' provide triangulated spheres

    Parameters
    ------------
    name : str
        which sphere - one of:
        * 'symmetric362' 
        * 'symmetric642'

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
    >>> verts, faces = get_sphere('not a sphere name')
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
    return Sphere(xyz=as_native_array(res['vertices']))


def get_data(name='small_64D'):
    """ provides filenames of some test datasets or other useful parametrisations

    Parameters
    ----------
    name: str
        the filename/s of which dataset to return, one of:
        'small_64D' small region of interest nifti,bvecs,bvals 64 directions
        'small_101D' small region of interest nifti,bvecs,bvals 101 directions
        'aniso_vox' volume with anisotropic voxel size as Nifti
        'fornix' 300 tracks in Trackvis format (from Pittsburgh Brain Competition)
        'gqi_vectors' the scanner wave vectors needed for a GQI acquisitions of 101 directions tested on Siemens 3T Trio
        
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

    if name=='small_64D':
        fbvals=pjoin(THIS_DIR,'small_64D.bvals.npy')
        fbvecs=pjoin(THIS_DIR,'small_64D.gradients.npy')
        fimg =pjoin(THIS_DIR,'small_64D.nii')
        return fimg,fbvals, fbvecs
    if name=='55dir_grad.bvec':
        return pjoin(THIS_DIR,'55dir_grad.bvec')
    if name=='small_101D':
        fbvals=pjoin(THIS_DIR,'small_101D.bval')
        fbvecs=pjoin(THIS_DIR,'small_101D.bvec')
        fimg=pjoin(THIS_DIR,'small_101D.nii.gz')
        return fimg,fbvals, fbvecs
    if name=='aniso_vox':
        return pjoin(THIS_DIR,'aniso_vox.nii.gz')
    if name=='fornix':
        return pjoin(THIS_DIR,'tracks300.trk')
    if name=='gqi_vectors':
        return pjoin(THIS_DIR,'ScannerVectors_GQI101.txt')
    if name=='dsi515btable':
        return pjoin(THIS_DIR,'dsi515_b_table.txt')
    if name=='grad514':
        return pjoin(THIS_DIR,'grad_514.txt')

def dsi_voxels():
    fimg,fbvals,fbvecs = get_data('small_101D')
    bvals = np.loadtxt(fbvals)
    bvecs = np.loadtxt(fbvecs).T
    img = load(fimg)
    data = img.get_data()
    gtab = gradient_table(bvals, bvecs)
    return data, gtab
