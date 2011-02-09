""" Read test or example data
"""
import os
import cPickle
import gzip

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
    These sim voxels where provided by M.M. Correia using Rician noise.    
        
    """
    
    if name=='fib0':
        fname=os.path.join(os.path.dirname(__file__),'fib0.pkl.gz')
    if name=='fib1':
        fname=os.path.join(os.path.dirname(__file__),'fib1.pkl.gz')
    if name=='fib2':
        fname=os.path.join(os.path.dirname(__file__),'fib2.pkl.gz')
    
    return cPickle.loads(gzip.open(fname,'rb').read())
        
def get_skeleton(name='C1'):
    """ provide skeletons generated from Local Skeleton Clustering (LSC)
    
    Parameters
    -----------
    name : str, 'C1' or 'C3'
    
    Returns
    ---------    
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
        fname=os.path.join(os.path.dirname(__file__),'C1.pkl.gz')
    if name=='C3':
        fname=os.path.join(os.path.dirname(__file__),'C3.pkl.gz')        
    return cPickle.loads(gzip.open(fname,'rb').read())

def get_sphere(name='symmetric362'):    
    ''' provide triangulated spheres
    
    Parameters
    ------------
    name : str, which sphere
        'symmetric362' 
        'symmetric642'
    
    Examples
    ----------    
    
    >>> import numpy as np
    >>> from dipy.data import get_sphere
    >>> fname=get_sphere('symmetric362')
    >>> sph=np.load(fname)
    >>> verts=sph['vertices']
    >>> faces=sph['faces']
    >>> verts.shape
    (362, 3)
    >>> faces.shape
    (720, 3)
            
    '''
    
    if name=='symmetric362':
        return os.path.join(os.path.dirname(__file__),'evenly_distributed_sphere_362.npz')
    if name=='symmetric642':
        return os.path.join(os.path.dirname(__file__),'evenly_distributed_sphere_642.npz')


def get_data(name='small_64D'):
    ''' provides filenames of some test datasets

    Parameters
    ------------
    name: str
        the filename/s of which dataset to return, one of:
        'small_64D' small region of interest nifti,bvecs,bvals 64 directions
        'small_101D' small region of interest nifti,bvecs,bvals 101 directions
        'aniso_vox' volume with anisotropic voxel size as Nifti
        'fornix' 300 tracks in Trackvis format (from Pittsburgh Brain Competition)

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
    '''
    if name=='small_64D':
        fbvals=os.path.join(os.path.dirname(__file__),'small_64D.bvals.npy')
        fbvecs=os.path.join(os.path.dirname(__file__),'small_64D.gradients.npy')
        fimg =os.path.join(os.path.dirname(__file__),'small_64D.nii')        
        return fimg,fbvals, fbvecs
    if name=='55dir_grad.bvec':
        return os.path.join(os.path.dirname(__file__),'55dir_grad.bvec')
    if name=='small_101D':
        fbvals=os.path.join(os.path.dirname(__file__),'small_101D.bval')
        fbvecs=os.path.join(os.path.dirname(__file__),'small_101D.bvec')
        fimg=os.path.join(os.path.dirname(__file__),'small_101D.nii.gz')
        return fimg,fbvals, fbvecs
    if name=='aniso_vox':
        return os.path.join(os.path.dirname(__file__),'aniso_vox.nii.gz')
    if name=='fornix':
        return os.path.join(os.path.dirname(__file__),'tracks300.trk')

