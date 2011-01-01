''' FSL IO '''

import os
from os.path import join as pjoin
import numpy as np
import nibabel as nib
import numpy.linalg as npl
from scipy.ndimage import map_coordinates as mc
from numpy import newaxis

_VAL_FMT = '   %e'

def write_bvals_bvecs(bvals, bvecs, outpath=None, prefix=''):
    ''' Write FSL FDT bvals and bvecs files

    Parameters
    -------------
    bvals : (N,) sequence
       Vector with diffusion gradient strength (one per diffusion
       acquisition, N=no of acquisitions)
    bvecs : (N, 3) array-like
       diffusion gradient directions
    outpath : None or str
       path to write FDT bvals, bvecs text files
       None results in current working directory.
    prefix : str
       prefix for bvals, bvecs files in directory.  Defaults to ''
    '''
    if outpath is None:
        outpath = os.getcwd()
    bvals = tuple(bvals)
    bvecs = np.asarray(bvecs)
    bvecs[np.isnan(bvecs)] = 0
    N = len(bvals)
    fname = pjoin(outpath, prefix + 'bvals')
    fmt = _VAL_FMT * N + '\n'
    open(fname, 'wt').write(fmt % bvals)
    fname = pjoin(outpath, prefix + 'bvecs')
    bvf = open(fname, 'wt')
    for dim_vals in bvecs.T:
        bvf.write(fmt % tuple(dim_vals))


def flirt2aff(mat, in_img, ref_img):
    """ Transform from `in_img` voxels to `ref_img` voxels given `matfile`

    Parameters
    ------------
    matfile : (4,4) array
        contents (as array) of output ``-omat`` transformation file from flirt
    in_img : img
        image passed (as filename) to flirt as ``-in`` image
    ref_img : img
        image passed (as filename) to flirt as ``-ref`` image

    Returns
    ---------
    aff : (4,4) array
        Transform from voxel coordinates in ``in_img`` to voxel coordinates in
        ``ref_img``
    """
    in_hdr = in_img.get_header()
    ref_hdr = ref_img.get_header()
    # get_zooms gets the positive voxel sizes as returned in the header
    in_zoomer = np.diag(in_hdr.get_zooms() + (1,))
    ref_zoomer = np.diag(ref_hdr.get_zooms() + (1,))
    
    if npl.det(in_img.get_affine())>=0:        
        print('positive determinant in in')
        print('swaping is needed i_s=Nx-1-i_o')
        print('which is not implemented yet')
    if npl.det(ref_img.get_affine())>=0:        
        print('positive determinant in ref')
        print('swapping is needed i_s=Nx-1-i_o')
        print('which is not implemented yet')
        
    ''' Notes from correspondence with Mark Jenkinson
    There is also the issue for FSL matrices of the handedness of the
    coordinate system.  If the nifti sform/qform has negative determinant
    for both input and reference images then what has been said is true.
    If there is a positive determinant then the mapping between voxel
    and world coordinates is complicated by the fact that we swap the
    "x" voxel coordinate (that is, coordinate "i" in Jesper's reply).  That is,
    i_swapped = Nx - 1 - i_orig, where i_swapped and i_orig are the voxel
    coordinates in the "x" direction and Nx is the number of voxels in this
    direction.  Note that there may be a swap for the input image, the
    reference image, or both - whichever has a positive determinant for
    the sform/qform needs to be swapped.  Also, if you are used to
    MATLAB, note that all of the voxel coordinates start at 0, not 1.
    
    '''
    # The in_img voxels to ref_img voxels as recorded in the current affines
    current_in2ref = np.dot(ref_img.get_affine(), in_img.get_affine())
    if npl.det(current_in2ref) < 0:
        raise ValueError('Negative determinant to current affine mapping - bailing out')
    return np.dot(npl.inv(ref_zoomer), np.dot(mat, in_zoomer))


def flirt2aff_files(matfile, in_fname, ref_fname):
    """ Map from `in_fname` image voxels to `ref_fname` voxels given `matfile`

    Parameters
    ------------
    matfile : str
        filename of output ``-omat`` transformation file from flirt
    in_fname : str
        filename for image passed to flirt as ``-in`` image
    ref_fname : str
        filename for image passed to flirt as ``-ref`` image

    Returns
    -------
    aff : (4,4) array
        Transform from voxel coordinates in image for ``in_fname`` to voxel
        coordinates in image for ``ref_fname``
    """
    mat = np.loadtxt(matfile)
    in_img = nib.load(in_fname)
    ref_img = nib.load(ref_fname)
    return flirt2aff(mat, in_img, ref_img)

def warp_displacements(ffa,flaff,fdis,fref,ffaw,order=1):
    ''' Warp an image using fsl displacements 
    
    Parameters
    ------------
    ffa : filename of nifti to be warped
    flaff : filename of .mat  (flirt)
    fdis :  filename of displacements (fnirtfileutils)
    fref : filename of reference volume e.g. (FMRIB58_FA_1mm.nii.gz)
    ffaw : filename for the output warped image
    
    '''
    
    refaff=nib.load(fref).get_affine()    
    disdata=nib.load(fdis).get_data()
    imgfa=nib.load(ffa)
    fadata=imgfa.get_data()
    fazooms=imgfa.get_header().get_zooms()    
    #from fa index to ref index
    res=flirt2aff_files(flaff,ffa,fref)
    #from ref index to fa index
    ires=np.linalg.inv(res)    
    #create the 4d volume which has the indices for the reference image  
    reftmp=np.zeros(disdata.shape)
    '''    
    #create the grid indices for the reference
    #refinds = np.ndindex(disdata.shape[:3])  
    for ijk_t in refinds:
        i,j,k = ijk_t   
        reftmp[i,j,k,0]=i
        reftmp[i,j,k,1]=j
        reftmp[i,j,k,2]=k
    '''
    #same as commented above but much faster
    reftmp[...,0] = np.arange(disdata.shape[0])[:,newaxis,newaxis]
    reftmp[...,1] = np.arange(disdata.shape[1])[newaxis,:,newaxis]
    reftmp[...,2] = np.arange(disdata.shape[2])[newaxis,newaxis,:]
        
    #affine transform from reference index to the fa index
    A = np.dot(reftmp,ires[:3,:3].T)+ires[:3,3]
    #add the displacements but first devide them by the voxel sizes
    A2=A+disdata/fazooms
    #hold the displacements' shape reshaping
    di,dj,dk,dl=disdata.shape
    #do the interpolation using map coordinates
    #the list of points where the interpolation is done given by the reshaped in 2D A2 (list of 3d points in fa index)
    W=mc(fadata,A2.reshape(di*dj*dk,dl).T,order=order).reshape(di,dj,dk)    
    #save the warped image
    Wimg=nib.Nifti1Image(W,refaff)
    nib.save(Wimg,ffaw)
    



