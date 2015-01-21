import numpy as np
from scipy.ndimage import affine_transform


def reslice(data, affine, zooms, new_zooms, order=1, mode='constant', cval=0):
    """Reslice data with new voxel resolution defined by ``new_zooms``

    Parameters
    ----------
    data : array, shape (I,J,K) or (I,J,K,N)
        3d volume or 4d volume with datasets
    affine : array, shape (4,4)
        mapping from voxel coordinates to world coordinates
    zooms : tuple, shape (3,)
        voxel size for (i,j,k) dimensions
    new_zooms : tuple, shape (3,)
        new voxel size for (i,j,k) after resampling
    order : int, from 0 to 5
        order of interpolation for resampling/reslicing,
        0 nearest interpolation, 1 trilinear etc..
        if you don't want any smoothing 0 is the option you need.
    mode : string ('constant', 'nearest', 'reflect' or 'wrap')
        Points outside the boundaries of the input are filled according
        to the given mode.
    cval : float
        Value used for points outside the boundaries of the input if
        mode='constant'.

    Returns
    -------
    data2 : array, shape (I,J,K) or (I,J,K,N)
        datasets resampled into isotropic voxel size
    affine2 : array, shape (4,4)
        new affine for the resampled image

    Examples
    --------
    >>> import nibabel as nib
    >>> from dipy.align.reslice import reslice
    >>> from dipy.data import get_data
    >>> fimg = get_data('aniso_vox')
    >>> img = nib.load(fimg)
    >>> data = img.get_data()
    >>> data.shape
    (58, 58, 24)
    >>> affine = img.get_affine()
    >>> zooms = img.get_header().get_zooms()[:3]
    >>> zooms
    (4.0, 4.0, 5.0)
    >>> new_zooms = (3.,3.,3.)
    >>> new_zooms
    (3.0, 3.0, 3.0)
    >>> data2, affine2 = reslice(data, affine, zooms, new_zooms)
    >>> data2.shape
    (77, 77, 40)
    """
    R = np.diag(np.array(new_zooms)/np.array(zooms))
    new_shape = np.array(zooms)/np.array(new_zooms) * np.array(data.shape[:3])
    new_shape = np.round(new_shape).astype('i8')
    if data.ndim == 3:
        data2 = affine_transform(input=data, matrix=R, offset=np.zeros(3,),
                                 output_shape=tuple(new_shape),
                                 order=order, mode=mode, cval=cval)
    if data.ndim == 4:
        data2l=[]
        for i in range(data.shape[-1]):
            tmp = affine_transform(input=data[..., i], matrix=R,
                                   offset=np.zeros(3,),
                                   output_shape=tuple(new_shape),
                                   order=order, mode=mode, cval=cval)
            data2l.append(tmp)
        data2 = np.zeros(tmp.shape+(data.shape[-1],), data.dtype)
        for i in range(data.shape[-1]):
            data2[..., i] = data2l[i]

    Rx = np.eye(4)
    Rx[:3, :3] = R
    affine2 = np.dot(affine, Rx)
    return data2, affine2
