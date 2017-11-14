from __future__ import division, print_function

from datetime import datetime

import nibabel as nib
import numpy as np
from dipy.denoise.denspeed import nlmeans_3d
from dipy.denoise.noise_estimate import estimate_sigma, piesno


def nlmeans(arr, sigma, mask=None, patch_radius=1, block_radius=5,
            rician=False, num_threads=None):
    r""" Non-local means for denoising 3D and 4D images

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be denoised
    mask : 3D ndarray
    sigma : float or 3D array
        standard deviation of the noise estimated from the data
    patch_radius : int
        patch size is ``2 x patch_radius + 1``. Default is 1.
    block_radius : int
        block size is ``2 x block_radius + 1``. Default is 5.
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.
    num_threads : int
        Number of threads. If None (default) then all available threads
        will be used (all CPU cores).

    Returns
    -------
    denoised_arr : ndarray
        the denoised ``arr`` which has the same shape as ``arr``.

    References
    ----------
    .. [Descoteaux08] Descoteaux, Maxim and Wiest-Daessle`, Nicolas and Prima,
                      Sylvain and Barillot, Christian and Deriche, Rachid
                      Impact of Rician Adapted Non-Local Means Filtering on
                      HARDI, MICCAI 2008

    """
    if arr.ndim == 3:
        sigma = np.ones(arr.shape, dtype=np.float64) * sigma
        return nlmeans_3d(arr, mask, sigma,
                          patch_radius, block_radius,
                          rician, num_threads).astype(arr.dtype)

    elif arr.ndim == 4:
        denoised_arr = np.zeros_like(arr)

        if isinstance(sigma, np.ndarray) and sigma.ndim == 3:
            sigma = (np.ones(arr.shape, dtype=np.float64) *
                     sigma[..., np.newaxis])
        else:
            sigma = np.ones(arr.shape, dtype=np.float64) * sigma

        for i in range(arr.shape[-1]):
            start = datetime.now()
            denoised_arr[..., i] = nlmeans_3d(arr[..., i],
                                              mask,
                                              sigma[..., i],
                                              patch_radius,
                                              block_radius,
                                              rician,
                                              num_threads).astype(arr.dtype)
            print(datetime.now() - start)

        return denoised_arr

    else:
        raise ValueError("Only 3D or 4D array are supported!", arr.shape)


def main():
    # input_path = '/home/nil/libs/scilpy-data/fa.nii.gz'
    # output_path = '/home/nil/libs/scilpy-data/fa_nlm.nii.gz'
    input_path = '/home/nil/dwi.nii.gz'
    output_path = '/home/nil/dwi_nlm.nii.gz'
    # input_path = '/home/nil/libs/scilpy-data/test.nii.gz'
    # output_path = '/home/nil/libs/scilpy-data/dwi_nlmeans.nii.gz'

    print('Denoising {0}'.format(input_path))
    image = nib.load(input_path)
    data = image.get_data()
    print(data.shape)
    # data = data[20:124, 20:168, 20:130]

    print('Estimating sigma')
    #sigma = estimate_sigma(data)
    sigma = piesno(data, 1)
    print('Found sigma {0}'.format(sigma))

    start = datetime.now()
    denoised_data = nlmeans(data, sigma, patch_radius=6, block_radius=5)
    print('Total time:', datetime.now() - start)
    denoised_image = nib.Nifti1Image(
        denoised_data, image.affine, image.header)

    denoised_image.to_filename(output_path)
    print('Denoised volume saved as {0}'.format(output_path))


main()
