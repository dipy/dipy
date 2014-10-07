#! /usr/bin/env python

from __future__ import division, print_function

import nibabel as nib
import numpy as np

import os
import argparse

from dipy.denoise.signal_transformation_framework import chi_to_gauss, fixed_point_finder, piesno
from scipy.stats import mode
from scipy.ndimage.filters import gaussian_filter
from dipy.denoise.nlmeans import nlmeans
from skimage.restoration import denoise_bilateral


DESCRIPTION = """
    Convenient script to transform noisy rician/non central chi signals into
    gaussian distributed signals.

    Reference:
    [1]. Koay CG, Ozarslan E and Basser PJ.
    A signal transformational framework for breaking the noise floor
    and its applications in MRI.
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('input', action='store', metavar=' ',
                   help='Path of the image file to stabilize.')

    p.add_argument('-N', action='store', dest='N',
                   metavar=' ', required=True, type=int,
                   help='Number of receiver coils of the scanner for GRAPPA \
                   reconstruction. Use 1 in the case of a SENSE reconstruction. \
                   Default : 4 for the 1.5T from Sherbrooke.')

    p.add_argument('-o', action='store', dest='savename',
                   metavar='savename', required=False, default=None, type=str,
                   help='Path and prefix for the saved transformed file. \
                   The name is always appended with _stabilized.nii.gz')

    return p


def main():

    parser = buildArgsParser()
    args = parser.parse_args()

    vol = nib.load(args.input)
    data = vol.get_data()
    header = vol.get_header()
    affine = vol.get_affine()

    dtype = data.dtype
    #data = data[:, 20:30, ...]
    # Since negatives are allowed, convert uint to int
    if dtype.kind == 'u':
        dtype = dtype.name[1:]

    if args.savename is None:
        if os.path.basename(args.input).endswith('.nii'):
            temp = os.path.basename(args.input)[:-4]
        elif os.path.basename(args.input).endswith('.nii.gz'):
            temp = os.path.basename(args.input)[:-7]

        filename = os.path.split(os.path.abspath(args.input))[0] + '/' + temp
        print("savename is", filename)

    else:
        filename = args.savename

    N = args.N
    sigma = np.zeros(data.shape[-2], dtype=np.float32)
    mask_noise = np.zeros(data.shape[:-1], dtype=np.bool)
    #eta = np.zeros_like(data, dtype=np.float32)
    #data_stabilized = np.zeros_like(data, dtype=np.int16)

    from time import time
    deb = time()


#    m_hat = np.zeros_like(data, dtype=np.float64)
#    for idx in range(data.shape[-1]):
#        m_hat[..., idx] = denoise_bilateral(data[..., idx], sigma_range=0.1, sigma_spatial=15)


    for idx in range(data.shape[-2]):
        print("Now processing slice", idx+1, "out of", data.shape[-2])

        sigma[idx], mask_noise[..., idx] = piesno(data[..., idx, :],  N)

    print(sigma)
    print(np.percentile(sigma, 10.),  np.percentile(sigma, 90.))

    #sigma_mode = np.load(filename + "_sigma.npy")

    sigma_mode, num = mode(sigma, axis=None)
    print("mode of sigma is", sigma_mode, "with nb", num, "median is", np.median(sigma))
    np.save(filename + "_sigma.npy", sigma_mode)
    nib.save(nib.Nifti1Image(mask_noise.astype(np.int8), affine, header), filename + '_mask_noise.nii.gz')

#    m_hat = np.zeros_like(data, dtype=np.float64)
#    for idx in range(data.shape[-1]):
#        m_hat[..., idx] = gaussian_filter(data[..., idx], 0.5)

  #  m_hat = nlmeans(data, sigma_mode, rician=False)
    m_hat = data
   # m_hat *= mask_noise[..., None]


  #  m_hat = nib.load('/home/local/USHERBROOKE/stjs2902/Bureau/phantomas_mic/b3000/dwis.nii.gz').get_data()
#    nib.save(nib.Nifti1Image(m_hat, affine, header), filename + '_m_hat.nii.gz')
    #sigma_mode=515.



    eta = fixed_point_finder(m_hat, sigma_mode, N)
    #eta=m_hat
    print(data.shape, m_hat.shape, eta.shape)
    nib.save(nib.Nifti1Image(eta.astype(dtype), affine, header), filename + '_eta.nii.gz')

        #eta[..., idx, :] = fixed_point_finder(data[..., idx, :], sigma[idx], N)

    ##data_stabilized = chi_to_gauss(m_hat, eta, sigma_mode, N)
    data_stabilized = chi_to_gauss(data, eta, sigma_mode, N)

    print("temps total:", time() - deb)
    nib.save(nib.Nifti1Image(data_stabilized.astype(dtype), affine, header), filename + "_stabilized.nii.gz")

    print("Detected noise std was :", sigma_mode)


if __name__ == "__main__":
    main()
