import numpy as np
from dipy.denoise.gibbs import gibbs_removal_2d
from numpy.testing import assert_

# Produce a 2D image
Nori = 32
image = np.zeros((6 * Nori, 6 * Nori))
image[Nori: 2 * Nori, Nori: 2 * Nori] = 1
image[Nori: 2 * Nori, 4 * Nori: 5 * Nori] = 1
image[2 * Nori: 3 * Nori, Nori: 3 * Nori] = 1
image[3 * Nori: 4 * Nori, 2 * Nori: 3 * Nori] = 2
image[3 * Nori: 4 * Nori, 4 * Nori: 5 * Nori] = 1
image[4 * Nori: 5 * Nori, 3 * Nori: 5 * Nori] = 3

# Corrupt image with gibbs ringing
c = np.fft.fft2(image)
c = np.fft.fftshift(c)
N = c.shape[0]
c_crop = c[48:144, 48:144]
N = c_crop.shape[0]
image_gibbs = abs(np.fft.ifft2(c_crop)/4)

# Produce ground truth
Nre = 16
image_gt = np.zeros((6 * Nre, 6 * Nre))
image_gt[Nre: 2 * Nre, Nre: 2 * Nre] = 1
image_gt[Nre: 2 * Nre, 4 * Nre: 5 * Nre] = 1
image_gt[2 * Nre: 3 * Nre, Nre: 3 * Nre] = 1
image_gt[3 * Nre: 4 * Nre, 2 * Nre: 3 * Nre] = 2
image_gt[3 * Nre: 4 * Nre, 4 * Nre: 5 * Nre] = 1
image_gt[4 * Nre: 5 * Nre, 3 * Nre: 5 * Nre] = 3


def test_gibbs_2d():
    image_cor, tv = gibbs_removal_2d(image_gibbs)

    # Correction of gibbs ringing have to be closer to gt than denoised image
    diff_raw = abs(image_gibbs - image_gt)
    diff_cor = abs(image_cor - image_gt)
    assert_(diff_raw.mean() > diff_cor.mean)
