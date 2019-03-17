import numpy as np
from dipy.denoise.gibbs import (gibbs_removal_2d, gibbs_removal)
from numpy.testing import (assert_, assert_array_almost_equal, assert_raises)

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

# Suppressing gibbs artefacts
image_cor = gibbs_removal_2d(image_gibbs)


def test_gibbs_2d():
    # Correction of gibbs ringing have to be closer to gt than denoised image
    diff_raw = np.mean(abs(image_gibbs - image_gt))
    diff_cor = np.mean(abs(image_cor - image_gt))
    assert_(diff_raw > diff_cor)


def test_gibbs_3d():
    image3d = np.zeros((6 * Nre, 6 * Nre, 2))
    image3d[:, :, 0] = image_gibbs
    image3d[:, :, 1] = image_gibbs

    image3d_cor = gibbs_removal(image3d, 2)
    assert_array_almost_equal(image3d_cor[:, :, 0], image_cor)
    assert_array_almost_equal(image3d_cor[:, :, 1], image_cor)


def test_gibbs_4d():
    image4d = np.zeros((6 * Nre, 6 * Nre, 2, 2))
    image4d[:, :, 0, 0] = image_gibbs
    image4d[:, :, 1, 0] = image_gibbs
    image4d[:, :, 0, 1] = image_gibbs
    image4d[:, :, 1, 1] = image_gibbs

    image4d_cor = gibbs_removal(image4d)
    assert_array_almost_equal(image4d_cor[:, :, 0, 0], image_cor)
    assert_array_almost_equal(image4d_cor[:, :, 1, 0], image_cor)
    assert_array_almost_equal(image4d_cor[:, :, 0, 1], image_cor)
    assert_array_almost_equal(image4d_cor[:, :, 1, 1], image_cor)


def test_swaped_gibbs_3d():
    image3d = np.zeros((6 * Nre, 2, 6 * Nre))
    image3d[:, 0, :] = image_gibbs
    image3d[:, 1, :] = image_gibbs

    image3d_cor = gibbs_removal(image3d, slice_axis=1)
    assert_array_almost_equal(image3d_cor[:, 0, :], image_cor)
    assert_array_almost_equal(image3d_cor[:, 1, :], image_cor)

    image3d = np.zeros((2, 6 * Nre, 6 * Nre))
    image3d[0, :, :] = image_gibbs
    image3d[1, :, :] = image_gibbs

    image3d_cor = gibbs_removal(image3d, slice_axis=0)
    assert_array_almost_equal(image3d_cor[0, :, :], image_cor)
    assert_array_almost_equal(image3d_cor[1, :, :], image_cor)


def test_swaped_gibbs_4d():
    image4d = np.zeros((2, 6 * Nre, 6 * Nre, 2))
    image4d[0, :, :, 0] = image_gibbs
    image4d[1, :, :, 0] = image_gibbs
    image4d[0, :, :, 1] = image_gibbs
    image4d[1, :, :, 1] = image_gibbs

    image4d_cor = gibbs_removal(image4d, slice_axis=0)
    assert_array_almost_equal(image4d_cor[0, :, :, 0], image_cor)
    assert_array_almost_equal(image4d_cor[1, :, :, 0], image_cor)
    assert_array_almost_equal(image4d_cor[0, :, :, 1], image_cor)
    assert_array_almost_equal(image4d_cor[1, :, :, 1], image_cor)


def test_gibbs_errors():
    assert_raises(ValueError, gibbs_removal, np.ones((2, 2, 2, 2, 2)))
    assert_raises(ValueError, gibbs_removal, np.ones((2, 2)))
    assert_raises(ValueError, gibbs_removal, np.ones((2, 2, 2)), 3)
