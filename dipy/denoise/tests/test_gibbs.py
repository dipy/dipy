import numpy as np
from dipy.denoise.gibbs import (_gibbs_removal_1d, _gibbs_removal_2d,
                                gibbs_removal, _image_tv)
from numpy.testing import (assert_, assert_array_almost_equal, assert_raises)


def setup_module():
    """Module-level setup"""
    global image_gibbs, image_gt, image_cor, Nre

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
    c_crop = c[48:144, 48:144]
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
    image_cor = _gibbs_removal_2d(image_gibbs)


def test_gibbs_2d():

    # Correction of gibbs ringing have to be closer to gt than denoised image
    diff_raw = np.mean(abs(image_gibbs - image_gt))
    diff_cor = np.mean(abs(image_cor - image_gt))
    assert_(diff_raw > diff_cor)

    # Test if gibbs_removal works for 2D data
    image_cor2 = gibbs_removal(image_gibbs)
    assert_array_almost_equal(image_cor2, image_cor)


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


def test_swapped_gibbs_2d():
    # 2D case: In this case slice_axis is a dummy variable. Since data is
    # already a single 2D image, to axis swapping is required
    image_cor0 = gibbs_removal(image_gibbs, slice_axis=0)
    assert_array_almost_equal(image_cor0, image_cor)

    image_cor1 = gibbs_removal(image_gibbs, slice_axis=1)
    assert_array_almost_equal(image_cor1, image_cor)

    image_cor2 = gibbs_removal(image_gibbs, slice_axis=2)
    assert_array_almost_equal(image_cor2, image_cor)


def test_swapped_gibbs_3d():
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


def test_swapped_gibbs_4d():
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
    assert_raises(ValueError, gibbs_removal, np.ones((2)))
    assert_raises(ValueError, gibbs_removal, np.ones((2, 2, 2)), 3)


def test_gibbs_subfunction():
    # This complementary test is to make sure that Gibbs suppression
    # sub-functions are properly implemented

    # Testing correction along axis 0
    image_a0 = _gibbs_removal_1d(image_gibbs, axis=0)
    # After this step tv along axis 0 should provide lower values than along
    # axis 1
    tv0_a0_r, tv0_a0_l = _image_tv(image_a0, axis=0)
    tv0_a0 = np.minimum(tv0_a0_r, tv0_a0_l)
    tv1_a0_r, tv1_a0_l = _image_tv(image_a0, axis=1)
    tv1_a0 = np.minimum(tv1_a0_r, tv1_a0_l)
    # Let's check that
    mean_tv0 = np.mean(abs(tv0_a0))
    mean_tv1 = np.mean(abs(tv1_a0))
    assert_(mean_tv0 < mean_tv1)

    # Testing correction along axis 1
    image_a1 = _gibbs_removal_1d(image_gibbs, axis=1)
    # After this step tv along axis 1 should provide higher values than along
    # axis 0
    tv0_a1_r, tv0_a1_l = _image_tv(image_a1, axis=0)
    tv0_a1 = np.minimum(tv0_a1_r, tv0_a1_l)
    tv1_a1_r, tv1_a1_l = _image_tv(image_a1, axis=1)
    tv1_a1 = np.minimum(tv1_a1_r, tv1_a1_l)
    # Let's check that
    mean_tv0 = np.mean(abs(tv0_a1))
    mean_tv1 = np.mean(abs(tv1_a1))
    assert_(mean_tv0 > mean_tv1)


def test_non_square_image():
    # Produce non-square 2D image
    Nori = 32
    img = np.zeros((6 * Nori, 6 * Nori))
    img[Nori: 2 * Nori, Nori: 2 * Nori] = 1
    img[2 * Nori: 3 * Nori, Nori: 3 * Nori] = 1
    img[3 * Nori: 4 * Nori, 2 * Nori: 3 * Nori] = 2
    img[4 * Nori: 5 * Nori, 3 * Nori: 5 * Nori] = 3

    # Corrupt image with gibbs ringing
    c = np.fft.fft2(img)
    c = np.fft.fftshift(c)
    c_crop = c[48:144, :]
    img_gibbs = abs(np.fft.ifft2(c_crop)/2)

    # Produce ground truth
    Nre = 16
    img_gt = np.zeros((6 * Nre, 6 * Nori))
    img_gt[Nre: 2 * Nre, Nori: 2 * Nori] = 1
    img_gt[2 * Nre: 3 * Nre, Nori: 3 * Nori] = 1
    img_gt[3 * Nre: 4 * Nre, 2 * Nori: 3 * Nori] = 2
    img_gt[4 * Nre: 5 * Nre, 3 * Nori: 5 * Nori] = 3

    # Suppressing gibbs artefacts
    img_cor = gibbs_removal(img_gibbs)

    # Correction of gibbs ringing have to be closer to gt than denoised image
    diff_raw = np.mean(abs(img_gibbs - img_gt))
    diff_cor = np.mean(abs(img_cor - img_gt))
    assert_(diff_raw > diff_cor)
