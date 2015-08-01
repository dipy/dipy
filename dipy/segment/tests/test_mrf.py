import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from dipy.data import get_data
from dipy.core.ndindex import ndindex
from dipy.sims.voxel import add_noise
from dipy.segment.mrf import (ConstantObservationModel,
                              IteratedConditionalModes,
                              ImageSegmenter)


# Load a coronal slice from a T1-weighted MRI
fname = get_data('t1_coronal_slice')
single_slice = np.load(fname)

# Stack a few copies to form a 3D volume
nslices = 5
image = np.zeros(shape=single_slice.shape + (nslices,))
image[..., :nslices] = single_slice[..., None]
#image = image * 200

# Execute the segmentation
nclasses = 4
beta = np.float64(0.01)
max_iter = 3

square = np.zeros((256, 256, 3))
square[42:213, 42:213, :] = 3
square[71:185, 71:185, :] = 2
square[99:157, 99:157, :] = 1

square_1 = np.zeros((256, 256, 3))
temp_1 = np.random.random_integers(20, size=(171, 171, 3))
temp_1 = np.where(temp_1 < 20, 3, 2)
square_1[42:213, 42:213, :] = temp_1
temp_2 = np.random.random_integers(20, size=(114, 114, 3))
temp_2 = np.where(temp_2 < 19, 2, 3)
square_1[71:185, 71:185, :] = temp_2
temp_3 = np.random.random_integers(20, size=(58, 58, 3))
temp_3 = np.where(temp_3 < 20, 1, 2)
square_1[99:157, 99:157, :] = temp_3

square_2 = np.zeros((256, 256, 3))
temp_1 = np.random.random_integers(20, size=(171, 171, 3))
temp_1 = np.where(temp_1 < 20, 3, 2)
square_2[42:213, 42:213, :] = temp_1
temp_2 = np.random.random_integers(20, size=(114, 114, 3))
temp_2 = np.where(temp_2 < 19, 2, np.where(temp_2 == 19, 1, 3))
square_2[71:185, 71:185, :] = temp_2
temp_3 = np.random.random_integers(20, size=(58, 58, 3))
temp_3 = np.where(temp_3 < 20, 1, 2)
square_2[99:157, 99:157, :] = temp_3

square_gauss = add_noise(square, 4, 1, noise_type='gaussian')


def test_initialize_param_uniform():

    com = ConstantObservationModel()

    mu, sigma = com.initialize_param_uniform(image, nclasses)

    print(mu)
    print(sigma)

    npt.assert_almost_equal(mu, np.array([0., 0.25, 0.5, 0.75]))
    npt.assert_equal(sigma, np.array([1.0, 1.0, 1.0, 1.0]))


def test_negloglikelihood():

    com = ConstantObservationModel()

    mu, sigma = com.initialize_param_uniform(image, nclasses)

    print(mu)
    print(sigma)

    npt.assert_almost_equal(mu, np.array([0., 0.25, 0.5, 0.75]))
    npt.assert_equal(sigma, np.array([1.0, 1.0, 1.0, 1.0]))

    sigmasq = sigma ** 2

    neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)

    print(neglogl.shape)
    print(neglogl.min())
    print(neglogl.max())

    # Testing the likelihood of the same voxel for two different labels
    print('negloglikelihood for one voxel in brain tissue')
    print(neglogl[150, 125, 2, 0])
    print(neglogl[150, 125, 2, 1])
    print(neglogl[150, 125, 2, 2])
    print(neglogl[150, 125, 2, 3])
    plt.figure()
    plt.imshow(neglogl[:, :, 2, 0])
    plt.figure()
    plt.imshow(neglogl[:, :, 2, 1])
    plt.figure()
    plt.imshow(neglogl[:, :, 2, 2])
    plt.figure()
    plt.imshow(neglogl[:, :, 2, 3])
    npt.assert_equal((neglogl[150, 125, 2, 0] != neglogl[150, 125, 2, 1]),
                     True)
    npt.assert_equal((neglogl[150, 125, 2, 1] != neglogl[150, 125, 2, 2]),
                     True)
    npt.assert_equal((neglogl[150, 125, 2, 2] != neglogl[150, 125, 2, 3]),
                     True)
    npt.assert_equal((neglogl[150, 125, 2, 1] != neglogl[150, 125, 2, 3]),
                     True)


def test_greyscale_image():

    com = ConstantObservationModel()
    icm = IteratedConditionalModes()

    mu, sigma = com.initialize_param_uniform(image, nclasses)

    print(mu)
    print(sigma)

    npt.assert_almost_equal(mu, np.array([0., 0.25, 0.5, 0.75]))
    npt.assert_equal(sigma, np.array([1.0, 1.0, 1.0, 1.0]))

    sigmasq = sigma ** 2

    neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)

    print(neglogl.shape)
    print(neglogl.min())
    print(neglogl.max())
    print('maximum negloglikelihood per label')
    print(neglogl[100, 100, 2, 0].max())
    print(neglogl[100, 100, 2, 1].max())
    print(neglogl[100, 100, 2, 2].max())
    print(neglogl[100, 100, 2, 3].max())

    # Testing the likelihood of the same voxel for two different labels
    print('negloglikelihood for one voxel in brain tissue')
    print(neglogl[100, 100, 2, 0])
    print(neglogl[100, 100, 2, 1])
    print(neglogl[100, 100, 2, 2])
    print(neglogl[100, 100, 2, 3])
    npt.assert_equal((neglogl[100, 100, 2, 0] != neglogl[100, 100, 2, 1]),
                     True)
    npt.assert_equal((neglogl[100, 100, 2, 1] != neglogl[100, 100, 2, 2]),
                     True)
    npt.assert_equal((neglogl[100, 100, 2, 2] != neglogl[100, 100, 2, 3]),
                     True)
    npt.assert_equal((neglogl[100, 100, 2, 1] != neglogl[100, 100, 2, 3]),
                     True)

    initial_segmentation = icm.initialize_maximum_likelihood(neglogl)

    plt.imshow(image[..., 1])
    plt.figure()
    plt.imshow(initial_segmentation[..., 1])

    npt.assert_equal(initial_segmentation.max(), 3)
    npt.assert_equal(initial_segmentation.min(), 0)

    # seg = np.float64(initial_segmentation)

    PLN = com.prob_neighborhood(image, initial_segmentation, beta, nclasses)

    plt.figure()
    plt.imshow(PLN[:, :, 2, 0])
    plt.figure()
    plt.imshow(PLN[:, :, 2, 1])
    plt.figure()
    plt.imshow(PLN[:, :, 2, 2])
    plt.figure()
    plt.imshow(PLN[:, :, 2, 3])

    PLY = com.prob_image(image, nclasses, mu, sigmasq, PLN)

    plt.figure()
    plt.imshow(PLY[:, :, 2, 0])
    plt.figure()
    plt.imshow(PLY[:, :, 2, 1])
    plt.figure()
    plt.imshow(PLY[:, :, 2, 2])
    plt.figure()
    plt.imshow(PLY[:, :, 2, 3])

    mu_upd, sigmasq_upd = com.update_param(image, PLY, mu, nclasses)
    print('updated vs old means and variances')
    print(mu)
    print(mu_upd)
    print(sigmasq)
    print(sigmasq_upd)
    npt.assert_equal(mu_upd != mu, True)
    npt.assert_equal(sigmasq_upd != sigmasq, True)

    icm_segmentation = icm.icm_ising(neglogl, beta, initial_segmentation)
    plt.figure()
    plt.imshow(icm_segmentation[..., 1])
    npt.assert_equal(np.abs(np.sum(icm_segmentation)) != 0, True)

    return initial_segmentation, icm_segmentation


def test_greyscale_iter():

    com = ConstantObservationModel()
    icm = IteratedConditionalModes()

    plt.figure()
    plt.imshow(image[..., 1])

    mu, sigma = com.initialize_param_uniform(image, nclasses)
    sigmasq = sigma ** 2
    neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)
#    plt.figure()
#    plt.imshow(neglogl[:, :, 1, 1])
    initial_segmentation = icm.initialize_maximum_likelihood(neglogl)

    mu, sigma, sigmasq = com.seg_stats(image, initial_segmentation, nclasses)
    print(mu)
    print(sigmasq)
    plt.figure()
    plt.imshow(initial_segmentation[..., 1])
#    npt.assert_equal(initial_segmentation.max(), 3)
#    npt.assert_equal(initial_segmentation.min(), 0)

    # final_segmentation = initial_segmentation.copy()
    final_segmentation = np.empty_like(image)
    seg_init = initial_segmentation.copy()

    for i in range(max_iter):

        print('iteration: ', i)

        PLN, PLN_norm = com.prob_neighborhood(image, initial_segmentation, beta, nclasses)
        # PLY = com.prob_image(image, nclasses, mu, sigmasq, PLN)
        PLY = prob_image(image, nclasses, mu, sigmasq, PLN)

#        npt.assert_equal(PLY.all() >= 0, True)

        # mu_upd, sigmasq_upd = com.update_param(image, PLY, mu, nclasses)
        mu_upd, sigmasq_upd = update_param(image, PLY, mu, nclasses)
        negll = com.negloglikelihood(image, mu_upd, sigmasq_upd, nclasses)
#        plt.figure()
#        plt.imshow(negll[:, :, 1, 1])
        final_segmentation = icm.icm_ising(negll, beta, initial_segmentation)

        plt.figure()
        plt.imshow(final_segmentation[..., 1])

        initial_segmentation = final_segmentation.copy()
        mu = mu_upd.copy()
        sigmasq = sigmasq_upd.copy()

#    Difference_map = np.abs(initial_segmentation - final_segmentation)
#    plt.figure()
#    plt.imshow(Difference_map[..., 1])

    return seg_init, final_segmentation, mu, sigmasq


def test_ImageSegmenter():

    imgseg = ImageSegmenter()

    T1_seg = imgseg.segment_HMRF(image, nclasses, beta, max_iter)

    plt.figure()
    plt.imshow(T1_seg[..., 1])

#    Square1_seg = imgseg.segment_HMRF(image, nclasses, beta, max_iter)
#
#    plt.figure()
#    plt.imshow(Square1_seg[..., 1])

    return T1_seg

def prob_image(img, nclasses, mu, sigmasq, P_L_N):
    r""" Conditional probability of the label given the image
    This is for equation 27 of the Zhang paper

    Parameters
    -----------
    img : ndarray 3D
        masked T1 structural image
    nclasses : int
        number of tissue classes
    mu : ndarray (1, 3)
        current estimate of mean of each tissue type
    sigmasq : ndarray (1, 3)
        current estimate of the variance of each tissue type
    P_L_N : ndarray 4D
        probability of the label given the neighborhood. Previously
        computed by function prob_neigh

    Returns
    --------
    P_L_Y : ndarray 4D
        Probability of the label given the input image

    """
    # probability of the tissue label (from the 3 classes) given the
    # voxel
    P_L_Y = np.zeros_like(P_L_N)
    P_L_Y_norm = np.zeros_like(img)
    # normal density equation 11 of the Zhang paper
    g = np.zeros_like(img)
    # mask = np.where(img > 0, 1, 0)
    Epsilon = 1e-6
    Epsilonsq = Epsilon*Epsilon
    
    
    for l in range(nclasses):
        
        print('This is the max PLN', P_L_N.max())
        print('This is the min PLN', P_L_N.min())
        for idx in ndindex(img.shape[:3]):
            idxl = idx + (l,)
            if sigmasq[l] < Epsilonsq:
                if np.abs(img[idx] - mu[l]) <= Epsilon:
                    g[idx] = 1
                else:
                    g[idx] = 0
            else:
                g[idx] = (np.exp(-((img[idx] - mu[l]) ** 2) / (2 * sigmasq[l]))) / (np.sqrt(2 * np.pi * sigmasq[l]))
            
            P_L_Y[idxl] = g[idx] * P_L_N[idxl]

        P_L_Y_norm[:, :, :] += P_L_Y[:, :, :, l]
        
    if np.max(P_L_Y[...,0]) < Epsilon:
        print('Esta es mi mu:', mu)
        print('Esta es mi varianza:', sigmasq)

    for l in range(nclasses):
        P_L_Y[:, :, :, l] = P_L_Y[:, :, :, l]/P_L_Y_norm

    #P_L_Y[np.isnan(P_L_Y)] = 0
    # P_L_Y[P_L_Y < 0] = 0

    return P_L_Y


def update_param(image, P_L_Y, mu, nclasses):
    r""" Updates the means and the variances in each iteration for all the
    labels. This is for equations 25 and 26 of the Zhang paper

    Parameters
    -----------
    image : ndarray
        Input T1 grey scale image

    P_L_Y : ndarray
        Probability of the label given the input image computed by the
        Expectation Maximization algorithm.

    Returns
    --------
    mu_upd : 1x3 ndarray - mean of each tissue class
    var_upd : 1x3 ndarray - variance of each tissue class

    """
    # temporary mu and var files to compute the update
    mu_upd = np.zeros(nclasses)
    var_upd = np.zeros(nclasses)
    mu_num = np.zeros(image.shape + (nclasses,))
    var_num = np.zeros(image.shape + (nclasses,))
    denm = np.zeros(image.shape + (nclasses,))

    for l in range(nclasses):
        print('This is the max PLY', P_L_Y[...,l].max(), 'label', l)
        print('This is the min PLY', P_L_Y[...,l].min(), 'label', l)
        for idx in ndindex(image.shape[:3]):
            idxl = idx + (l,)
            mu_num[idxl] = (P_L_Y[idxl] * image[idx])
            var_num[idxl] = (P_L_Y[idxl] * (image[idx] - mu[l]) ** 2)
            denm[idxl] = P_L_Y[idxl]

        mu_upd[l] = np.sum(mu_num[:, :, :, l]) / np.sum(denm[:, :, :, l])
        var_upd[l] = np.sum(var_num[:, :, :, l]) / np.sum(denm[:, :, :, l])
#        if var_upd[l] < 1e-6:
#            var_upd[l] = 1e-6
            
        print('updated means and variances per class')
        print('class: ', l)
        print('updated_mu:', mu_upd[l])
        print('updated_var:', var_upd[l])

    return mu_upd, var_upd


if __name__ == '__main__':

    # test_initialize_param_uniform()
    # test_negloglikelihood()
    # initial_segmentation, final_segmentation = test_greyscale_image()
    seg_init, final_segmentation, mu, sigmasq = test_greyscale_iter()
    # segmented = test_ImageSegmenter()
