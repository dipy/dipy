import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from dipy.data import get_data
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

# Execute the segmentation
nclasses = 4
beta = np.float64(0.1)
max_iter = 12

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


def test_greyscale_image():

    com = ConstantObservationModel()
    icm = IteratedConditionalModes()

    mu, sigma = com.initialize_param_uniform(image, nclasses)
    sigmasq = sigma ** 2
    npt.assert_equal(mu, np.array([0., 0.25, 0.5, 0.75]))
    npt.assert_equal(sigma, np.array([1.0, 1.0, 1.0, 1.0]))
    npt.assert_equal(sigmasq, np.array([1.0, 1.0, 1.0, 1.0]))

    neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)
    npt.assert_equal(neglogl.all() <= 1, True)
    npt.assert_equal(neglogl.all() > 0, True)
    print('negloglikelihood for one voxel in white matter')
    npt.assert_equal((neglogl[100, 100, 2, 0] != neglogl[100, 100, 2, 1]),
                     True)
    npt.assert_equal((neglogl[100, 100, 2, 1] != neglogl[100, 100, 2, 2]),
                     True)
    npt.assert_equal((neglogl[100, 100, 2, 2] != neglogl[100, 100, 2, 3]),
                     True)
    npt.assert_equal((neglogl[100, 100, 2, 1] != neglogl[100, 100, 2, 3]),
                     True)

    initial_segmentation = icm.initialize_maximum_likelihood(neglogl)
    npt.assert_equal(initial_segmentation.max(), nclasses - 1)
    npt.assert_equal(initial_segmentation.min(), 0)

    PLN = com.prob_neighborhood(image, initial_segmentation, beta, nclasses)
    npt.assert_equal(PLN.all() >= 0.0, True)
    npt.assert_equal(PLN.all() <= 1.0, True)
    PLY = com.prob_image(image, nclasses, mu, sigmasq, PLN)
    npt.assert_equal(PLY.all() >= 0.0, True)
    npt.assert_equal(PLY.all() <= 1.0, True)

    mu_upd, sigmasq_upd = com.update_param(image, PLY, mu, nclasses)
    print('updated means and variances vs original ones')
    npt.assert_equal(mu_upd != mu, True)
    npt.assert_equal(sigmasq_upd != sigmasq, True)

    icm_segmentation = icm.icm_ising(neglogl, beta, initial_segmentation)
    npt.assert_equal(np.abs(np.sum(icm_segmentation)) != 0, True)
    npt.assert_equal(icm_segmentation.max(), nclasses - 1)
    npt.assert_equal(icm_segmentation.min(), 0)

    return initial_segmentation, icm_segmentation


def test_greyscale_iter():

    com = ConstantObservationModel()
    icm = IteratedConditionalModes()

    mu, sigma = com.initialize_param_uniform(image, nclasses)
    sigmasq = sigma ** 2
    neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)
    initial_segmentation = icm.initialize_maximum_likelihood(neglogl)
    npt.assert_equal(initial_segmentation.max(), nclasses - 1)
    npt.assert_equal(initial_segmentation.min(), 0)

    mu, sigma, sigmasq = com.seg_stats(image, initial_segmentation, nclasses)
    print('initial mu:', mu)
    print('initial var:', sigmasq)
    npt.assert_equal(mu.all() >= 0, True)
    npt.assert_equal(sigmasq.all() >= 0, True)

    zero = np.zeros_like(image)+0.01
    zero_noise = add_noise(zero, 1000, 1, noise_type='gaussian')
    image_gauss = np.where(image == 0, zero_noise, image)

#    image_gauss = add_noise(image, 1000, 1, noise_type='gaussian')

    plt.figure()
    plt.imshow(image_gauss[..., 1]) 
    plt.colorbar()

    final_segmentation = np.empty_like(image)
    seg_init = initial_segmentation.copy()

    for i in range(max_iter):

        print('iteration: ', i)

        PLN = com.prob_neighborhood(image_gauss, initial_segmentation, beta,
                                    nclasses)
        npt.assert_equal(PLN.all() >= 0.0, True)
        PLY = com.prob_image(image_gauss, nclasses, mu, sigmasq, PLN)
        npt.assert_equal(PLY.all() >= 0.0, True)

        mu_upd, sigmasq_upd = com.update_param(image_gauss, PLY, mu, nclasses)
        npt.assert_equal(mu_upd.all() >= 0.0, True)
        npt.assert_equal(sigmasq_upd.all() >= 0.0, True)
        negll = com.negloglikelihood(image_gauss, mu_upd, sigmasq_upd, nclasses)
        npt.assert_equal(negll.all() >= 0.0, True)
        
        final_segmentation, energy = icm.icm_ising(negll, beta,
                                                   initial_segmentation)

#        if i > 0:
#            npt.assert_equal(energy[100, 100, 2] <= energy_pre[100, 100, 2], True)
#        energy_pre = energy.copy()

#        plt.figure()
#        plt.imshow(final_segmentation[..., 1])

        initial_segmentation = final_segmentation.copy()
        mu = mu_upd.copy()
        sigmasq = sigmasq_upd.copy()

    difference_map = np.abs(seg_init - final_segmentation)
    npt.assert_equal(np.abs(np.sum(difference_map)) != 0, True)

    return seg_init, final_segmentation, PLY


def test_square_iter():

    com = ConstantObservationModel()
    icm = IteratedConditionalModes()

    initial_segmentation = square.copy()
    npt.assert_equal(initial_segmentation.max(), nclasses - 1)
    npt.assert_equal(initial_segmentation.min(), 0)

    mu, sigma, sigmasq = com.seg_stats(square_1, initial_segmentation,
                                       nclasses)
    npt.assert_equal(mu.all() >= 0, True)
    npt.assert_equal(sigmasq.all() >= 0, True)

    final_segmentation = np.empty_like(square_1)
    seg_init = initial_segmentation.copy()

#    energy_pre = np.zeros_like(image)

    for i in range(max_iter):

        print("Iteration: %d"%(max_iter,))

        PLN = com.prob_neighborhood(square_1, initial_segmentation, beta,
                                    nclasses)
        npt.assert_equal(PLN.all() >= 0.0, True)
        PLY = com.prob_image(square_1, nclasses, mu, sigmasq, PLN)
        npt.assert_equal(PLY.all() >= 0.0, True)

        mu_upd, sigmasq_upd = com.update_param(square_1, PLY, mu, nclasses)
        npt.assert_equal(mu_upd.all() >= 0.0, True)
        npt.assert_equal(sigmasq_upd.all() >= 0.0, True)
        negll = com.negloglikelihood(square_1, mu_upd, sigmasq_upd, nclasses)
        npt.assert_equal(negll.all() >= 0.0, True)
        final_segmentation, energy = icm.icm_ising(negll, beta,
                                                   initial_segmentation)

#        if i > 0:
#            npt.assert_equal(energy[100, 100, 2] <= energy_pre[100, 100, 2], True)
#        energy_pre = energy.copy()

#        plt.figure()
#        plt.imshow(final_segmentation[..., 1])

        initial_segmentation = final_segmentation.copy()
        mu = mu_upd.copy()
        sigmasq = sigmasq_upd.copy()

    difference_map = np.abs(seg_init - final_segmentation)
    npt.assert_equal(np.abs(np.sum(difference_map)) != 0, True)

    return seg_init, final_segmentation, PLY


def test_segment_hmrf():

    imgseg = ImageSegmenter()

    T1coronal_init, T1coronal_final, PLY = imgseg.segment_hmrf(image, nclasses,
                                                          beta, max_iter)
    
    npt.assert_equal(T1coronal_final.max(), nclasses - 1)
    npt.assert_equal(T1coronal_final.min(), 0)

    return T1coronal_init, T1coronal_final, PLY 


if __name__ == '__main__':
    pass
    # npt.run_module_suite()
    # initial_segmentation, final_segmentation = test_greyscale_image()
    seg_init, final_segmentation, PLY = test_greyscale_iter()
    # seg_init, final_segmentation, PLY = test_square_iter()
    # T1init, T1final, PLY = test_segment_hmrf()
