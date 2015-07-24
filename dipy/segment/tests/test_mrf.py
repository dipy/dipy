import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from dipy.data import get_data
from dipy.sims.voxel import add_noise
from dipy.segment.mrf import (ConstantObservationModel,
                              IteratedConditionalModes)


# Load a coronal slice from a T1-weighted MRI
fname = get_data('t1_coronal_slice')
single_slice = np.load(fname)

# Stack a few copies to form a 3D volume
nslices = 5
image = np.zeros(shape=single_slice.shape + (nslices,))
image[..., :nslices] = single_slice[..., None]

# Execute the segmentation
nclasses = 4
beta = 0.1
max_iter = 2

square = np.zeros((256, 256, 3))
square[42:213, 42:213, :] = 3
square[71:185, 71:185, :] = 2
square[99:157, 99:157, :] = 1

A = np.zeros((256, 256, 3))
temp_1 = np.random.random_integers(20, size=(171, 171, 3))
temp_1 = np.where(temp_1 < 20, 3, 2)
A[42:213, 42:213, :] = temp_1
temp_2 = np.random.random_integers(20, size=(114, 114, 3))
temp_2 = np.where(temp_2 < 19, 2, 3)
A[71:185, 71:185, :] = temp_2
temp_3 = np.random.random_integers(20, size=(58, 58, 3))
temp_3 = np.where(temp_3 < 20, 1, 2)
A[99:157, 99:157, :] = temp_3

B = np.zeros((256, 256, 3))
temp_1 = np.random.random_integers(20, size=(171, 171, 3))
temp_1 = np.where(temp_1 < 20, 3, 2)
B[42:213, 42:213, :] = temp_1
temp_2 = np.random.random_integers(20, size=(114, 114, 3))
temp_2 = np.where(temp_2 < 19, 2, np.where(temp_2 == 19, 1, 3))
B[71:185, 71:185, :] = temp_2
temp_3 = np.random.random_integers(20, size=(58, 58, 3))
temp_3 = np.where(temp_3 < 20, 1, 2)
B[99:157, 99:157, :] = temp_3

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
    print(neglogl[100, 100, 2, 0].max())
    print(neglogl[100, 100, 2, 1].max())
    print(neglogl[100, 100, 2, 2].max())
    print(neglogl[100, 100, 2, 3].max())

    # Testing the likelihood of the same voxel for two different labels
    print(neglogl[100, 100, 2, 0])
    print(neglogl[100, 100, 2, 1])
    print(neglogl[100, 100, 2, 2])
    print(neglogl[100, 100, 2, 3])
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

    PLN = com.prob_neighborhood(image, initial_segmentation, beta, nclasses)

    plt.figure()
    plt.imshow(PLN[..., 0])
    plt.figure()
    plt.imshow(PLN[..., 1])
    plt.figure()
    plt.imshow(PLN[..., 2])
    plt.figure()
    plt.imshow(PLN[..., 3])
    
    PLY = com.prob_image(image, nclasses, mu, sigmasq, PLN)

    plt.figure()
    plt.imshow(PLY[..., 0])
    plt.imshow(PLY[..., 1])
    plt.imshow(PLY[..., 2])
    plt.imshow(PLY[..., 3])

    mu_upd, sigmasq_upd = com.update_param(image, PLY, mu, nclasses)

    npt.assert_equal(mu_upd != mu, True)
    npt.assert_equal(sigmasq_upd != sigma, True)

#    final_segmentation = initial_segmentation.copy()
#
#    for i in range(max_iter):
#
#        print(i)
#
#        PLN = com.prob_neighborhood(image, initial_segmentation, beta, nclasses)
#        PLY = com.prob_image(image, nclasses, mu, sigmasq, PLN)
#        mu_upd, sigmasq_upd = com.update_param(image, PLY)
#        negll = com.negloglikelihood(image, mu_upd, sigmasq_upd, nclasses)
#        final_segmentation = icm.icm_ising(negll, beta, segm)
#
#        figure()
#        plt.imshow(final_segmentation[..., 1])
#
#    plt.figure()
#    D = np.abs(initial_segmentation - final_segmentation)
#    plt.imshow(D[..., 1])

    return initial_segmentation #, final_segmentation


if __name__ == '__main__':
    
    test_negloglikelihood()
    #test_greyscale_image()
    #initial_segmentation, final_segmentation = test_in_parts()
    #test_segmentation()
