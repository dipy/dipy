import numpy as np
import numpy.testing as npt
from dipy.data import get_fnames
from dipy.sims.voxel import add_noise
from dipy.segment.mrf import (ConstantObservationModel,
                              IteratedConditionalModes)
from dipy.segment.tissue import (TissueClassifierHMRF)
from dipy.testing.decorators import set_random_number_generator


def create_image():
    # Load a coronal slice from a T1-weighted MRI
    fname = get_fnames('t1_coronal_slice')
    single_slice = np.load(fname)

    # Stack a few copies to form a 3D volume
    nslices = 5
    image = np.zeros(shape=single_slice.shape + (nslices,))
    image[..., :nslices] = single_slice[..., None]
    return image


# Making squares
def create_square():
    square = np.zeros((256, 256, 3), dtype=np.int16)
    square[42:213, 42:213, :] = 1
    square[71:185, 71:185, :] = 2
    square[99:157, 99:157, :] = 3

    return square


def create_square_uniform(rng):
    square_1 = np.zeros((256, 256, 3)) + 0.001
    square_1 = add_noise(square_1, 10000, 1,
                         noise_type='gaussian', rng=rng)
    temp_1 = rng.integers(1, 21, size=(171, 171, 3))
    temp_1 = np.where(temp_1 < 20, 1, 3)
    square_1[42:213, 42:213, :] = temp_1
    temp_2 = rng.integers(1, 21, size=(114, 114, 3))
    temp_2 = np.where(temp_2 < 19, 2, 1)
    square_1[71:185, 71:185, :] = temp_2
    temp_3 = rng.integers(1, 21, size=(58, 58, 3))
    temp_3 = np.where(temp_3 < 20, 3, 1)
    square_1[99:157, 99:157, :] = temp_3

    return square_1


def create_square_gauss(rng):
    square_gauss = np.zeros((256, 256, 3)) + 0.001
    square_gauss = add_noise(square_gauss, 10000, 1,
                             noise_type='gaussian', rng=rng)
    square_gauss[42:213, 42:213, :] = 1
    noise_1 = rng.normal(1.001, 0.0001,
                         size=square_gauss[42:213, 42:213, :].shape)
    square_gauss[42:213, 42:213, :] = square_gauss[42:213, 42:213, :] + noise_1
    square_gauss[71:185, 71:185, :] = 2
    noise_2 = rng.normal(2.001, 0.0001,
                         size=square_gauss[71:185, 71:185, :].shape)
    square_gauss[71:185, 71:185, :] = square_gauss[71:185, 71:185, :] + noise_2
    square_gauss[99:157, 99:157, :] = 3
    noise_3 = rng.normal(3.001, 0.0001,
                         size=square_gauss[99:157, 99:157, :].shape)
    square_gauss[99:157, 99:157, :] = square_gauss[99:157, 99:157, :] + noise_3

    return square_gauss


def test_grayscale_image():

    nclasses = 4
    beta = np.float64(0.0)

    image = create_image()
    com = ConstantObservationModel()
    icm = IteratedConditionalModes()

    mu, sigmasq = com.initialize_param_uniform(image, nclasses)
    npt.assert_array_almost_equal(mu, np.array([0., 0.25, 0.5, 0.75]))
    npt.assert_array_almost_equal(sigmasq, np.array([1.0, 1.0, 1.0, 1.0]))

    neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)
    npt.assert_(neglogl[100, 100, 1, 0] != neglogl[100, 100, 1, 1])
    npt.assert_(neglogl[100, 100, 1, 1] != neglogl[100, 100, 1, 2])
    npt.assert_(neglogl[100, 100, 1, 2] != neglogl[100, 100, 1, 3])
    npt.assert_(neglogl[100, 100, 1, 1] != neglogl[100, 100, 1, 3])

    initial_segmentation = icm.initialize_maximum_likelihood(neglogl)
    npt.assert_(initial_segmentation.max() == nclasses - 1)
    npt.assert_(initial_segmentation.min() == 0)

    PLN = icm.prob_neighborhood(initial_segmentation, beta, nclasses)
    print(PLN.shape)
    npt.assert_(np.all((PLN >= 0) & (PLN <= 1.0)))

    if beta == 0.0:
        npt.assert_almost_equal(PLN[50, 50, 1, 0], 0.25, True)
        npt.assert_almost_equal(PLN[50, 50, 1, 1], 0.25, True)
        npt.assert_almost_equal(PLN[50, 50, 1, 2], 0.25, True)
        npt.assert_almost_equal(PLN[50, 50, 1, 3], 0.25, True)
        npt.assert_almost_equal(PLN[147, 129, 1, 0], 0.25, True)
        npt.assert_almost_equal(PLN[147, 129, 1, 1], 0.25, True)
        npt.assert_almost_equal(PLN[147, 129, 1, 2], 0.25, True)
        npt.assert_almost_equal(PLN[147, 129, 1, 3], 0.25, True)
        npt.assert_almost_equal(PLN[61, 152, 1, 0], 0.25, True)
        npt.assert_almost_equal(PLN[61, 152, 1, 1], 0.25, True)
        npt.assert_almost_equal(PLN[61, 152, 1, 2], 0.25, True)
        npt.assert_almost_equal(PLN[61, 152, 1, 3], 0.25, True)
        npt.assert_almost_equal(PLN[100, 100, 1, 0], 0.25, True)
        npt.assert_almost_equal(PLN[100, 100, 1, 1], 0.25, True)
        npt.assert_almost_equal(PLN[100, 100, 1, 2], 0.25, True)
        npt.assert_almost_equal(PLN[100, 100, 1, 3], 0.25, True)

    PLY = com.prob_image(image, nclasses, mu, sigmasq, PLN)
    print(PLY)
    npt.assert_(np.all((PLY >= 0) & (PLY <= 1.0)))

    mu_upd, sigmasq_upd = com.update_param(image, PLY, mu, nclasses)
    print(mu)
    print(mu_upd)
    npt.assert_(mu_upd[0] != mu[0])
    npt.assert_(mu_upd[1] != mu[1])
    npt.assert_(mu_upd[2] != mu[2])
    npt.assert_(mu_upd[3] != mu[3])
    print(sigmasq)
    print(sigmasq_upd)
    npt.assert_(sigmasq_upd[0] != sigmasq[0])
    npt.assert_(sigmasq_upd[1] != sigmasq[1])
    npt.assert_(sigmasq_upd[2] != sigmasq[2])
    npt.assert_(sigmasq_upd[3] != sigmasq[3])

    icm_segmentation, energy = icm.icm_ising(neglogl, beta,
                                             initial_segmentation)
    npt.assert_(np.abs(np.sum(icm_segmentation)) != 0)
    npt.assert_(icm_segmentation.max() == nclasses - 1)
    npt.assert_(icm_segmentation.min() == 0)


def test_grayscale_iter():

    nclasses = 4
    beta = np.float64(0.1)
    max_iter = 15
    background_noise = True

    image = create_image()
    com = ConstantObservationModel()
    icm = IteratedConditionalModes()

    mu, sigmasq = com.initialize_param_uniform(image, nclasses)
    neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)
    initial_segmentation = icm.initialize_maximum_likelihood(neglogl)
    npt.assert_(initial_segmentation.max() == nclasses - 1)
    npt.assert_(initial_segmentation.min() == 0)

    mu, sigmasq = com.seg_stats(image, initial_segmentation, nclasses)
    npt.assert_(mu[0] >= 0.0)
    npt.assert_(mu[1] >= 0.0)
    npt.assert_(mu[2] >= 0.0)
    npt.assert_(mu[3] >= 0.0)
    npt.assert_(sigmasq[0] >= 0.0)
    npt.assert_(sigmasq[1] >= 0.0)
    npt.assert_(sigmasq[2] >= 0.0)
    npt.assert_(sigmasq[3] >= 0.0)

    if background_noise:
        zero = np.zeros_like(image) + 0.001
        zero_noise = add_noise(zero, 10000, 1, noise_type='gaussian')
        image_gauss = np.where(image == 0, zero_noise, image)
    else:
        image_gauss = image

    final_segmentation = np.empty_like(image)
    seg_init = initial_segmentation
    energies = []

    for i in range(max_iter):

        PLN = icm.prob_neighborhood(initial_segmentation, beta,
                                    nclasses)
        npt.assert_(np.all((PLN >= 0) & (PLN <= 1.0)))

        if beta == 0.0:

            npt.assert_almost_equal(PLN[50, 50, 1, 0], 0.25, True)
            npt.assert_almost_equal(PLN[50, 50, 1, 1], 0.25, True)
            npt.assert_almost_equal(PLN[50, 50, 1, 2], 0.25, True)
            npt.assert_almost_equal(PLN[50, 50, 1, 3], 0.25, True)
            npt.assert_almost_equal(PLN[147, 129, 1, 0], 0.25, True)
            npt.assert_almost_equal(PLN[147, 129, 1, 1], 0.25, True)
            npt.assert_almost_equal(PLN[147, 129, 1, 2], 0.25, True)
            npt.assert_almost_equal(PLN[147, 129, 1, 3], 0.25, True)
            npt.assert_almost_equal(PLN[61, 152, 1, 0], 0.25, True)
            npt.assert_almost_equal(PLN[61, 152, 1, 1], 0.25, True)
            npt.assert_almost_equal(PLN[61, 152, 1, 2], 0.25, True)
            npt.assert_almost_equal(PLN[61, 152, 1, 3], 0.25, True)
            npt.assert_almost_equal(PLN[100, 100, 1, 0], 0.25, True)
            npt.assert_almost_equal(PLN[100, 100, 1, 1], 0.25, True)
            npt.assert_almost_equal(PLN[100, 100, 1, 2], 0.25, True)
            npt.assert_almost_equal(PLN[100, 100, 1, 3], 0.25, True)

        PLY = com.prob_image(image_gauss, nclasses, mu, sigmasq, PLN)
        npt.assert_(np.all((PLY >= 0) & (PLY <= 1.0)))
        npt.assert_(PLY[50, 50, 1, 0] > PLY[50, 50, 1, 1])
        npt.assert_(PLY[50, 50, 1, 0] > PLY[50, 50, 1, 2])
        npt.assert_(PLY[50, 50, 1, 0] > PLY[50, 50, 1, 3])
        npt.assert_(PLY[100, 100, 1, 3] > PLY[100, 100, 1, 0])
        npt.assert_(PLY[100, 100, 1, 3] > PLY[100, 100, 1, 1])
        npt.assert_(PLY[100, 100, 1, 3] > PLY[100, 100, 1, 2])

        mu_upd, sigmasq_upd = com.update_param(image_gauss, PLY, mu, nclasses)
        npt.assert_(mu_upd[0] >= 0.0)
        npt.assert_(mu_upd[1] >= 0.0)
        npt.assert_(mu_upd[2] >= 0.0)
        npt.assert_(mu_upd[3] >= 0.0)
        npt.assert_(sigmasq_upd[0] >= 0.0)
        npt.assert_(sigmasq_upd[1] >= 0.0)
        npt.assert_(sigmasq_upd[2] >= 0.0)
        npt.assert_(sigmasq_upd[3] >= 0.0)

        negll = com.negloglikelihood(image_gauss,
                                     mu_upd, sigmasq_upd, nclasses)
        npt.assert_(negll[50, 50, 1, 0] < negll[50, 50, 1, 1])
        npt.assert_(negll[50, 50, 1, 0] < negll[50, 50, 1, 2])
        npt.assert_(negll[50, 50, 1, 0] < negll[50, 50, 1, 3])
        npt.assert_(negll[100, 100, 1, 3] < negll[100, 100, 1, 0])
        npt.assert_(negll[100, 100, 1, 3] < negll[100, 100, 1, 1])
        npt.assert_(negll[100, 100, 1, 3] < negll[100, 100, 1, 2])

        final_segmentation, energy = icm.icm_ising(negll, beta,
                                                   initial_segmentation)
        print(energy[energy > -np.inf].sum())
        energies.append(energy[energy > -np.inf].sum())

        initial_segmentation = final_segmentation
        mu = mu_upd
        sigmasq = sigmasq_upd

    npt.assert_(energies[-1] < energies[0])

    difference_map = np.abs(seg_init - final_segmentation)
    npt.assert_(np.abs(np.sum(difference_map)) != 0)


@set_random_number_generator()
def test_square_iter(rng):

    nclasses = 4
    beta = np.float64(0.0)
    max_iter = 10

    com = ConstantObservationModel()
    icm = IteratedConditionalModes()

    initial_segmentation = create_square()
    square_gauss = create_square_gauss(rng)

    mu, sigmasq = com.seg_stats(square_gauss, initial_segmentation,
                                nclasses)
    npt.assert_(mu[0] >= 0.0)
    npt.assert_(mu[1] >= 0.0)
    npt.assert_(mu[2] >= 0.0)
    npt.assert_(mu[3] >= 0.0)
    npt.assert_(sigmasq[0] >= 0.0)
    npt.assert_(sigmasq[1] >= 0.0)
    npt.assert_(sigmasq[2] >= 0.0)
    npt.assert_(sigmasq[3] >= 0.0)

    final_segmentation = np.empty_like(square_gauss)
    seg_init = initial_segmentation
    energies = []

    for i in range(max_iter):

        print('\n')
        print('>> Iteration: ' + str(i))
        print('\n')

        PLN = icm.prob_neighborhood(initial_segmentation, beta,
                                    nclasses)
        npt.assert_(np.all((PLN >= 0) & (PLN <= 1.0)))

        if beta == 0.0:

            npt.assert_(PLN[25, 25, 1, 0] == 0.25)
            npt.assert_(PLN[25, 25, 1, 1] == 0.25)
            npt.assert_(PLN[25, 25, 1, 2] == 0.25)
            npt.assert_(PLN[25, 25, 1, 3] == 0.25)
            npt.assert_(PLN[50, 50, 1, 0] == 0.25)
            npt.assert_(PLN[50, 50, 1, 1] == 0.25)
            npt.assert_(PLN[50, 50, 1, 2] == 0.25)
            npt.assert_(PLN[50, 50, 1, 3] == 0.25)
            npt.assert_(PLN[90, 90, 1, 0] == 0.25)
            npt.assert_(PLN[90, 90, 1, 1] == 0.25)
            npt.assert_(PLN[90, 90, 1, 2] == 0.25)
            npt.assert_(PLN[90, 90, 1, 3] == 0.25)
            npt.assert_(PLN[125, 125, 1, 0] == 0.25)
            npt.assert_(PLN[125, 125, 1, 1] == 0.25)
            npt.assert_(PLN[125, 125, 1, 2] == 0.25)
            npt.assert_(PLN[125, 125, 1, 3] == 0.25)

        PLY = com.prob_image(square_gauss, nclasses, mu, sigmasq, PLN)
        npt.assert_(np.all((PLY >= 0) & (PLY <= 1.0)))
        npt.assert_(PLY[25, 25, 1, 0] > PLY[25, 25, 1, 1])
        npt.assert_(PLY[25, 25, 1, 0] > PLY[25, 25, 1, 2])
        npt.assert_(PLY[25, 25, 1, 0] > PLY[25, 25, 1, 3])
        npt.assert_(PLY[125, 125, 1, 3] > PLY[125, 125, 1, 0])
        npt.assert_(PLY[125, 125, 1, 3] > PLY[125, 125, 1, 1])
        npt.assert_(PLY[125, 125, 1, 3] > PLY[125, 125, 1, 2])

        mu_upd, sigmasq_upd = com.update_param(square_gauss, PLY, mu, nclasses)
        npt.assert_(mu_upd[0] >= 0.0)
        npt.assert_(mu_upd[1] >= 0.0)
        npt.assert_(mu_upd[2] >= 0.0)
        npt.assert_(mu_upd[3] >= 0.0)
        npt.assert_(sigmasq_upd[0] >= 0.0)
        npt.assert_(sigmasq_upd[1] >= 0.0)
        npt.assert_(sigmasq_upd[2] >= 0.0)
        npt.assert_(sigmasq_upd[3] >= 0.0)

        negll = com.negloglikelihood(square_gauss,
                                     mu_upd, sigmasq_upd, nclasses)
        npt.assert_(negll[25, 25, 1, 0] < negll[25, 25, 1, 1])
        npt.assert_(negll[25, 25, 1, 0] < negll[25, 25, 1, 2])
        npt.assert_(negll[25, 25, 1, 0] < negll[25, 25, 1, 3])
        npt.assert_(negll[100, 100, 1, 3] < negll[125, 125, 1, 0])
        npt.assert_(negll[100, 100, 1, 3] < negll[125, 125, 1, 1])
        npt.assert_(negll[100, 100, 1, 3] < negll[125, 125, 1, 2])

        final_segmentation, energy = icm.icm_ising(negll, beta,
                                                   initial_segmentation)

        energies.append(energy[energy > -np.inf].sum())

        initial_segmentation = final_segmentation
        mu = mu_upd
        sigmasq = sigmasq_upd

    difference_map = np.abs(seg_init - final_segmentation)
    npt.assert_(np.abs(np.sum(difference_map)) == 0.0)


@set_random_number_generator()
def test_icm_square(rng):

    nclasses = 4
    max_iter = 10

    com = ConstantObservationModel()
    icm = IteratedConditionalModes()

    initial_segmentation = create_square()
    square_1 = create_square_uniform(rng)

    mu, sigma = com.seg_stats(square_1, initial_segmentation,
                              nclasses)
    sigmasq = sigma ** 2
    npt.assert_(mu[0] >= 0.0)
    npt.assert_(mu[1] >= 0.0)
    npt.assert_(mu[2] >= 0.0)
    npt.assert_(mu[3] >= 0.0)
    npt.assert_(sigmasq[0] >= 0.0)
    npt.assert_(sigmasq[1] >= 0.0)
    npt.assert_(sigmasq[2] >= 0.0)
    npt.assert_(sigmasq[3] >= 0.0)

    negll = com.negloglikelihood(square_1, mu, sigmasq, nclasses)

    final_segmentation_1 = np.empty_like(square_1)
    final_segmentation_2 = np.empty_like(square_1)

    beta = 0.0

    for i in range(max_iter):

        print('\n')
        print('>> Iteration: ' + str(i))
        print('\n')

        final_segmentation_1, energy_1 = icm.icm_ising(negll, beta,
                                                       initial_segmentation)
        initial_segmentation = final_segmentation_1

    beta = 2
    initial_segmentation = create_square()

    for j in range(max_iter):

        print('\n')
        print('>> Iteration: ' + str(j))
        print('\n')

        final_segmentation_2, energy_2 = icm.icm_ising(negll, beta,
                                                       initial_segmentation)
        initial_segmentation = final_segmentation_2

    difference_map = np.abs(final_segmentation_1 - final_segmentation_2)
    npt.assert_(np.abs(np.sum(difference_map)) != 0)


def test_classify():

    imgseg = TissueClassifierHMRF()

    nclasses = 4
    beta = 0.1
    tolerance = 0.0001
    max_iter = 10

    image = create_image()

    npt.assert_(image.max() == 1.0)
    npt.assert_(image.min() == 0.0)

    # First we test without setting iterations and tolerance
    seg_init, seg_final, PVE = imgseg.classify(image, nclasses, beta)

    npt.assert_(seg_final.max() == nclasses)
    npt.assert_(seg_final.min() == 0.0)

    # Second we test it with just changing the tolerance
    seg_init, seg_final, PVE = imgseg.classify(image, nclasses, beta,
                                               tolerance)

    npt.assert_(seg_final.max() == nclasses)
    npt.assert_(seg_final.min() == 0.0)

    # Third we test it with just the iterations
    seg_init, seg_final, PVE = imgseg.classify(image, nclasses, beta, max_iter)

    npt.assert_(seg_final.max() == nclasses)
    npt.assert_(seg_final.min() == 0.0)

    # Next we test saving the history of accumulated energies from ICM
    imgseg = TissueClassifierHMRF(save_history=True)

    seg_init, seg_final, PVE = imgseg.classify(200 * image, nclasses,
                                               beta, tolerance)

    npt.assert_(seg_final.max() == nclasses)
    npt.assert_(seg_final.min() == 0.0)

    npt.assert_(imgseg.energies_sum[0] > imgseg.energies_sum[-1])
