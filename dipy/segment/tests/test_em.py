import numpy as np
import numpy.testing as npt
from dipy.data import get_data
from dipy.sims.voxel import add_noise
from dipy.segment.mrf_map import (FASTImageSegmenter,
                                  initialize_constant_models_uniform,
                                  neg_log_likelihood_gaussian,
                                  initialize_maximum_likelihood,
                                  iterate_icm_ising)
import matplotlib.pyplot as plt
from dipy.segment.icm_segmenter import icm
from dipy.segment.energy_mrf import (total_energy, neg_log_likelihood,
                                     gibbs_energy)
# from dipy.segment.rois_stats import seg_stats
# from dipy.segment.mrf_em import prob_neigh, prob_image, update_param

# Load a coronal slice from a T1-weighted MRI
fname = get_data('t1_coronal_slice')
single_slice = np.load(fname)

# Stack a few copies to form a 3D volume
nslices = 5
image = np.zeros(shape=single_slice.shape + (nslices,))
image[..., :nslices] = single_slice[..., None]

# Execute the segmentation
num_classes = 6
beta = 0.1
max_iter = 2

# dname = '/Users/jvillalo/Documents/GSoC_2015/Code/Data/T1_coronal/'
# dname = '/home/eleftherios/Dropbox/DIPY_GSoC_2015/T1_coronal/'
# img = nib.load(dname + 't1_coronal_stack.nii.gz')
# dataimg = img.get_data()
# mask = nib.load(dname + 't1mask_coronal_stack.nii.gz')
# datamask = mask.get_data()
# seg = nib.load(dname + 't1seg_coronal_stack.nii.gz')
# seg_init = seg.get_data()
# ones = np.ones_like(dataimg)
# masked_ones = applymask(ones, datamask)
# masked_img = applymask(dataimg, datamask)
# seg_init_masked = applymask(seg_init, datamask)

square = np.zeros((256, 256, 3))
square[42:213, 42:213, :] = 3
square[71:185, 71:185, :] = 2
square[99:157, 99:157, :] = 1

square_mask = np.zeros((256, 256, 3))
square_mask[42:213, 42:213, :] = 1
square_masked = applymask(square, square_mask)

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
A_masked = applymask(A, square_mask)

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
B_masked = applymask(B, square_mask)

square_gauss = add_noise(square, 4, 1, noise_type='gaussian')
square_gauss_masked = applymask(square_gauss, square_mask)


def test_icm():

    classes = 3

    # testing with original T1
    mu = np.array([0.44230444, 0.64952929, 0.78890026])
    var = np.array([0.01061197, 0.00227377, 0.00127969])

    seg1, totale1 = icm(mu, var, masked_img, seg_init_masked, classes,
                        beta=1.5)

    # Testing with image of ones
    mu = np.array([1., 1., 1.])
    var = np.array([0.0, 0.0, 0.0])
    seg2, totale2 = icm(mu, var, masked_ones, seg_init_masked, classes,
                        beta=1.5)

    plt.figure()
    plt.imshow(seg1[:, :, 1])
    plt.figure()
    plt.imshow(seg2[:, :, 1])
    plt.figure()
    plt.imshow(totale1[:, :, 1])
    plt.figure()
    plt.imshow(totale2[:, :, 1])

    npt.assert_array_almost_equal(seg1, seg1)
    npt.assert_equal(np.sum(np.abs(seg1 - seg2)) > 0, True)
    npt.assert_equal(np.sum(np.abs(seg1 - seg_init_masked)) > 0, True)
    npt.assert_equal(np.sum(np.abs(totale1 - totale2)) > 0, True)

    # Testing with initital segmentation
    mu = np.array([1, 2, 3])
    var = np.array([0.0, 0.0, 0.0])
    seg3, totale3 = icm(mu, var, seg_init_masked, seg_init_masked, classes,
                        beta=1.5)
    plt.figure()
    plt.imshow(seg3[:, :, 1])
    plt.figure()
    plt.imshow(np.abs(seg_init_masked[:, :, 1] - seg3[:, :, 1]))
    # Should be different!!
    npt.assert_equal(np.sum(np.abs(seg3 - seg_init_masked)) > 0, True)

    # Testing with initital segmentation with beta=0
    seg3_0, totale3_0 = icm(mu, var, seg_init_masked, seg_init_masked, classes,
                            beta=0.0)
    plt.figure()
    plt.imshow(np.abs(seg_init_masked[:, :, 1] - seg3_0[:, :, 1]))
    # Input and Output should be exactly the same
#    npt.assert_equal(np.sum(np.abs(seg3_0 - seg_init_masked)) == 0, True)
#    npt.assert_array_almost_equal(seg3_0, seg_init_masked)

    # Testing with square
    seg4, totale4 = icm(mu, var, square_masked, square_masked, classes,
                        beta=1.5)
    npt.assert_array_almost_equal(seg4, square_masked)
    npt.assert_array_equal(seg4, square_masked)

#    plt.figure()
#    plt.imshow(square_masked[:, :, 1])
#    plt.figure()
#    plt.imshow(seg4[:, :, 1])
#    plt.figure()
#    plt.imshow(np.abs(square_masked[:, :, 1] - seg4[:, :, 1]))

    # Testing with square beta=0. Should be exactly the same.
    seg4_0, totale4_0 = icm(mu, var, square_masked, square_masked, classes,
                            beta=0.0)
#    npt.assert_array_almost_equal(seg4_0, square_masked)
#    npt.assert_array_equal(seg4_0, square_masked)

    # Testing with square with huge beta. Should smooth the output
    seg4_huge, totale4_huge = icm(mu, var, square_masked, square_masked,
                                  classes, beta=100000000000000.)
    npt.assert_equal(np.sum(np.abs(seg4_huge - seg_init_masked)) > 0, True)
    plt.figure()
    plt.imshow(seg4_huge[:, :, 1])

    # Testing with noisy square
    seg5, totale5 = icm(mu, var, A_masked, square_masked, classes,
                        beta=1.5)
    npt.assert_array_almost_equal(seg5, square_masked)
    npt.assert_array_equal(seg5, square_masked)

#    plt.figure()
#    plt.imshow(A_masked[:, :, 1])
#    plt.figure()
#    plt.imshow(seg5[:, :, 1])
#    plt.figure()
#    plt.imshow(np.abs(square_masked[:, :, 1] - seg5[:, :, 1]))

    # Testing with another noisy square-three different values in a region
    seg6, totale6 = icm(mu, var, B_masked, square_masked, classes,
                        beta=1.5)
    npt.assert_array_almost_equal(seg6, square_masked)
    npt.assert_array_equal(seg6, square_masked)

#    plt.figure()
#    plt.imshow(B_masked[:, :, 1])
#    plt.figure()
#    plt.imshow(seg6[:, :, 1])
#    plt.figure()
#    plt.imshow(np.abs(square_masked[:, :, 1] - seg6[:, :, 1]))

    # Testing with gaussian noisy square
    seg7, totale7 = icm(mu, var, square_gauss_masked, square_masked, classes,
                        beta=1.5)
    npt.assert_array_almost_equal(seg7, square_masked)

#    plt.figure()
#    plt.imshow(square_gauss_masked[:, :, 1])
#    plt.figure()
#    plt.imshow(seg7[:, :, 1])
#    plt.figure()
#    plt.imshow(np.abs(square_masked[:, :, 1] - seg7[:, :, 1]))

    mu = np.array([0.44230444, 0.64952929, 0.78890026])
    var = np.array([0.01061197, 0.00227377, 0.00127969])
    seg8, totale8 = icm(mu, var, masked_img, seg_init_masked, classes, beta=15)
    npt.assert_equal(np.sum(np.abs(seg8 - seg_init_masked)) > 0, True)
    npt.assert_equal(np.sum(np.abs(seg1 - seg8)) > 0, True)
    plt.figure()
    plt.imshow(seg_init_masked[:, :, 1])
    plt.figure()
    plt.imshow(seg8[:, :, 1])
    plt.figure()
    plt.imshow(np.abs(seg_init_masked[:, :, 1] - seg8[:, :, 1]))

    mu = np.array([1, 2, 3])
    var = np.array([0.0, 0.0, 0.0])
    seg9, totale9 = icm(mu, var, square_masked, square_masked, classes,
                        beta=15)
    npt.assert_array_almost_equal(seg9, square_masked)
    npt.assert_array_equal(seg9, square_masked)
    plt.figure()
    plt.imshow(seg9[:, :, 1])
    plt.figure()
    plt.imshow(np.abs(square_masked[:, :, 1] - seg9[:, :, 1]))

    seg10, totale10 = icm(mu, var, masked_img, seg_init_masked, classes,
                          beta=0.015)
    npt.assert_equal(np.sum(np.abs(seg10 - seg_init_masked)) > 0, True)
    npt.assert_equal(np.sum(np.abs(seg1 - seg10)) > 0, True)
    plt.figure()
    plt.imshow(seg_init_masked[:, :, 1])
    plt.figure()
    plt.imshow(seg10[:, :, 1])
    plt.figure()
    plt.imshow(np.abs(seg_init_masked[:, :, 1] - seg10[:, :, 1]))

    mu = np.array([1, 2, 3])
    var = np.array([0.0, 0.0, 0.0])
    seg11, totale11 = icm(mu, var, square_masked, square_masked, classes,
                          beta=0.015)
    npt.assert_array_almost_equal(seg11, square_masked)
    npt.assert_array_equal(seg11, square_masked)
    plt.figure()
    plt.imshow(seg11[:, :, 1])
    plt.figure()
    plt.imshow(np.abs(square_masked[:, :, 1] - seg11[:, :, 1]))


def test_total_energy():

    img = masked_img.copy(order='C')
    masked_img_pad = add_padding_reflection(img, 1)
    seg_img = seg_init_masked.copy(order='C')
    seg_img_pad = add_padding_reflection(seg_img, 1)

    mu = np.array([0.44230444, 0.64952929, 0.78890026])
    var = np.array([0.01061197, 0.00227377, 0.00127969])
    index = (125, 125, 2)
    beta = 1.5

    label = 0
    energy1 = total_energy(masked_img_pad, seg_img_pad, mu, var, index, label,
                           beta)
    print('energy1: ', energy1)

    label = 1
    energy2 = total_energy(masked_img_pad, seg_img_pad, mu, var, index, label,
                           beta)
    print('energy2: ', energy2)

    label = 2
    energy3 = total_energy(masked_img_pad, seg_img_pad, mu, var, index, label,
                           beta)
    print('energy3: ', energy3)

    npt.assert_equal((energy1 != energy2), True)
    npt.assert_equal((energy1 != energy3), True)
    npt.assert_equal((energy2 != energy3), True)

    img = square_masked.copy(order='C')
    square_masked_pad = add_padding_reflection(img, 1)
    mu = np.array([1., 2., 3.])
    var = np.array([0.0, 0.0, 0.0])
    index = (200, 125, 2)
    print(square_masked_pad[200, 125, 2])

    label = 0
    energy1_square = total_energy(square_masked_pad, square_masked_pad, mu,
                                  var, index, label, beta)
    print('energy1_square: ', energy1_square)

    label = 1
    energy2_square = total_energy(square_masked_pad, square_masked_pad, mu,
                                  var, index, label, beta)
    print('energy2_square: ', energy2_square)

    label = 2
    energy3_square = total_energy(square_masked_pad, square_masked_pad, mu,
                                  var, index, label, beta)
    print('energy3_square: ', energy3_square)

#    npt.assert_equal((energy1_square != energy2_square), True)
    npt.assert_equal((energy1_square != energy3_square), True)
    npt.assert_equal((energy2_square != energy3_square), True)


def test_neg_log_likelihood():

    mu = np.array([0.44230444, 0.64952929, 0.78890026])
    var = np.array([0.01061197, 0.00227377, 0.00127969])
    index = (150, 125, 1)
    print(masked_img[150, 125, 1])
    label = 0

    loglike1 = neg_log_likelihood(masked_img, mu, var, index, label)

    label = 1

    loglike2 = neg_log_likelihood(masked_img, mu, var, index, label)

    npt.assert_equal((loglike1 != loglike2), True)

    mu = np.array([1., 2., 3.])
    var = np.array([0.0, 0.0, 0.0])
    index = (200, 125, 2)
    print(square_masked[200, 125, 2])

    label = 0

    loglike1_square = neg_log_likelihood(square_masked, mu, var, index, label)

    label = 1

    loglike2_square = neg_log_likelihood(square_masked, mu, var, index, label)

 #   npt.assert_equal((loglike1_square != loglike2_square), True)


def test_gibbs_energy():

    index = (160, 105, 2)
    print('index1: ', seg_init_masked[160, 105, 2])  # This is 1
    index = (160, 105, 2)
    print('nei1: ', seg_init_masked[index[0] - 1, index[1], index[2]])
    print('nei2: ', seg_init_masked[index[0] + 1, index[1], index[2]])
    print('nei3: ', seg_init_masked[index[0], index[1] - 1, index[2]])
    print('nei4: ', seg_init_masked[index[0], index[1] + 1, index[2]])
    print('nei5: ', seg_init_masked[index[0], index[1], index[2] - 1])
    print('nei6: ', seg_init_masked[index[0], index[1], index[2] + 1])

    label = 0  # if the label matches the voxel value it does not add
                # label needs to be 0,1,2. The function adds +1 to the given
                # number

    gibbs1 = gibbs_energy(seg_init_masked, index, label, beta=1)
    print('gibbs1: ', gibbs1)

    gibbs2 = gibbs_energy(seg_init_masked, index, label, beta=0)
    print('gibb2s: ', gibbs2)

#    npt.assert_equal(np.abs(gibbs2) > np.abs(gibbs1), True)



#    sqimg = square_masked.copy(order='C')
#    sqimg_pad = add_padding_reflection(sqimg, 1)
#    index = (150, 125, 1)
#    print(sqimg_pad[150, 125, 1])
#
#    label = 1
#
#    beta = 1.5
#
#    gibbs1 = gibbs_energy(sqimg_pad, index, label, beta)
#
#    beta = 2
#
#    gibbs2 = gibbs_energy(sqimg_pad, index, label, beta)
#
#    npt.assert_equal(np.abs(gibbs2) > np.abs(gibbs1), True)

#def test_ising():
#
#    l = 1
#    vox = 2
#    beta = 20
#
#    npt.assert_equal(ising(l, vox, beta), beta)
#
#    l = 1
#    vox = 1
#    beta = 20
#
#    npt.assert_equal(ising(l, vox, beta), - beta)
#
#
#def test_seg_stats():
#
#    nclass = 3
#
#    Mu, Std, Var = seg_stats(masked_img, seg_init_masked, nclass)
#    print(Mu)
#    print(Std)
#    print(Var)
#
#    Mu, Std, Var = seg_stats(seg_init_masked, seg_init_masked, nclass)
#    print(Mu)
#    print(Std)
#    print(Var)


#def test_emfuncs():
#
#    seg_img = seg_init_masked.copy(order='C')
#    seg_img_pad = add_padding_reflection(seg_img, 1)
#    nclass = 3
#    beta = 1.5
#
#    # Testing prob_neigh
#    PLN1 = prob_neigh(nclass, masked_img, seg_img_pad, beta)
#
#    beta = 1.5
#
#    PLN2 = prob_neigh(nclass, masked_img, seg_img_pad, beta)
#
#    npt.assert_array_equal(np.sum(np.abs(PLN1 - PLN2)) > 0, True)
#
#    # Testing prob_image with current estimates of mu and var
#    mu = np.array([0.44230444, 0.64952929, 0.78890026])
#    var = np.array([0.01061197, 0.00227377, 0.00127969])
#
#    PLY1 = prob_image(nclass, masked_img, mu, var, PLN1)
#
#    mu_update, var_update = update_param(nclass, masked_img, datamask, mu,
#                                         PLY1)
#
#    npt.assert_array_equal(np.sum(np.abs(mu - mu_update)) > 0, True)
#    npt.assert_array_equal(np.sum(np.abs(var - var_update)) > 0, True)
#
##    npt.assert_raises
##    np.assert_equal(value, np.inf)

if __name__ == '__main__':

#    test_icm()
#    test_total_energy()
#    test_neg_log_likelihood()
    test_gibbs_energy()
#    test_ising()

    npt.run_module_suite()
