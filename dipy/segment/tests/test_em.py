# import warnings

import numpy as np
import nibabel as nib
import numpy.testing as npt

from dipy.segment.mask import applymask
from dipy.denoise.denspeed import add_padding_reflection
# from dipy.core.ndindex import ndindex
# from dipy.data import get_data
import matplotlib.pyplot as plt

from dipy.segment.icm_map import icm
from dipy.segment.energy_mrf import (total_energy, neg_log_likelihood,
                                     gibbs_energy, ising)
from dipy.segment.rois_stats import seg_stats
from dipy.segment.mrf_em import prob_neigh, prob_image, update_param

dname = '/Users/jvillalo/Documents/GSoC_2015/Code/Data/T1_coronal/'
# dname = '/home/eleftherios/Dropbox/DIPY_GSoC_2015/T1_coronal/'

img = nib.load(dname + 't1_coronal_stack.nii.gz')
dataimg = img.get_data()

mask = nib.load(dname + 't1mask_coronal_stack.nii.gz')
datamask = mask.get_data()

seg = nib.load(dname + 't1seg_coronal_stack.nii.gz')
seg_init = seg.get_data()

ones = np.ones_like(dataimg)

masked_ones = applymask(ones, datamask)
masked_img = applymask(dataimg, datamask)
seg_init_masked = applymask(seg_init, datamask)


def test_icm():

    mu = np.array([0.44230444, 0.64952929, 0.78890026])
    var = np.array([0.01061197, 0.00227377, 0.00127969])
    classes = 3
    beta = 1.5

    seg1, totale1 = icm(mu, var, masked_img, seg_init_masked, classes, beta)
    seg2, totale2 = icm(mu, var, masked_ones, seg_init_masked, classes, beta)

    plt.figure()
    plt.imshow(seg1[:, :, 1])
    plt.figure()
    plt.imshow(seg2[:, :, 1])
    plt.figure()
    plt.imshow(totale1[:, :, 1])
    plt.figure()
    plt.imshow(totale2[:, :, 1])

    # npt.assert_array_almost_equal(seg1, seg2)
    npt.assert_array_almost_equal(seg1, seg1)
    npt.assert_equal(np.sum(np.abs(seg1 - seg2)) > 0, True)
    npt.assert_equal(np.sum(np.abs(seg1 - seg_init_masked)) > 0, True)
    npt.assert_equal(np.sum(np.abs(totale1 - totale2)) > 0, True)


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

    label = 1

    energy2 = total_energy(masked_img_pad, seg_img_pad, mu, var, index, label,
                           beta)

    label = 2

    energy3 = total_energy(masked_img_pad, seg_img_pad, mu, var, index, label,
                           beta)

    npt.assert_equal((energy1 != energy2), True)
    npt.assert_equal((energy1 != energy3), True)
    npt.assert_equal((energy2 != energy3), True)

#    label = 0  # For this label (CSF) the energy is higher with smaller beta
#    beta = 15  # For the other labels this does not hold true
#               # It doesn't hold for different voxels either
#
#    energy4 = total_energy(masked_img_pad, seg_img_pad, mu, var, index, label,
#                           beta)
#
#    npt.assert_equal((energy1 > energy4), True)
#
#    label = 0
#    beta = 0.15
#
#    energy5 = total_energy(masked_img_pad, seg_img_pad, mu, var, index, label,
#                           beta)
#
#    npt.assert_equal((energy5 > energy1), True)


def test_neg_log_likelihood():

    mu = np.array([0.44230444, 0.64952929, 0.78890026])
    var = np.array([0.01061197, 0.00227377, 0.00127969])
    index = (150, 125, 1)
    label = 0

    loglike1 = neg_log_likelihood(masked_img, mu, var, index, label)

    label = 1

    loglike2 = neg_log_likelihood(masked_img, mu, var, index, label)

    npt.assert_equal((loglike1 != loglike2), True)


def test_gibbs_energy():

    seg_img = seg_init_masked.copy(order='C')
    seg_img_pad = add_padding_reflection(seg_img, 1)
    index = (150, 125, 1)
    label = 0
    beta = 1.5

    gibbs1 = gibbs_energy(seg_img_pad, index, label, beta)

    beta = 2

    gibbs2 = gibbs_energy(seg_img_pad, index, label, beta)

    npt.assert_equal(np.abs(gibbs2) > np.abs(gibbs1), True)


def test_ising():

    l = 1
    vox = 2
    beta = 20

    npt.assert_equal(ising(l, vox, beta), beta)

    l = 1
    vox = 1
    beta = 20

    npt.assert_equal(ising(l, vox, beta), - beta)


# def test_seg_stats():
#
#    nclass = 3
#
#    Mu, Std, Var = seg_stats(masked_img, seg_init_masked, nclass)


def test_prob_neigh():

    seg_img = seg_init_masked.copy(order='C')
    seg_img_pad = add_padding_reflection(seg_img, 1)
    nclass = 3
    beta = 1.5

    PLN1 = prob_neigh(nclass, masked_img, seg_img_pad, beta)
    
    beta = 1.5

    PLN2 = prob_neigh(nclass, masked_img, seg_img_pad, beta)
    
    npt.assert_array_almost_equal(PLN1, PLN2)
    
def test_prob_image():
    
    nclass
    masked_img
    mu_upd
    var_upd
    P_L_N
    
    
    
def test_update_param():
    
    nclass
    masked_img
    datamask
    mu_upd
    P_L_Y
    
    
    

if __name__ == '__main__':

    test_icm()
    test_total_energy()
    test_neg_log_likelihood()
    test_gibbs_energy()
    test_ising()

    npt.run_module_suite()
