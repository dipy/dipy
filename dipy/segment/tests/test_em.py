# import warnings

import numpy as np
import nibabel as nib
import numpy.testing as npt

from dipy.segment.mask import applymask
# from dipy.core.ndindex import ndindex
# from dipy.data import get_data

from dipy.segment.icm_map import icm
from dipy.segment.energy_mrf import (total_energy, neg_log_likelihood,
                                     gibbs_energy, ising)

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
    #img = masked_img
    seg_img = seg_init_masked
    classes = 3
    beta = 1.5
    
    seg1 = icm(mu, var, masked_img, seg_img, classes, beta)    
    
    mu = np.array([0.44230444, 0.64952929, 0.78890026])
    var = np.array([0.01061197, 0.00227377, 0.00127969])
    #img = masked_ones
    seg_img = seg_init_masked
    classes = 3
    beta = 1.5
    
    seg2 = icm(mu, var, masked_ones, seg_img, classes, beta)
    
    npt.assert_array_almost_equal(seg1, seg2)


def test_total_energy():

    image = masked_img
    segmentation = seg_init_masked
    mu = np.array([0.44230444, 0.64952929, 0.78890026])
    var = np.array([0.01061197, 0.00227377, 0.00127969])
    index = (150, 125, 1)
    label = 0
    beta = 1.5

    npt.assert_equal(total_energy(image, segmentation,
                     mu, var, index, label, beta), beta)

    image = masked_ones
    segmentation = seg_init_masked
    mu = np.array([0.44230444, 0.64952929, 0.78890026])
    var = np.array([0.01061197, 0.00227377, 0.00127969])
    index = (150, 125, 1)
    label = 0
    beta = 1.5

    npt.assert_equal(total_energy(image, segmentation,
                     mu, var, index, label, beta), beta)


def test_neg_log_likelihood():

    img = masked_img
    mu = np.array([0.44230444, 0.64952929, 0.78890026])
    var = np.array([0.01061197, 0.00227377, 0.00127969])
    index = (150, 125, 1)
    label = 0

    npt.assert_equal(neg_log_likelihood(img, mu, var, index, label), mu)

    img = masked_ones
    mu = np.array([0.44230444, 0.64952929, 0.78890026])
    var = np.array([0.01061197, 0.00227377, 0.00127969])
    index = (150, 125, 1)
    label = 0

    npt.assert_equal(neg_log_likelihood(img, mu, var, index, label), mu)


def test_gibbs_energy():

    seg = seg_init_masked
    index = (150, 125, 1)
    label = 0
    beta = 1.5

    npt.assert_equal(gibbs_energy(seg, index, label, beta), beta)

    seg = masked_ones
    index = (150, 125, 1)
    label = 0
    beta = 1.5

    npt.assert_equal(gibbs_energy(seg, index, label, beta), beta)


def test_ising():

    l = 1
    vox = 2
    beta = 20

    npt.assert_equal(ising(l, vox, beta), beta)

    l = 1
    vox = 1
    beta = 20

    npt.assert_equal(ising(l, vox, beta), - beta)


if __name__ == '__main__':

    test_ising()
    npt.run_module_suite()
