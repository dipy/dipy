import numpy as np
import math
import scipy
import scipy.stats
from scipy.fftpack import fft, ifft
import scipy.interpolate
import nibabel as nib
import matplotlib.pyplot as plt
from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import median_otsu
import scipy.ndimage as ndimage
from scipy.ndimage import zoom
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from dipy.core.ndindex import ndindex
from dipy.bias import correction

test_subject = np.zeros(shape=(100, 100, 100),)

for i in np.array(range(100)):
    for j in np.array(range(100)):
        for k in np.array(range(100)):
            test_subject[i,j,k] = 50
            if(k > 30 and k < 70 and j > 30 and j < 70 and i > 30 and i < 70):
                test_subject[i, j, k] = 100
                if(k > 40 and k < 60 and j > 40 and j < 60 and i > 40 and i < 60):
                    test_subject[i, j, k] = 200

test_subject_adding_noise = np.zeros(shape=(100, 100, 100),)

for i in np.array(range(100)):
    for j in np.array(range(100)):
        for k in np.array(range(100)):
            test_subject_adding_noise[i,j,k] = 50
            if(k > 30 and k < 70 and j > 30 and j < 70 and i > 30 and i < 70):
                test_subject_adding_noise[i, j, k] = 100
                if(k > 40 and k < 60 and j > 40 and j < 60 and i > 40 and i < 60):
                    test_subject_adding_noise[i, j, k] = 200
for i in np.array(range(100)):
    for j in np.array(range(100)):
        for k in np.array(range(100)):
            if(k == 60 and i == 60 and j == 60):
                test_subject_adding_noise[i, j, k] = test_subject_adding_noise[i,j,k] + np.random.normal(60,1,1)
            if(k == 40 and i == 40 and j == 40):
                test_subject_adding_noise[i, j, k] = test_subject_adding_noise[i,j,k] - np.random.normal(30,1,1)

logUncorrectedImage = np.log(test_subject_adding_noise)

sharpenedimg = correction.SharpenImage(logUncorrectedImage)
