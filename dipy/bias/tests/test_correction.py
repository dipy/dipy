import numpy as np
import math
import scipy
import scipy.stats
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
from dipy.bias import bias_correction
from scipy.stats import multivariate_normal
from Goal0_1 import show_volume

test_subject = np.zeros(shape=(100, 100, 100),)

for i in np.array(range(100)):
    for j in np.array(range(100)):
        for k in np.array(range(100)):
            test_subject[i,j,k] = 50
#            if(k > 10 and k < 90 and j > 10 and j < 90 and i > 10 and i < 90):
#                test_subject[i, j, k] = 100
#                if(k > 30 and k < 70 and j > 30 and j < 70 and i > 30 and i < 70):
#                    test_subject[i, j, k] = 200

"""Three dimensional gaussian kernel
"""
"""
noise = [[[1/16, 1/6, 1/16],
          [1/8, 1/4, 1/8],
          [1/16, 1/8, 1/16]],
         [[1/8, 1/3, 1/8],
          [1/4, 1/2, 1/4],
          [1/8, 1/4, 1/8]],
         [[1/16, 1/6, 1/16],
          [1/8, 1/4, 1/8],
          [1/16, 1/8, 1/16]]]
"""
noise = np.zeros((5, 5, 5))
noise[3, 3, 3] = 0.49
noise[3, 3, 2] = 0.49

noise = zoom(noise, (160/5, 239/5, 200/5))


#test_subject_adding_noise = np.log(test_subject)

#test_subject_adding_noise = test_subject_adding_noise + noise

#test_subject_adding_noise = np.exp(test_subject_adding_noise)

dname = "/Users/tiwanyan/ANTs/Images/N3_correction/"
ft1 = dname + "Q_0011_T1_N3.nii.gz"
t1 = nib.load(ft1).get_data()

t2 = np.exp(np.log(t1) + noise)
#Correct = NP_correction(t1)
#Correct.optimization_converge()
#Correct.field_smoothing()
"""
for i in np.array(range(100)):
    for j in np.array(range(100)):
        for k in np.array(range(100)):
            test_subject_adding_noise = test_subject_adding_noise + y.pdf([i,j,k]) * 10000
"""
#test_subject_adding_noise = np.exp(test_subject_adding_noise)

#logUncorrectedImage = np.log(test_subject_adding_noise)

#sharpenedimg = correction.SharpenImage(logUncorrectedImage)
