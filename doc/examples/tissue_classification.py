"""
=======================================================
Tissue Classification for a T1-weighted Strutural Image
=======================================================
This example explains how to segment a T1-weighted structural image using a MRF
approach. Similar algorithms have been proposed by Zhang et al. [Zhang2001]_,
and Avants et al. [Avants2011]_ available in FAST-FSL and ANTS-Artropos,
respectively.

Here we will use a T1-weighted image, that has been previously skull-stripped
and bias field corrected.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from dipy.data import get_data
from dipy.segment.tissue import TissueClassifierHMRF

"""
First we fetch the T1 volume from the Syn dataset
"""

img = nib.load('/Users/jvillalo/.dipy/syn_test/t1_brain.nii.gz')
t1_img = img.get_data()
print('t1_img.shape (%d, %d, %d)' % t1_img.shape)

#from dipy.data.fetcher import fetch_syn_data, read_syn_data
#fetch_syn_data()
#nib_syn_t1, nib_syn_b0 = read_syn_data()
#syn_b0 = np.array(nib_syn_b0.get_data())

"""
Now we will define the other three parameters for the segmentation algorith

First, the number of tissue classes
"""

nclass = 4

"""
Then, the smoothnes factor of the segmentation. Good performance is achieved
with values between [0 - 0,5]
"""

beta = 0.1

"""
Then the number of iterations. Most of the time 10-15 iterations are optimal
"""

niter = 20

"""
Now we call an instace of the class TissueClassifierHMRF and its method
called classify with the correspondings inputs.
"""

hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(t1_img, 
                                                              nclass, beta, 
                                                              niter)
                                                              
"""
Now we plot the resulting segmentation and the partial volume 
"""

plt.figure()
plt.imshow(final_segmentation[:, :, 89])
plt.figure()

"""

"""

plt.figure()
plt.imshow(PVE[:, :, 89, 1])
