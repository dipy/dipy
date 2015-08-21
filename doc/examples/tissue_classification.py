# -*- coding: utf-8 -*-
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

t1_img = np.interp(t1_img, [0, 400], [0, 1])

#from dipy.data import fetch_mni_template, read_mni_template
#fetch_mni_template()
#t1_img = read_mni_template(contrast="T1")

#from dipy.data.fetcher import fetch_syn_data, read_syn_data
#fetch_syn_data()
#nib_syn_t1, nib_syn_b0 = read_syn_data()
#syn_b0 = np.array(nib_syn_b0.get_data())

"""
We can have a look at a axial slice of the image
"""

plt.figure()
plt.imshow(t1_img[..., 89], cmap="gray")
plt.savefig('t1_image.png')


"""
.. figure:: t1_image.png
   :align: center

   **This is the probability map of the white matter**.

"""
"""
Now we will define the other three parameters for the segmentation algorith
We will segment CSF, WM and GM (three classes) and we must add the background
First, the number of tissue classes
"""

nclass = 3

"""
Then, the smoothnes factor of the segmentation. Good performance is achieved
with values between 0 and 0,5
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
                                                          
plt.figure()
plt.imshow(final_segmentation[:, :, 89])
plt.savefig('final_seg.png')

"""
Now we plot the resulting segmentation.

.. figure:: final_seg.png
   :align: center

   **This is the resulting segmentation into 3 tissue types**.

"""

plt.figure()
plt.imshow(PVE[:, :, 89, 2])
plt.savefig('pve_whitematter.png')

"""
And we also have a look at the white matter probability map at the same slice.

.. figure:: pve_whitematter.png
   :align: center

   **This is the probability map of the white matter**.
"""

"""
.. [Zhang2001] Zhang, Y., Brady, M. and Smith, S. Segmentation of Brain MR Images Through a Hidden Markov Random Field Model and the Expectation-Maximization Algorithm IEEE Transactions on Medical Imaging, 20(1): 45-56, 2001
.. [Avants2011] Avants, B. B., Tustison, N. J., Wu, J., Cook, P. A. and Gee, J. C. An open source multivariate framework for n-tissue segmentation with evaluation on public data. Neuroinformatics, 9(4): 381–400, 2011.

.. include:: ../links_names.inc

"""
