# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:57:42 2016

@author: jvillalo
"""
"""
=======================================================
Tissue Classification of a T1-weighted Strutural Image
=======================================================
This example explains how to segment a T1-weighted structural image using a MRF
approach. Similar algorithms have been proposed by Zhang et al. [Zhang2001]_,
and Avants et al. [Avants2011]_ available in FAST-FSL and ANTS-Artropos,
respectively.

Here we will use a T1-weighted image, that has been previously skull-stripped
and bias field corrected.
"""

import numpy as np
#import nibabel as nib
import matplotlib.pyplot as plt
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.data import fetch_tissue_data, read_tissue_data
from dipy.segment.tissue import TissueClassifierHMRF

"""
First we fetch the T1 volume from the Syn dataset and will determine its shape.
"""

fetch_tissue_data()
t1_img = read_tissue_data()
t1 = t1_img.get_data()
print('t1.shape (%d, %d, %d)' % t1.shape)

"""
Once we have fetched the T1 volume, we proceed to denoise it with the NLM algorithm.
"""

t1 = np.interp(t1, [0, 400], [0, 1])
sigma = estimate_sigma(t1)
print(sigma)
t1 = nlmeans(t1, sigma=sigma)

"""
We will look at the axial and the coronal slices of the image.
"""

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
img_ax = np.rot90(t1[..., 89])
imgplot = plt.imshow(img_ax, cmap="gray")
a.axis('off')
a.set_title('Axial')
a = fig.add_subplot(1, 2, 2)
img_cor = np.rot90(t1[:, 128, :])
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('Coronal')
plt.savefig('t1_image.png', bbox_inches='tight', pad_inches=0)

"""
.. figure:: t1_image.png
   :align: center

   **T1-weighted image of healthy adult**.

Now we will define the other two parameters for the segmentation algorithm.
We will segment three classes, namely corticospinal fluid (CSF), white matter (WM) and 
gray matter (GM),
"""

nclass = 3

"""
Then, the smoothnes factor of the segmentation. Good performance is achieved
with values between 0 and 0.5.
"""

beta = 0.1

"""
Now we set the convergence criterion.
"""

tolerance = 0.01

"""
Now we call an instace of the class TissueClassifierHMRF and its method
called classify and input the parameters defined above to perform the segmentation task.
"""

hmrf = TissueClassifierHMRF(save_history=True)
initial_segmentation, final_segmentation, PVE, EN = hmrf.classify(t1,
                                                                  nclass, beta,
                                                                  tolerance)

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
img_ax = np.rot90(final_segmentation[..., 89])
imgplot = plt.imshow(img_ax)
a.axis('off')
a.set_title('Axial')
a = fig.add_subplot(1, 2, 2)
img_cor = np.rot90(final_segmentation[:, 128, :])
imgplot = plt.imshow(img_cor)
a.axis('off')
a.set_title('Coronal')
plt.savefig('final_seg.png', bbox_inches='tight', pad_inches=0)

"""
Now we plot the resulting segmentation.

.. figure:: final_seg.png
   :align: center

   **Each tissue class is color coded separately, red for the WM, yellow for the GM and light blue for the CSF**.

And we will also have a look at the probability maps for each tissue class.
"""

fig = plt.figure()
a = fig.add_subplot(1, 3, 1)
img_ax = np.rot90(PVE[..., 89, 0])
imgplot = plt.imshow(img_ax, cmap="gray")
a.axis('off')
a.set_title('CSF')
a = fig.add_subplot(1, 3, 2)
img_cor = np.rot90(PVE[:, :, 89, 1])
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('Gray Matter')
a = fig.add_subplot(1, 3, 3)
img_cor = np.rot90(PVE[:, :, 89, 2])
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('White Matter')
plt.savefig('probabilities.png', bbox_inches='tight', pad_inches=0)

"""
.. figure:: probabilities.png
   :align: center
   :scale: 120

   **These are the probability maps of each of the three tissue classes**.

.. [Zhang2001] Zhang, Y., Brady, M. and Smith, S. Segmentation of Brain MR Images Through a Hidden Markov Random Field Model and the Expectation-Maximization Algorithm IEEE Transactions on Medical Imaging, 20(1): 45-56, 2001
.. [Avants2011] Avants, B. B., Tustison, N. J., Wu, J., Cook, P. A. and Gee, J. C. An open source multivariate framework for n-tissue segmentation with evaluation on public data. Neuroinformatics, 9(4): 381–400, 2011.

"""
