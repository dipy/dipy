# -*- coding: utf-8 -*-
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
import nibabel as nib
import matplotlib.pyplot as plt
from dipy.segment.tissue import TissueClassifierHMRF

"""
First we fetch the T1 volume from the Syn dataset and will determine its shape.
"""

img = nib.load('/Users/jvillalo/.dipy/syn_test/t1_brain_denoised.nii')
t1_img = img.get_data()
print('t1_img.shape (%d, %d, %d)' % t1_img.shape)
t1_img = np.interp(t1_img, [0, 400], [0, 1])

"""
We will look at the axial and the coronal slices of the image.
"""

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
img_ax = np.rot90(t1_img[..., 89])
imgplot = plt.imshow(img_ax, cmap="gray")
a.axis('off')
a.set_title('Axial')
a = fig.add_subplot(1, 2, 2)
img_cor = np.rot90(t1_img[:, 128, :])
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('Coronal')
plt.savefig('t1_image.png')

"""
.. figure:: t1_image.png
   :align: center

   **T1-weighted image of healthy adult**.

Now we will define the other three parameters for the segmentation algorithm.
We will segment three classes, namely corticospinal fluid (CSF), white matter (WM) and 
gray matter (GM),
"""

nclass = 3

"""
Then, the smoothnes factor of the segmentation. Good performance is achieved
with values between 0 and 0.5
"""

beta = 0.1

"""
Then we set the number of iterations. Most of the time between 10-20 iterations are enough to get a good segmentation.
"""

niter = 20

"""
Now we call an instace of the class TissueClassifierHMRF and its method
called classify and input the parameters defined above to perform the segmentation task.
"""

hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(t1_img,
                                                              nclass, beta,
                                                              niter)
                                                              
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
plt.savefig('final_seg.png')

"""
Now we plot the resulting segmentation.

.. figure:: final_seg.png
   :align: center

   **This is the resulting segmentation. Each tissue class is color coded separately, red for the WM, yellow for the GM and light blue for the CSF**.

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
plt.savefig('probabilities.png')

"""
And we will also have a look at the probability maps for each tissue class.

.. figure:: probabilities.png
   :align: center

   **These are the probability maps of each of the three tissue classes**.

.. [Zhang2001] Zhang, Y., Brady, M. and Smith, S. Segmentation of Brain MR Images Through a Hidden Markov Random Field Model and the Expectation-Maximization Algorithm IEEE Transactions on Medical Imaging, 20(1): 45-56, 2001
.. [Avants2011] Avants, B. B., Tustison, N. J., Wu, J., Cook, P. A. and Gee, J. C. An open source multivariate framework for n-tissue segmentation with evaluation on public data. Neuroinformatics, 9(4): 381–400, 2011.

"""
