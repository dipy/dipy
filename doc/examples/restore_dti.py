"""

=====================================================
Using the RESTORE algorithm for robust tensor fitting
=====================================================

The diffusion tensor model takes into account certain kinds of noise (thermal)

"""

import numpy as np

"""
``nibabel`` is for loading imaging datasets
"""

import nibabel as nib

"""
``dipy.reconst.dti`` contains the implementation of WLS tensor fitting as well
as RESTORE. 
"""

import dipy.reconst.dti as dti
reload(dti)

"""
``dipy.data`` is used for small datasets that we use in tests and examples.
"""

import dipy.data as dpd


"""

``dipy.viz.fvtk`` is used for 3D visualization

"""
import dipy.viz.fvtk as fvtk

import matplotlib.pyplot as plt

"""
If needed, the fetch_stanford_hardi function will download the raw dMRI dataset
of a single subject. The size of this dataset is 87 MBytes. You only need to
fetch once. 
"""

dpd.fetch_stanford_hardi()
img, gtab = dpd.read_stanford_hardi()

"""
We initialize a DTI model class instance using the gradient table used in this
measurement. Per defails
"""

dti_wls = dti.TensorModel(gtab)

"""
For the purpose of this example, we will focus on the data from a limited
ROI. We define that ROI as the following indices:
"""

roi_idx = (slice(20,50), slice(55,85), slice(38,39))

"""
And use them to index into the data:
"""

data = img.get_data()[roi_idx]

"""
This data is not very noisy and we will artificially corrupt it to simulate the
effects of 'physiological' noise, such as subject motion. But first, let's
establish a baseline, using the data as it is:   
"""

fit_wls = dti_wls.fit(data)

fa1 = fit_wls.fa
evals1 = fit_wls.evals
evecs1 = fit_wls.evecs
cfa1 = dti.color_fa(fa1, evecs1)
sphere = dpd.get_sphere('symmetric724')

"""
We visualize the ODFs in the ROI using fvtk:
"""

#r = fvtk.ren()
#fvtk.add(r, fvtk.tensor(evals1, evecs1, cfa1, sphere))

#print('Saving illustration as tensor_ellipsoids_wls.png')
#fvtk.record(r, n_frames=1, out_path='tensor_ellipsoids_wls.png',
#            size=(1200, 1200))

"""
.. figure:: tensor_ellipsoids_wls.png
   :align: center

   **Tensor Ellipsoids**.
"""

#fvtk.clear(r)

"""
Next, we corrupt the data with some noise. To simulate a subject that moves
intermittently, we will replace a few of the images with some Rician noise.

"""

noisy_data = np.copy(data)

# We estimate the variance from the b0 volumes:
mean_var = np.mean(np.var(data[..., gtab.b0s_mask], -1))

# We will corrupt the last few volumes:
noisy_idx = slice(-20, None)

noise_real = np.random.randn(*noisy_data[..., noisy_idx].shape) * mean_var
noise_imag = np.random.randn(*noisy_data[..., noisy_idx].shape) * mean_var
rician_noise = np.sqrt(noise_real ** 2 + noise_imag ** 2)
noisy_data[..., noisy_idx] = rician_noise

"""

We use the same model to fit this noisy data

"""

fit_wls_noisy = dti_wls.fit(noisy_data)
fa2 = fit_wls_noisy.fa
evals2 = fit_wls_noisy.evals
evecs2 = fit_wls_noisy.evecs
cfa2 = dti.color_fa(fa2, evecs2)

#r = fvtk.ren()
#fvtk.add(r, fvtk.tensor(evals2, evecs2, cfa2, sphere))

#print('Saving illustration as tensor_ellipsoids_wls_noisy.png')
#fvtk.record(r, n_frames=1, out_path='tensor_ellipsoids_wls_noisy.png',
#            size=(1200, 1200))


"""
In places where the tensor model is particularly sensitive to noise, the
resulting ODF field will be distorted 

.. figure:: tensor_ellipsoids_wls.png
   :align: center

   **Tensor Ellipsoids from noisy data**.

"""

dti_restore = dti.TensorModel(gtab, fit_method='RESTORE',
                              sigma=np.sqrt(mean_var)/1000.)

fit_restore_noisy = dti_restore.fit(noisy_data)
fa3 = fit_restore_noisy.fa
evals3 = fit_restore_noisy.evals
evecs3 = fit_restore_noisy.evecs
cfa3 = dti.color_fa(fa3, evecs3)

#r = fvtk.ren()
#fvtk.add(r, fvtk.tensor(evals3, evecs3, cfa3, sphere))
#print('Saving illustration as tensor_ellipsoids_restore_noisy.png')
#fvtk.record(r, n_frames=1, out_path='tensor_ellipsoids_restore_noisy.png',
#            size=(1200, 1200))




"""

.. include:: ../links_names.inc

"""

fig, ax = plt.subplots(1)
ax.hist(np.ravel(fa1), histtype='step')
ax.hist(np.ravel(fa2), histtype='step')
ax.hist(np.ravel(fa3), histtype='step')
