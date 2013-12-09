"""
=====================================================
Using the RESTORE algorithm for robust tensor fitting
=====================================================

The diffusion tensor model takes into account certain kinds of noise (thermal),
but not other kinds, such as 'physiological' noise. For example, if a subject
moves during the acquisition of one of the diffusion-weighted samples, this
might have a substantial effect on the parameters of the tensor fit calculated
in all voxels in the brain for that subject. One of the pernicious consequences
of this is that it can lead to wrong interepertation of group differences. For
example, some groups of participants (e.g. young children, patient groups,
etc.) are particularly prone to motion and differences in tensor parameters and
derived statistics (such as FA) due to motion would be confounded with actual
differences in the physical properties of the white matter.

One of the strategies to deal with this problem is to apply an automatic method
for detecting outliers in the data, excluding these outliers and refitting the
model without the presence of these outliers. This is often referred to as
"robust model fitting". One of the common algorithms for robust tensor fitting
is called RESTORE, and was first proposed by Chang et al. [1]_.

In the following example, we will demonstrate how to use RESTORE on a simulated
data-set, which we will corrupt by adding intermittent noise.

We start by importing a few of the libraries we will use. ``Numpy`` for numeric
computation: 

"""

import numpy as np

"""
``nibabel`` is for loading imaging datasets
"""

import nibabel as nib

"""
The module ``dipy.reconst.dti`` contains the implementation of tensor fitting,
including an implementation of the RESTORE algorithm.
"""

import dipy.reconst.dti as dti

"""
``dipy.data`` is used for small datasets that we use in tests and examples,
while ``dipy.core.gradients`` is used to represent gradient tables

"""

import dipy.data as dpd
import dipy.core.gradients as grad

"""
We initialize a DTI model class instance using the gradient table used in the
measurement. Per default, dti.Tensor model will use a weighted least-squares
algorithm (described in [2]_) to fit the parameters of the model. We initialize
this model as a baseline for comparison of noise-corrupted models:
"""


b0 = 1000.
bvecs, bval = dpd.read_bvec_file(dpd.get_data('55dir_grad.bvec'))
gtab = grad.gradient_table(bval, bvecs)
B = bval[1]

#Scale the eigenvalues and tensor by the B value so the units match
D = np.array([1., 1., 1., 0., 0., 1., -np.log(b0) * B]) / B
evals = np.array([2., 1., 0.]) / B
md = evals.mean()
tensor = dti.from_lower_triangular(D)

#Design Matrix
X = dti.design_matrix(bvecs, bval)

#Signals
data = np.exp(np.dot(X,D))
data.shape = (-1,) + data.shape

"""
We initialize a DTI model class instance using the gradient table we just
created. Per default, dti.Tensor model will use a weighted least-squares
algorithm (described in [2]_) to fit the parameters of the model. We initialize
this model as a baseline for comparison of noise-corrupted models:
"""

dti_wls = dti.TensorModel(gtab)
fit_wls = dti_wls.fit(data)
fa1 = fit_wls.fa

"""
Next, we corrupt the data by introducing an outlier. To simulate a subject that
moves intermittently, we will replace one of the images with a very low signal
"""

noisy_data = np.copy(data)
noisy_data[..., -1] = 1.0

"""
We use the same model to fit this noisy data
"""

fit_wls_noisy = dti_wls.fit(noisy_data)
fa2 = fit_wls_noisy.fa

"""
To estimate the parameters from the noisy data using RESTORE, we need to
estimate what would be a reasonable amount of noise to expect in the
measurement. There are two common ways of doing that. The first is to look at
the variance in the signal in parts of the volume outside of the brain, or in
the ventricles, where the signal is expected to be identical regardless of the
direction of diffusion weighting. If several non diffusion-weighted volumes
were acquired, another way is to calculate the variance in these volumes.

Here, we estimate that the noise above which the outlier in the simulation is
properly recognized is approximately 67: 
"""

dti_restore = dti.TensorModel(gtab,  fit_method='RESTORE', sigma=67.)
fit_restore_noisy = dti_restore.fit(noisy_data)
fa3 = fit_restore_noisy.fa

print("FA for noiseless data: %s"%fa1)
print("FA for noise-introduced data: %s"%fa2)
print("FA for noise-introduced data, analyzed with RESTORE: %s"%fa3)

"""
This demonstrates that RESTORE can recover the parameters of the outlier-less 
signal. 


References
----------

.. [1] Chang, L-C, Jones, DK and Pierpaoli, C (2005). RESTORE: robust estimation
       of tensors by outlier rejection. MRM, 53: 1088-95. 

.. [2] Chung, SW., Lu, Y., Henry, R.G., 2006. Comparison of bootstrap
       approaches for estimation of uncertainties of DTI parameters.
       NeuroImage 33, 531-541.

.. include:: ../links_names.inc


"""
