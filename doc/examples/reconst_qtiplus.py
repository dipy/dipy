"""
=====================================================================
Applying positivity constraints to Q-space Trajectory Imaging (QTI+)
=====================================================================

Q-space trajectory imaging (QTI) [1]_ with applied positivity constraints
(QTI+) is an estimation framework proposed by Herberthson et al. [2]_ which
enforces necessary constraints during the estimation of the QTI model
parameters.

This tutorial briefly summarizes the theory and provides a comparison between
performing the constrained and unconstrained QTI reconstruction in DIPY.

Theory
======

In QTI, the tissue microstructure is represented by a diffusion tensor
distribution (DTD). Here, DTD is denoted by $\mathbf{D}$ and the voxel-level
diffusion tensor from DTI by $\langle\mathbf{D}\rangle$, where
$\langle \ \rangle$ denotes averaging over the DTD. The covariance of
$\mathbf{D}$ is given by a fourth-order covariance tensor $\mathbb{C}$ defined
as

.. math::

   \mathbb{C} = \langle \mathbf{D} \otimes \mathbf{D} \rangle - \langle
   \mathbf{D} \rangle \otimes \langle \mathbf{D} \rangle ,

where $\otimes$ denotes a tensor outer product. $\mathbb{C}$ has 21 unique
elements and enables the calculation of several microstructural parameters.

Using the cumulant expansion, the diffusion-weighted signal can be approximated
as

.. math::
   S \approx S_0 \exp \left(- \mathbf{b} : \langle \mathbf{D} \rangle +
   \frac{1}{2}(\mathbf{b} \otimes \mathbf{b}) : \mathbb{C} \right) ,

where $S_0$ is the signal without diffusion-weighting, $\mathbf{b}$ is the
b-tensor used in the acquisition, and $:$ denotes a tensor inner product.

The model parameters $S_0$, $\langle \mathbf{D}\rangle$, and $\mathbb{C}$
can be estimated by solving the following weighted problem, where the
heteroskedasticity introduced by the taking the logarithm of the signal is
accounted for:

.. math::

   \\underset{S_0,\langle \mathbf{D} \rangle, \mathbb{C}}{\mathrm{argmin}}
   \sum_{k=1}^n S_k^2 \left| \ln(S_k)-\ln(S_0)+\mathbf{b}^{(k)} \langle
   \mathbf{D} \rangle -\frac{1}{2} (\mathbf{b} \otimes \mathbf{b})^{(k)}
   \mathbb{C} \right|^2 ,

the above can be written as a weighted least squares problem

.. math::

   \mathbf{Ax} = \mathbf{y},

where

.. math::

   y = \begin{pmatrix} \ S_1 \ ln S_1 \\ \vdots \\
   \ S_n \ ln S_n \end{pmatrix} ,

.. math::

   x = \begin{pmatrix} \ln S_0 & \langle \mathbf{D} \rangle & \mathbb{C}
   \end{pmatrix}^\text{T} ,

.. math::

   A =
   \begin{pmatrix}
   S_1 & 0 & \ldots & 0 \\ 0 & \ddots & \ddots & \vdots \\ \vdots & \ddots
   & \ddots & 0 \\ 0 & \ldots & 0 & S_n
   \end{pmatrix}
   \begin{pmatrix}
   1 & -\mathbf{b}_1^\text{T} & \frac{1}{2} (\mathbf{b}_1 \otimes \mathbf{b}_1)
   \text{T} \\
   \vdots & \vdots & \vdots \\
   \vdots & \vdots & \vdots \\
   1 & -\mathbf{b}_n^\text{T} & \frac{1}{2} (\mathbf{b}_n \otimes \mathbf{b}_n)
   ^\text{T}
   \end{pmatrix} ,

where $n$ is the number of acquisitions and $\langle\mathbf{D}\rangle$,
$\mathbb{C}$, $\mathbf{b}_i$, and $(\mathbf{b}_i \otimes \mathbf{b}_i)$ are
represented by column vectors using Voigt notation.

The estimated $\langle\mathbf{D}\rangle$ and $\mathbb{C}$ tensors
should observe mathematical and physical conditions dictated by the model
itself: since $\langle\mathbf{D}\rangle$ represents a diffusivity, it should be
positive semi-definite: $\langle\mathbf{D}\rangle \succeq 0$. Similarly, since
$\mathbf{C}$ represents a covariance, it's $6 \times 6$ representation,
$\mathbf{C}$, should be positive semi-definite:  $\mathbf{C} \succeq 0$

When not imposed, violations of these conditions can occur in presence of noise
and/or in sparsely sampled data. This could results in metrics derived from the
model parameters to be unreliable. Both these conditions can be enfoced while
estimating the QTI model's parameters using semidefinite programming (SDP) as
shown by Herberthson et al. [2]_. This corresponds to solve the problem

.. math::

    \mathbf{Ax} = \mathbf{y} \\

    \text{subject to:} \\

    \langle\mathbf{D}\rangle \succeq 0 , \\
    \mathbf{C} \succeq 0

Installation
=============

The constrained problem stated above can be solved using the cvxpy library.
Instructions on how to install cvxpy
can be found at https://www.cvxpy.org/install/. A free SDP solver
called 'SCS' is installed with cvxpy, and can be readily used for
performing the fit. However, it is recommended to install an
alternative solver, MOSEK, for improved speed and performance.
MOSEK requires a licence which is free for academic use.
Instructions on how to install Mosek and setting up a licence can be found
at https://docs.mosek.com/latest/install/installation.html

Usage example
==============

Here we show how metrics derived from the
QTI model parameters compares when the fit is performed with and without
applying the positivity constraints.

In DIPY, the constrained estimation routine is avaiable as part of the
`dipy.reconst.qti` module.
First we import all the necessary modules to perform the QTI fit:
"""

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
import matplotlib.pyplot as plt
import numpy as np
import dipy.reconst.qti as qti

"""
To showcase why enforcing positivity constraints in QTI is relevant, we use
a human brain dataset comprising 70 volumes acquired with tensor-encoding.
This dataset was obtained by subsampling a richer dataset containing 217
diffusion measurements, which we will use as ground truth when comparing
the parameters estimation with and without applied constraints. This emulates
performing short data acquisition which can correspond to scanning patients
in clinical settings.

The full dataset used in this tutorial was originally published at
https://github.com/filip-szczepankiewicz/Szczepankiewicz_DIB_2019,
and is described in [3]_.

"""

"""
First, let's load the complete dataset and create the gradient table.
We mark these two with the '_217' suffix.
"""
fdata_1, fdata_2, fbvals, fbvecs, fmask = get_fnames('DiB_217_lte_pte_ste')
data_1, _ = load_nifti(fdata_1)
data_2, _ = load_nifti(fdata_2)
data_217 = np.concatenate((data_1, data_2), axis=3)
mask, _ = load_nifti(fmask)
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
btens = np.array(['LTE' for i in range(13)] +
                 ['LTE' for i in range(10)] +
                 ['PTE' for i in range(10)] +
                 ['STE' for i in range(10)] +
                 ['LTE' for i in range(10)] +
                 ['PTE' for i in range(10)] +
                 ['STE' for i in range(10)] +
                 ['LTE' for i in range(16)] +
                 ['PTE' for i in range(16)] +
                 ['STE' for i in range(10)] +
                 ['LTE' for i in range(46)] +
                 ['PTE' for i in range(46)] +
                 ['STE' for i in range(10)])
gtab_217 = gradient_table(bvals, bvecs, btens=btens)

"""
Second, let's load the downsampled dataset and create the gradient table.
We mark these two with the '_70' suffix.
"""

fdata, fbvals, fbvecs, _ = get_fnames('DiB_70_lte_pte_ste')
data, _ = load_nifti(fdata)
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
data_70, _ = load_nifti(fdata)
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
btens = np.array(['LTE' for i in range(1)] +
                 ['LTE' for i in range(4)] +
                 ['PTE' for i in range(3)] +
                 ['STE' for i in range(3)] +
                 ['STE' for i in range(4)] +
                 ['LTE' for i in range(8)] +
                 ['PTE' for i in range(5)] +
                 ['STE' for i in range(5)] +
                 ['LTE' for i in range(21)] +
                 ['PTE' for i in range(10)] +
                 ['STE' for i in range(6)]
                 )
gtab_70 = gradient_table(bvals, bvecs, btens=btens)


"""
Now we can fit the QTI model to the datasets containing 217 measurements, and
use it as reference to compare the constrained and unconstrained fit on the
smaller dataset. For time considerations, we restrict the fit to a
single slice.
"""

mask[:, :, :13] = 0
mask[:, :, 14:] = 0

qtimodel_217 = qti.QtiModel(gtab_217)
qtifit_217 = qtimodel_217.fit(data_217, mask)


"""
Now we can fit the QTI model using the default unconstrained fitting method
to the subsampled dataset:
"""

qtimodel_unconstrained = qti.QtiModel(gtab_70)
qtifit_unconstrained = qtimodel_unconstrained.fit(data_70, mask)

"""
Now we repeat the fit but with the constraints applied.
To perform the constrained fit, we select the 'SDPdc' fit method when creating
the QtiModel object.
Note: this fit method is slower compared to the defaults unconstrained.

If mosek is installed, it can be specified as the solver to be used
as follows:

qtimodel = qti.QtiModel(gtab, fit_method='SDPdc', cvxpy_solver='MOSEK')
qtifit = qtimodel.fit(data, mask)

If Mosek is not installed, the constrained fit can still be performed, and
SCS will be used as solver. SCS is typically much slower than Mosek, but
provides similar results in terms of accuracy. To give an example, the fit
performed in the next line will take approximately 15 minutes when using SCS,
and 2 minute when using Mosek!
"""

qtimodel_constrained = qti.QtiModel(gtab_70, fit_method='SDPdc',
                                    cvxpy_solver='MOSEK')
qtifit_constrained = qtimodel_constrained.fit(data_70, mask)

"""
Now we can visualize the results obtained with the constrained and
unconstrained fit on the small dataset, and compare them with the
"ground truth" provided by fitting the QTI model to the full dataset.
For example, we can look at the FA and $\mu$FA maps, and their value
distribution in White Matter in comparison to the ground truth.
"""

z = 13
wm_mask = qtifit_217.ufa[:, :, z] > 0.6

fig, ax = plt.subplots(2, 4, figsize=(12, 9))

background = np.zeros(data_217.shape[0:2])
for i in range(2):
    for j in range(3):
        ax[i, j].imshow(background, cmap='gray')
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

ax[0, 0].imshow(np.rot90(qtifit_217.fa[:, :, z]), cmap='gray', vmin=0, vmax=1)
ax[0, 0].set_title('GROUND TRUTH')
ax[0, 0].set_ylabel('FA', fontsize=20)
ax[0, 1].imshow(np.rot90(qtifit_unconstrained.fa[:, :, z]),
                cmap='gray', vmin=0, vmax=1)
ax[0, 1].set_title('QTI')
ax[0, 2].imshow(np.rot90(qtifit_constrained.fa[:, :, z]), cmap='gray', vmin=0,
                vmax=1)
ax[0, 2].set_title('QTI+')
ax[0, 3].hist((qtifit_unconstrained.fa[wm_mask, z]).flatten(),
              density=True, bins=40, label='QTI')
ax[0, 3].hist((qtifit_constrained.fa[wm_mask, z]).flatten(),
              density=True, bins=40, label='QTI+', alpha=0.7)
ax[0, 3].hist((qtifit_217.fa[wm_mask, z]).flatten(), histtype='stepfilled',
              density=True, bins=40, label='GT', ec="k", alpha=1,
              linewidth=1.5, fc="None")
ax[0, 3].legend()
ax[0, 3].set_title('VALUE DISTRIBUTION')

ax[1, 0].imshow(np.rot90(qtifit_217.ufa[:, :, z]), cmap='gray', vmin=0, vmax=1)
ax[1, 0].set_title('GROUND TRUTH')
ax[1, 0].set_ylabel('μFA', fontsize=20)
ax[1, 1].imshow(np.rot90(qtifit_unconstrained.ufa[:, :, z]),
                cmap='gray', vmin=0, vmax=1)
ax[1, 1].set_title('QTI')
ax[1, 2].imshow(np.rot90(qtifit_constrained.ufa[:, :, z]), cmap='gray', vmin=0,
                vmax=1)
ax[1, 2].set_title('QTI+')
ax[1, 3].hist((qtifit_unconstrained.ufa[wm_mask, z]).flatten(),
              density=True, bins=40, label='QTI', alpha=1)
ax[1, 3].hist((qtifit_constrained.ufa[wm_mask, z]).flatten(),
              density=True, bins=40, label='QTI+', alpha=0.7)
ax[1, 3].hist((qtifit_217.ufa[wm_mask, z]).flatten(), histtype='stepfilled',
              density=True, bins=40, label='GT', ec="k", alpha=1,
              linewidth=1.5, fc="None")
ax[1, 3].legend()
ax[1, 3].set_title('VALUE DISTRIBUTION')
ax[1, 3].set_xlim([0.4, 1.5])

fig.tight_layout()
plt.show()

"""
The results clearly shows how many of the FA and $\mu$FA values
obtained with the unconstrained fit falls outside the correct
theoretical range [0 1], while the constrained fit provides
more sound results. Note also that even when fitting the rich
dataset, some values of the parameters fall outside the
theoretical range, suggesting that the constrained fit should
be performed even on very densely sampled diffusion data.

For more information about QTI and QTI+, please read the articles by
Westin et al. [1]_ and Herberthson et al. [2]_.


References
----------
.. [1] Westin, Carl-Fredrik, et al. "Q-space trajectory imaging for
   multidimensional diffusion MRI of the human brain." Neuroimage 135
   (2016): 345-362. https://doi.org/10.1016/j.neuroimage.2016.02.039.
.. [2] Herberthson M., Boito D., Dela Haije T., Feragen A., Westin C.-F.,
   Özarslan E., "Q-space trajectory imaging with positivity constraints
   (QTI+)" in Neuroimage, Volume 238, 2021.
.. [3] F Szczepankiewicz, S Hoge, C-F Westin. Linear, planar and spherical
   tensor-valued diffusion MRI data by free waveform encoding in healthy
   brain, water, oil and liquid crystals. Data in Brief (2019),
   DOI: https://doi.org/10.1016/j.dib.2019.104208
.. include:: ../links_names.inc
"""
