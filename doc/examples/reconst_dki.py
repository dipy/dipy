"""

=====================================================================
Reconstruction of the diffusion signal with the kurtosis tensor model
=====================================================================

The diffusion kurtois model is an expansion of the diffusion tensor model. In
addition to the diffusio tensor (DT), the diffusion kurtosis model quantifies
the diffusion kurtosis tensor (KT) which measures the degree to which water
diffusion in biologic tissues is non-Gaussian [Jensen2005]_. Measurements of
non-Gaussian diffusion are of interest since they can be used to charaterize
tissue microstructural heterogeneity [Jensen2010]_ and to derive concrete
biophysical parameters as the density of axonal fibres and diffusion tortuosity
[Fierem2011]_. Moreover, DKI can be used to resolve crossing fibers on
tractography and to obtain invariant rotational measures not limited to well
aligned fiber populations [NetoHe2015]_.

The diffusion kurtosis model relates the diffusion-weighted signal,
$S(\mathbf{n}, b)$, to the applied diffusion weighting, $\mathbf{b}$, the
signal in the absence of diffusion gradient sensitisation, $S_0$, and the
values of diffusion, $\mathbf{D(n)}$, and diffusion kurtosis, $\mathbf{K(n)}$,
along the spatial direction $\mathbf{n}$ [NetoHe2015]_: 

.. math::
    S(n,b)=S_{0}e^{-bD(n)+\frac{1}{6}b^{2}D(n)^{2}K(n)}
    
$\mathbf{D(n)}$ and $\mathbf{K(n)}$ can be computed from the KT and DT using
the following equations:

.. math::
     D(n)=\sum_{i=1}^{3}\sum_{j=1}^{3}n_{i}n_{j}D_{ij}
     
and 

.. math::
     K(n)=\frac{MD^{2}}{D(n)^{2}}\sum_{i=1}^{3}\sum_{j=1}^{3}\sum_{k=1}^{3}
     \sum_{l=1}^{3}n_{i}n_{j}n_{k}n_{l}W_{ijkl}

where $D_{ij}$ and $W_{ijkl}$ are the elements of the second-order DT and the
fourth-order KT tensors, respectively, and $MD$ is the mean diffusivity.
As the DT, KT has antipodal symmetry and thus only 15 Wijkl elemments are 
needed to fully characterize the KT:  

.. math::
   \begin{matrix} ( & W_{xxxx} & W_{yyyy} & W_{zzzz} & W_{xxxy} & W_{xxxz}
                    & ... \\ 
                    & W_{xyyy} & W_{yyyz} & W_{xzzz} & W_{yzzz} & W_{xxyy}
                    & ... \\
                    & W_{xxzz} & W_{yyzz} & W_{xxyz} & W_{xyyz} & W_{xyzz}
                    & & )\end{matrix}

In the following example we show how to reconstruct your diffusion datasets
using the kurtosis tensor model.

First import the necessary modules:

``numpy`` is for numerical computation

"""

import numpy as np

"""
``nibabel`` is for loading imaging datasets
"""

import nibabel as nib

"""
``dipy.reconst`` is for the reconstruction algorithms which we use to create
voxel models from the raw data.
"""

import dipy.reconst.dki as dki

"""
``dipy.data`` is used for small datasets that we use in tests and examples. DKI
required multi shell data, i.e. data acquired from more than one non-zero
b-value.
"""

from dipy.data import fetch_sherbrooke_3shell

"""
Fetch will download the raw HARDI dMRI dataset of a single subject. The size of
the dataset is 188 MBytes, however you only need to fetch it once.
"""

fetch_sherbrooke_3shell()

"""
Next, we read the saved dataset
"""

from dipy.data import read_sherbrooke_3shell

img, gtab = read_sherbrooke_3shell()

"""
img contains a nibabel Nifti1Image object (with the data) and gtab contains a
GradientTable object with information about the b-values and b-vectors. The
b-values used on the loaded dataset are visualized above
"""

import matplotlib.pyplot as plt

plt.plot(gtab.bvals, label='b-values')
plt.legend()
plt.show()
plt.savefig('HARDI193_bvalues.png')

"""
.. figure:: HARDI193_bvalues.png
   :align: center
   **b-values of the loaded dataset**.
   
From the figure above we can check that the loaded dataset containing three
non-zero b-values as required for DKI. However the highest b-value of 3500
$s.mm^{-2}$ is higher than normally used on DKI. Since DKI negletes diffusion
signal components higher than the 4th order KT, a upper bound of b-value < 3000
$s.mm^{-2}$ is normally implied to insure the fitting viability[Jensen2010]_. 
Following this, we discard the b-value shell of 3500 $s.mm^{-2}$ before DKI
fitting
"""

select_ind = gtab.bvals < 3000

selected_bvals = gtab.bvals[select_ind]
selected_bvecs = gtab.bvecs[select_ind, :]

data = img.get_data()
selected_data = data[:, :, :, select_ind]

"""
Before fitting the data some data pre-processing is done. First, we mask and
crop the data to avoid calculating Tensors on the background of the image.
"""

from dipy.segment.mask import median_otsu

maskdata, mask = median_otsu(data, 3, 1, True,
                             vol_idx=range(10, 50), dilate=2)
print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)

"""
maskdata.shape ``(72, 87, 59, 160)``

Now that we have prepared the datasets we can go forward with the voxel
reconstruction. First, we instantiate the Tensor model in the following way.
"""

tenmodel = dti.TensorModel(gtab)

"""
Fitting the data is very simple. We just need to call the fit method of the
TensorModel in the following way:
"""

tenfit = tenmodel.fit(maskdata)

"""
References:

.. [Jensen2005] Jensen JH, Helpern JA, Ramani A, Lu H, Kaczynski K (2005).
                Diffusional Kurtosis Imaging: The Quantification of
                Non_Gaussian Water Diffusion by Means of Magnetic Resonance
                Imaging. Magnetic Resonance in Medicine 53: 1432-1440.
.. [Jensen2010] Jensen JH, Helpern JA (2010). MRI quantification of
                non-Gaussian water diffusion by kurtosis analysis. NMR in 
                Biomedicine 23(7): 698–710.
.. [Fierem2011] Fieremans E, Jensen JH, Helpern JA (2011). White matter
                characterization with diffusion kurtosis imaging. NeuroImage 
                58: 177–188.
.. [NetoHe2015] Neto Henriques R, Correia MM, Nunes RG, Ferreira HA (2015).
                Exploring the 3D geometry of the diffusion kurtosis tensor -
                Impact on the development of robust tractography procedures and
                novel biomarkers, NeuroImage 111: 85-99

.. include:: ../links_names.inc

"""
