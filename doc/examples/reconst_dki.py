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
along the spatial direction $\mathbf{n}$ [NetoHe2015]: 

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

import dipy.reconst.dti as dti

"""
``dipy.data`` is used for small datasets that we use in tests and examples.
"""

from dipy.data import fetch_stanford_hardi

"""
Fetch will download the raw dMRI dataset of a single subject. The size of the
dataset is 87 MBytes.  You only need to fetch once.
"""

fetch_stanford_hardi()

"""
Next, we read the saved dataset
"""

from dipy.data import read_stanford_hardi

img, gtab = read_stanford_hardi()

"""
img contains a nibabel Nifti1Image object (with the data) and gtab contains a
GradientTable object (information about the gradients e.g. b-values and
b-vectors).
"""

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
data.shape ``(81, 106, 76, 160)``

First of all, we mask and crop the data. This is a quick way to avoid
calculating Tensors on the background of the image. This is done using dipy's
mask module.
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
The fit method creates a TensorFit object which contains the fitting parameters
and other attributes of the model. For example we can generate fractional
anisotropy (FA) from the eigen-values of the tensor. FA is used to characterize
the degree to which the distribution of diffusion in a voxel is
directional. That is, whether there is relatively unrestricted diffusion in one
particular direction.

Mathematically, FA is defined as the normalized variance of the eigen-values of
the tensor:

.. math::

        FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
                    \lambda_3)^2+(\lambda_2-\lambda_3)^2}{\lambda_1^2+
                    \lambda_2^2+\lambda_3^2}}

Note that FA should be interperted carefully. It may be an indication of the
density of packing of fibers in a voxel, and the amount of myelin wrapping these
axons, but it is not always a measure of "tissue integrity". For example, FA
may decrease in locations in which there is fanning of white matter fibers, or
where more than one population of white matter fibers crosses.
"""

print('Computing anisotropy measures (FA, MD, RGB)')
from dipy.reconst.dti import fractional_anisotropy, color_fa, lower_triangular

FA = fractional_anisotropy(tenfit.evals)

"""
In the background of the image the fitting will not be accurate there is no
signal and possibly we will find FA values with nans (not a number). We can
easily remove these in the following way.
"""

FA[np.isnan(FA)] = 0

"""
Saving the FA images is very easy using nibabel. We need the FA volume and the
affine matrix which transform the image's coordinates to the world coordinates.
Here, we choose to save the FA in float32.
"""

fa_img = nib.Nifti1Image(FA.astype(np.float32), img.get_affine())
nib.save(fa_img, 'tensor_fa.nii.gz')

"""
You can now see the result with any nifti viewer or check it slice by slice
using matplotlib_'s imshow. In the same way you can save the eigen values, the
eigen vectors or any other properties of the Tensor.
"""

evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), img.get_affine())
nib.save(evecs_img, 'tensor_evecs.nii.gz')

"""
Other tensor statistics can be calculated from the `tenfit` object. For example,
a commonly calculated statistic is the mean diffusivity (MD). This is simply the
mean of the  eigenvalues of the tensor. Since FA is a normalized
measure of variance and MD is the mean, they are often used as complimentary
measures. In `dipy`, there are two equivalent ways to calculate the mean
diffusivity. One is by calling the `mean_diffusivity` module function on the
eigen-values of the TensorFit class instance:
"""

MD1 = dti.mean_diffusivity(tenfit.evals)
nib.save(nib.Nifti1Image(MD1.astype(np.float32), img.get_affine()), 'tensors_md.nii.gz')

"""
The other is to call the TensorFit class method:
"""

MD2 = tenfit.md

"""
Obviously, the quantities are identical.

We can also compute the colored FA or RGB-map [Pajevic1999]_. First, we make sure
that the FA is scaled between 0 and 1, we compute the RGB map and save it.
"""

FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)
nib.save(nib.Nifti1Image(np.array(255 * RGB, 'uint8'), img.get_affine()), 'tensor_rgb.nii.gz')

"""
Let's try to visualize the tensor ellipsoids of a small rectangular
area in an axial slice of the splenium of the corpus callosum (CC).
"""

print('Computing tensor ellipsoids in a part of the splenium of the CC')

from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

from dipy.viz import fvtk
ren = fvtk.ren()

evals = tenfit.evals[13:43, 44:74, 28:29]
evecs = tenfit.evecs[13:43, 44:74, 28:29]

"""
We can color the ellipsoids using the ``color_fa`` values that we calculated
above. In this example we additionally normalize the values to increase the contrast.
"""

cfa = RGB[13:43, 44:74, 28:29]
cfa /= cfa.max()

fvtk.add(ren, fvtk.tensor(evals, evecs, cfa, sphere))

print('Saving illustration as tensor_ellipsoids.png')
fvtk.record(ren, n_frames=1, out_path='tensor_ellipsoids.png', size=(600, 600))

"""
.. figure:: tensor_ellipsoids.png
   :align: center

   **Tensor Ellipsoids**.
"""

fvtk.clear(ren)

"""
Finally, we can visualize the tensor orientation distribution functions
for the same area as we did with the ellipsoids.
"""

tensor_odfs = tenmodel.fit(data[20:50, 55:85, 38:39]).odf(sphere)

fvtk.add(ren, fvtk.sphere_funcs(tensor_odfs, sphere, colormap=None))
#fvtk.show(r)
print('Saving illustration as tensor_odfs.png')
fvtk.record(ren, n_frames=1, out_path='tensor_odfs.png', size=(600, 600))

"""
.. figure:: tensor_odfs.png
   :align: center

   **Tensor ODFs**.

Note that while the tensor model is an accurate and reliable model of the
diffusion signal in the white matter, it has the drawback that it only has one
principal diffusion direction. Therefore, in locations in the brain that
contain multiple fiber populations crossing each other, the tensor model may
indicate that the principal diffusion direction is intermediate to these
directions. Therefore, using the principal diffusion direction for tracking in
these locations may be misleading and may lead to errors in defining the
tracks. Fortunately, other reconstruction methods can be used to represent the
diffusion and fiber orientations in those locations. These are presented in
other examples.


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
