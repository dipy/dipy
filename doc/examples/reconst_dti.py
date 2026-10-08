"""
=====================================================================
Reconstruction of the diffusion signal with DTI (single tensor) model
=====================================================================

The diffusion tensor model is a model that describes the diffusion within a
voxel. First proposed by Basser and colleagues :footcite:p:`Basser1994a`, it has
been very influential in demonstrating the utility of diffusion MRI in
characterizing the micro-structure of white matter tissue and of the biophysical
properties of tissue, inferred from local diffusion properties and it is still
very commonly used.

The diffusion tensor models the diffusion signal as:

.. math::

    \\frac{S(\\mathbf{g}, b)}{S_0} = e^{-b\\mathbf{g}^T \\mathbf{D} \\mathbf{g}}

Where $\\mathbf{g}$ is a unit vector in 3 space indicating the direction of
measurement and b are the parameters of measurement, such as the strength and
duration of diffusion-weighting gradient. $S(\\mathbf{g}, b)$ is the
diffusion-weighted signal measured and $S_0$ is the signal conducted in a
measurement with no diffusion weighting. $\\mathbf{D}$ is a positive-definite
quadratic form, which contains six free parameters to be fit. These six
parameters are:

.. math::

    \\mathbf{D} = \\begin{pmatrix} D_{xx} & D_{xy} & D_{xz} \\\\
                       D_{yx} & D_{yy} & D_{yz} \\\\
                       D_{zx} & D_{zy} & D_{zz} \\\\ \\end{pmatrix}

This matrix is a variance/covariance matrix of the diffusivity along the three
spatial dimensions. Note that we can assume that diffusivity has antipodal
symmetry, so elements across the diagonal are equal. For example:
$D_{xy} = D_{yx}$. This is why there are only 6 free parameters to estimate
here.

In the following example we show how to reconstruct your diffusion datasets
using a single tensor model.

First import the necessary modules:

``numpy`` is for numerical computation

"""

import numpy as np

"""
``dipy.io.image`` is for loading / saving imaging datasets
``dipy.io.gradients`` is for loading / saving our bvals and bvecs
"""

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

"""
``dipy.reconst`` is for the reconstruction algorithms which we use to create
voxel models from the raw data.
"""

import dipy.reconst.dti as dti

"""
``dipy.data`` is used for small datasets that we use in tests and examples.
"""

from dipy.data import get_fnames

"""
``get_fnames`` will download the raw dMRI dataset of a single subject.
The size of the dataset is 87 MBytes. You only need to fetch once. It
will return the file names of our data.
"""

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")

"""
Next, we read the saved dataset. gtab contains a ``GradientTable``
object (information about the gradients e.g. b-values and b-vectors).
"""

data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

print(f"data.shape {data.shape}")

"""
data.shape ``(81, 106, 76, 160)``

First of all, we mask and crop the data. This is a quick way to avoid
calculating Tensors on the background of the image. This is done using DIPY_'s
``mask`` module.
"""
from dipy.segment.mask import bounding_box, crop, median_otsu

maskdata, mask = median_otsu(
    data, vol_idx=range(10, 50), median_radius=3, numpass=1, dilate=2
)

mins, maxs = bounding_box(mask)

"""
The ``bounding_box`` function returns the minimum and maximum indices of the
non-zero voxels in every dimension. We use these indices to crop the data
and mask to the smallest possible region that contains all the non-zero voxels.
"""

maskdata = crop(maskdata, mins, maxs)
mask = crop(mask, mins, maxs)

print(f"maskdata.shape {maskdata.shape}")

"""
maskdata.shape ``(72, 87, 59, 160)``

Now that we have prepared the datasets we can go forward with the voxel
reconstruction. First, we instantiate the Tensor model in the following way.
"""

tenmodel = dti.TensorModel(gtab, fit_method="WLS")

"""
The ``fit_method`` argument gives the method that will be used when fitting the
data. Several options are available, such as weighted least squares ``WLS``
(default), non-linear least squares ``NLLS``, as well as robust fitting methods
such as ``RWLS`` and ``RNLLS`` as in :footcite:t:`Coveney2025`.
"""

"""
Fitting the data is very simple. We just need to call the fit method of the
TensorModel in the following way:
"""

tenfit = tenmodel.fit(maskdata)

"""
The fit method creates a ``TensorFit`` object which contains the fitting
parameters and other attributes of the model. You can recover the 6 values
of the triangular matrix representing the tensor D. By default, in DIPY, values
are ordered as (Dxx, Dxy, Dyy, Dxz, Dyz, Dzz). The ``tensor_vals`` variable
defined below is a 4D data with last dimension of size 6.
"""

tensor_vals = dti.lower_triangular(tenfit.quadratic_form)

r"""
You can also recover other metrics from the model. For example we can generate
fractional anisotropy (FA) from the eigen-values of the tensor. FA is used to
characterize the degree to which the distribution of diffusion in a voxel is
directional. That is, whether there is relatively unrestricted diffusion in one
particular direction.

Mathematically, FA is defined as the normalized variance of the eigen-values of
the tensor:

.. math::

        FA = \sqrt{\frac{1}{2} \cdot \frac{(\lambda_1-\lambda_2)^2 +
            (\lambda_1-\lambda_3)^2 + (\lambda_2-\lambda_3)^2}
            {\lambda_1^2 + \lambda_2^2 + \lambda_3^2}}

Where $\lambda_1$, $\lambda_2$ and $\lambda_3$ are the eigen-values of the
tensor.

Note that FA should be interpreted carefully. It may be an indication of
the density of packing of fibers in a voxel, and the amount of myelin wrapping
these axons, but it is not always a measure of "tissue integrity". For example,
FA may decrease in locations in which there is fanning of white matter fibers,
or where more than one population of white matter fibers crosses.
"""

print("Computing anisotropy measures (FA, MD, RGB)")
from dipy.reconst.dti import color_fa, fractional_anisotropy

FA = fractional_anisotropy(tenfit.evals)

"""
In the background of the image the fitting will not be accurate there is no
signal and possibly we will find FA values with nans (not a number). We can
easily remove these in the following way.
"""

FA[np.isnan(FA)] = 0

"""
Saving the FA images is very easy using nibabel_. We need the FA volume and the
affine matrix which transform the image's coordinates to the world coordinates.
Here, we choose to save the FA in ``float32``.
"""

save_nifti("tensor_fa.nii.gz", FA.astype(np.float32), affine)

"""
You can now see the result with any nifti viewer or check it slice by slice
using matplotlib_'s ``imshow``. In the same way you can save the eigen values,
the eigen vectors or any other properties of the tensor.
"""

save_nifti("tensor_evecs.nii.gz", tenfit.evecs.astype(np.float32), affine)

"""
Other tensor statistics can be calculated from the ``tenfit`` object. For
example, a commonly calculated statistic is the mean diffusivity (MD). This is
simply the mean of the  eigenvalues of the tensor. Since FA is a normalized
measure of variance and MD is the mean, they are often used as complimentary
measures. In DIPY, there are two equivalent ways to calculate the mean
diffusivity. One is by calling the ``mean_diffusivity`` module function on the
eigen-values of the ``TensorFit`` class instance:
"""

MD1 = dti.mean_diffusivity(tenfit.evals)
save_nifti("tensors_md.nii.gz", MD1.astype(np.float32), affine)

"""
The other is to call the ``TensorFit`` class method:
"""

MD2 = tenfit.md

"""
Obviously, the quantities are identical.

We can also compute the colored FA or RGB-map :footcite:p:`Pajevic1999`. First,
we make sure that the FA is scaled between 0 and 1, we compute the RGB map and
save it.
"""

FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)
save_nifti("tensor_rgb.nii.gz", np.array(255 * RGB, "uint8"), affine)

"""
Derived parameter maps of the diffusion tensor
----------------------------------------------

Many summary measures of the diffusion tensor have been proposed, each
combining the eigenvalues of the tensor in a different way. DIPY implements
many of them as properties of the ``TensorFit`` class.

Besides FA and MD, the most commonly reported metrics are the axial
diffusivity (AD) and the radial diffusivity (RD). AD is the diffusivity along
the principal diffusion direction, the first eigenvalue:

.. math::

    AD = \\lambda_1

RD is the diffusivity perpendicular to it, the average of the second and third
eigenvalues:

.. math::

    RD = \\frac{\\lambda_2 + \\lambda_3}{2}
"""

AD = tenfit.ad
RD = tenfit.rd

"""
Other measures summarize the tensor while accounting for different
assumptions about the data. For example, the geodesic anisotropy (GA)
:footcite:p:`Batchelor2005` measures anisotropy with a distance that respects
the geometry of positive definite matrices:

.. math::

    GA = \\sqrt{\\sum_{i=1}^3
         \\log^2{\\left ( \\lambda_i/<\\mathbf{D}> \\right )}},
         \\quad \\textrm{where} \\quad <\\mathbf{D}> =
         (\\lambda_1\\lambda_2\\lambda_3)^{1/3}

Some later papers reproduce this equation incorrectly; DIPY follows the
original definition (see the ``geodesic_anisotropy`` docstring for details).
"""

GA = tenfit.ga

"""
The apparent diffusion coefficient (ADC) is the diffusivity along a given
direction $\\mathbf{g}$:

.. math::

    ADC = \\mathbf{g}^T \\mathbf{D} \\mathbf{g}

``TensorFit.adc`` evaluates it for every vertex of a sphere, so it returns one
volume per direction. Here we compute it along the three principal axes.
"""

from dipy.core.sphere import Sphere

axes = Sphere(xyz=np.eye(3))
ADC = tenfit.adc(axes)
print("ADC shape (one volume per direction):", ADC.shape)

"""
Descriptive operations on the tensor
------------------------------------

Several basic operations on the tensor matrix are used to build more
specialized metrics: the determinant, the norm and the trace. Note that the
module functions differ in their input: ``determinant``, ``norm``,
``isotropic``, ``deviatoric``, ``norm_anisotropy`` and ``mode`` take the
quadratic form $\\mathbf{D}$ of the tensor (``tenfit.quadratic_form``), while
``trace``, ``fractional_anisotropy``, ``mean_diffusivity`` and the Westin
measures take its eigenvalues (``tenfit.evals``). The ``TensorFit`` properties
pass the right input for you.

The determinant is the product of the eigenvalues. It is proportional to the
volume of the diffusion ellipsoid:

.. math::

    \\det(\\mathbf{D}) = \\lambda_1 \\lambda_2 \\lambda_3
"""

from dipy.reconst.dti import determinant, norm

q_form = tenfit.quadratic_form
Det = determinant(q_form)

"""
The Frobenius norm of the tensor is its overall magnitude:

.. math::

    \\lVert \\mathbf{D} \\rVert = \\sqrt{\\sum_{i,j} D_{ij}^2}
    = \\sqrt{\\lambda_1^2 + \\lambda_2^2 + \\lambda_3^2}
"""

Norm = norm(q_form)

"""
The trace is the sum of the eigenvalues, which is three times MD. It is
rarely reported on its own, but is useful for deriving other metrics and for
quality assurance:

.. math::

    \\mathrm{Tr}(\\mathbf{D}) = \\lambda_1 + \\lambda_2 + \\lambda_3
"""

Trace = tenfit.trace

"""
The Westin shape measures :footcite:p:`Westin1997` describe how much the
tensor resembles a line, a plane or a sphere. The three measures sum to one.

Linearity is high when one eigenvalue dominates, as in a single coherent fiber
bundle:

.. math::

    c_l = \\frac{\\lambda_1 - \\lambda_2}{\\lambda_1 + \\lambda_2 + \\lambda_3}

Planarity is high when two eigenvalues are large and similar, as where two
fiber populations cross in a plane:

.. math::

    c_p = \\frac{2 (\\lambda_2 - \\lambda_3)}{\\lambda_1 + \\lambda_2 + \\lambda_3}

Sphericity is high when all three eigenvalues are similar, as in isotropic
tissue such as gray matter or cerebrospinal fluid:

.. math::

    c_s = \\frac{3 \\lambda_3}{\\lambda_1 + \\lambda_2 + \\lambda_3}
"""

linearity = tenfit.linearity
planarity = tenfit.planarity
sphericity = tenfit.sphericity

"""
Isotropic and deviatoric parts of the tensor
--------------------------------------------

The tensor can be split into an isotropic part, which describes the mean
diffusivity, and a deviatoric part, which describes how the diffusion deviates
from isotropy :footcite:p:`Ennis2006`:

.. math::

    \\bar{\\mathbf{D}} = \\frac{1}{3} \\mathrm{Tr}(\\mathbf{D}) \\mathbf{I}
    = MD \\, \\mathbf{I}, \\qquad
    \\widetilde{\\mathbf{D}} = \\mathbf{D} - \\bar{\\mathbf{D}}
"""

from dipy.reconst.dti import deviatoric, isotropic

D_iso = isotropic(q_form)
D_dev = deviatoric(q_form)

"""
Orthogonal moments of the diffusion tensor
------------------------------------------

:footcite:t:`Ennis2006` showed that the tensor shape can be described by
three mutually orthogonal invariants. :footcite:t:`Chad2021` interpret them as
the first three moments of the eigenvalue distribution:

- the mean diffusivity (MD), the first moment, describes the overall magnitude
  of diffusion;
- the norm of anisotropy (NA), the second moment, is the norm of the
  deviatoric tensor, $NA = \\lVert \\widetilde{\\mathbf{D}} \\rVert$. It
  measures the spread of the eigenvalues;
- the mode of anisotropy (MO), the third moment, describes the shape of the
  anisotropy. It ranges from -1 (planar) through 0 (orthotropic) to +1
  (linear).

The three measures are orthogonal in the sense that their gradients with
respect to the tensor are mutually orthogonal: changing the tensor along the
gradient of one measure leaves the other two unchanged to first order. For
example, adding the same
amount to all three eigenvalues changes MD but leaves NA and MO unchanged.
This is not true of every change: scaling all eigenvalues by the same factor
scales both MD and NA, and leaves MO unchanged.

NA is related to FA, which is NA normalized by the norm of the tensor,
$FA = \\sqrt{3/2} \\, NA / \\lVert \\mathbf{D} \\rVert$. Because the norm
of the tensor depends on MD, FA changes with a uniform shift of the
eigenvalues, while NA does not. Conversely, FA is unchanged by a uniform
scaling of the eigenvalues, while NA scales with it.

:footcite:t:`Chad2021` suggest that degeneration of regions with a single
fiber population appears as decreased NA, while selective degeneration of
secondary crossing fibers appears as increased NA.
"""

MD = tenfit.md
NA = tenfit.na
MO = tenfit.mode

"""
Let's try to visualize the tensor ellipsoids of a small rectangular
area in an axial slice of the splenium of the corpus callosum (CC).
"""

print("Computing tensor ellipsoids in a part of the splenium of the CC")

from dipy.data import get_sphere

sphere = get_sphere(name="repulsion724")

from dipy.viz import actor, window

# Enables/disables interactive visualization
interactive = False

scene = window.Scene()

evals = tenfit.evals[13:43, 44:74, 28:29]
evecs = tenfit.evecs[13:43, 44:74, 28:29]

"""
We can color the ellipsoids using the ``color_fa`` values that we calculated
above. In this example we additionally normalize the values to increase the
contrast.
"""

cfa = RGB[13:43, 44:74, 28:29]
cfa /= cfa.max()

scene.add(
    actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=sphere, scale=0.3)
)

print("Saving illustration as tensor_ellipsoids.png")
window.record(
    scene=scene, n_frames=1, out_path="tensor_ellipsoids.png", size=(600, 600)
)
if interactive:
    window.show(scene)

"""
.. rst-class:: centered small fst-italic fw-semibold

Tensor Ellipsoids.
"""

scene.clear()

"""
Finally, we can visualize the tensor Orientation Distribution Functions
for the same area as we did with the ellipsoids.
"""

tensor_odfs = tenmodel.fit(data[20:50, 55:85, 38:39]).odf(sphere)

odf_actor = actor.odf_slicer(tensor_odfs, sphere=sphere, scale=0.5, colormap=None)
scene.add(odf_actor)
print("Saving illustration as tensor_odfs.png")
window.record(scene=scene, n_frames=1, out_path="tensor_odfs.png", size=(600, 600))
if interactive:
    window.show(scene)

"""
.. rst-class:: centered small fst-italic fw-semibold

Tensor ODFs.


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

References
----------
.. footbibliography::

"""
