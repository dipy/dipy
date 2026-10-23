"""
=================================================
Advanced metrics of the diffusion tensor (DTI)
=================================================

The :ref:`sphx_glr_examples_built_reconstruction_reconst_dti.py` example shows
how to fit the diffusion tensor and compute its two most common summary
measures, fractional anisotropy (FA) and mean diffusivity (MD). Many other
summary measures of the diffusion tensor have been proposed, each combining the
eigenvalues of the tensor in a different way. DIPY implements many of them as
properties of the ``TensorFit`` class and as functions in ``dipy.reconst.dti``.

This example presents these measures, starting from simple diffusivities and
ending with the orthogonal moments of the tensor.

First, we load, mask and fit the same dataset as in the basic DTI example.
"""

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
import dipy.reconst.dti as dti
from dipy.segment.mask import bounding_box, crop, median_otsu

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")

data, affine = load_nifti(hardi_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

maskdata, mask = median_otsu(
    data, vol_idx=range(10, 50), median_radius=3, numpass=1, dilate=2
)
mins, maxs = bounding_box(mask)
maskdata = crop(maskdata, mins, maxs)

tenmodel = dti.TensorModel(gtab, fit_method="WLS")
tenfit = tenmodel.fit(maskdata)

"""
Diffusivities from the eigenvalues
----------------------------------

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
The trace is the sum of the eigenvalues, which is three times MD. It is
commonly reconstructed at the scanner as part of clinical protocols for
assessment and diagnostics:

.. math::

    \\mathrm{Tr}(\\mathbf{D}) = \\lambda_1 + \\lambda_2 + \\lambda_3
"""

Trace = tenfit.trace

"""
Diffusivity along a direction
-----------------------------

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
Shape of the tensor
-------------------

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
Descriptive operations on the tensor
------------------------------------

Several basic operations on the tensor matrix are used to build more
specialized metrics, such as the determinant and the norm. Note that the
module functions differ in their input: ``determinant``, ``norm``,
``isotropic``, ``deviatoric``, ``norm_anisotropy`` and ``mode`` take the
quadratic form $\\mathbf{D}$ of the tensor (``tenfit.quadratic_form``), while
``trace``, ``fractional_anisotropy``, ``mean_diffusivity`` and the so-called Westin
measures (see above) take its eigenvalues (``tenfit.evals``). The ``TensorFit`` properties
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
Geodesic anisotropy
-------------------

Other measures summarize the tensor while accounting for different
assumptions about the data. For example, the geodesic anisotropy (GA)
:footcite:p:`Batchelor2005` measures anisotropy with a distance that respects
the geometry of positive definite matrices. It compares each eigenvalue to
their geometric mean, the cube root of the determinant:

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
References
----------
.. footbibliography::

"""
