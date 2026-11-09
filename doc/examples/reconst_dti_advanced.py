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
from dipy.core.sphere import Sphere
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst import dti
from dipy.reconst.dti import determinant, deviatoric, isotropic, norm
from dipy.segment.mask import bounding_box, crop, median_otsu
from dipy.viz import actor, window
from dipy.viz.plotting import compare_maps

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")

data, affine = load_nifti(hardi_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

maskdata, mask = median_otsu(
    data, vol_idx=range(10, 50), median_radius=3, numpass=1, dilate=2
)
mins, maxs = bounding_box(mask)
maskdata = crop(maskdata, mins, maxs)
mask = crop(mask, mins, maxs)

tenmodel = dti.TensorModel(gtab, fit_method="WLS")
tenfit = tenmodel.fit(maskdata)

###############################################################################
# Diffusivities from the eigenvalues
# ----------------------------------
#
# Besides FA and MD, the most commonly reported metrics are the axial
# diffusivity (AD) and the radial diffusivity (RD). AD is the diffusivity along
# the principal diffusion direction, the first eigenvalue:
#
# .. math::
#
#     AD = \lambda_1
#
# RD is the diffusivity perpendicular to it, the average of the second and third
# eigenvalues:
#
# .. math::
#
#     RD = \frac{\lambda_2 + \lambda_3}{2}

AD = tenfit.ad
RD = tenfit.rd

###############################################################################
# The trace is the sum of the eigenvalues, which is three times MD. It is
# commonly reconstructed at the scanner as part of clinical protocols for
# assessment and diagnostics:
#
# .. math::
#
#     \mathrm{Tr}(\mathbf{D}) = \lambda_1 + \lambda_2 + \lambda_3

Trace = tenfit.trace

###############################################################################
# Diffusivity along a direction
# -----------------------------
#
# The apparent diffusion coefficient (ADC) is the diffusivity along a given
# direction :math:`\mathbf{g}`:
#
# .. math::
#
#     ADC = \mathbf{g}^T \mathbf{D} \mathbf{g}
#
# ``TensorFit.adc`` evaluates it for every vertex of a sphere, so it returns one
# volume per direction. Here we compute it along the three principal axes.

axes = Sphere(xyz=np.eye(3))
ADC = tenfit.adc(axes)
print("ADC shape (one volume per direction):", ADC.shape)

###############################################################################
# Shape of the tensor
# -------------------
#
# The Westin shape measures :footcite:p:`Westin1997` describe how much the
# tensor resembles a line, a plane or a sphere. The three measures sum to one.
#
# Linearity is high when one eigenvalue dominates, as in a single coherent fiber
# bundle:
#
# .. math::
#
#     c_l = \frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2 + \lambda_3}
#
# Planarity is high when two eigenvalues are large and similar, as where two
# fiber populations cross in a plane:
#
# .. math::
#
#     c_p = \frac{2 (\lambda_2 - \lambda_3)}{\lambda_1 + \lambda_2 + \lambda_3}
#
# Sphericity is high when all three eigenvalues are similar, as in isotropic
# tissue such as gray matter or cerebrospinal fluid:
#
# .. math::
#
#     c_s = \frac{3 \lambda_3}{\lambda_1 + \lambda_2 + \lambda_3}

linearity = tenfit.linearity
planarity = tenfit.planarity
sphericity = tenfit.sphericity

###############################################################################
# To see how these measures relate to the tensor, we draw tensor ellipsoids in the
# same axial slice of the splenium of the corpus callosum as in the
# :ref:`sphx_glr_examples_built_reconstruction_reconst_dti.py` example. Each
# ellipsoid is normalized, so its form shows the shape of the tensor and its color
# shows the value of the metric. Voxels outside the brain mask are not drawn.
#
# The three Westin measures sum to one, so they can be shown together by using
# them as the red, green and blue channels of the color. Red ellipsoids are
# linear, green ellipsoids are planar and blue ellipsoids are spherical. We scale
# each voxel by its largest channel, because sphericity is otherwise dominant.

roi = (slice(13, 43), slice(44, 74), slice(28, 29))
evals = tenfit.evals[roi]
evecs = tenfit.evecs[roi]

westin_colors = np.stack([linearity[roi], planarity[roi], sphericity[roi]], axis=-1)
westin_colors /= westin_colors.max(axis=-1, keepdims=True)
scene = window.Scene()
scene.add(
    actor.tensor_slicer(evals, evecs, mask=mask[roi], colors=westin_colors, scale=0.4)
)
show_manager = window.ShowManager(title="tensor_westin", size=(600, 600), scene=scene)
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Tensor ellipsoids colored by linearity (red), planarity (green) and sphericity
# (blue).
#
#
# Descriptive operations on the tensor
# ------------------------------------
#
# Several basic operations on the tensor matrix are used to build more
# specialized metrics, such as the determinant and the norm. Note that the
# module functions differ in their input: ``determinant``, ``norm``,
# ``isotropic``, ``deviatoric``, ``norm_anisotropy`` and ``mode`` take the
# quadratic form :math:`\mathbf{D}` of the tensor (``tenfit.quadratic_form``), while
# ``trace``, ``fractional_anisotropy``, ``mean_diffusivity`` and the so-called Westin
# measures (see above) take its eigenvalues (``tenfit.evals``). The ``TensorFit`` properties
# pass the right input for you.
#
# The determinant is the product of the eigenvalues. It is proportional to the
# volume of the diffusion ellipsoid:
#
# .. math::
#
#     \det(\mathbf{D}) = \lambda_1 \lambda_2 \lambda_3

q_form = tenfit.quadratic_form
Det = determinant(q_form)

###############################################################################
# The Frobenius norm of the tensor is its overall magnitude:
#
# .. math::
#
#     \lVert \mathbf{D} \rVert = \sqrt{\sum_{i,j} D_{ij}^2}
#     = \sqrt{\lambda_1^2 + \lambda_2^2 + \lambda_3^2}

Norm = norm(q_form)

###############################################################################
# Geodesic anisotropy
# -------------------
#
# Other measures summarize the tensor while accounting for different
# assumptions about the data. For example, the geodesic anisotropy (GA)
# :footcite:p:`Batchelor2005` measures anisotropy with a distance that respects
# the geometry of positive definite matrices. It compares each eigenvalue to
# their geometric mean, the cube root of the determinant:
#
# .. math::
#
#     GA = \sqrt{\sum_{i=1}^3
#          \log^2{\left ( \lambda_i/\langle \mathbf{D} \rangle \right )}},
#          \quad \textrm{where} \quad \langle \mathbf{D} \rangle =
#          (\lambda_1\lambda_2\lambda_3)^{1/3}
#
# Some later papers reproduce this equation incorrectly; DIPY follows the
# original definition (see the ``geodesic_anisotropy`` docstring for details).

GA = tenfit.ga

###############################################################################
# Isotropic and deviatoric parts of the tensor
# --------------------------------------------
#
# The tensor can be split into an isotropic part, which describes the mean
# diffusivity, and a deviatoric part, which describes how the diffusion deviates
# from isotropy :footcite:p:`Ennis2006`:
#
# .. math::
#
#     \bar{\mathbf{D}} = \frac{1}{3} \mathrm{Tr}(\mathbf{D}) \mathbf{I}
#     = MD \, \mathbf{I}, \qquad
#     \widetilde{\mathbf{D}} = \mathbf{D} - \bar{\mathbf{D}}

D_iso = isotropic(q_form)
D_dev = deviatoric(q_form)

###############################################################################
# Orthogonal moments of the diffusion tensor
# ------------------------------------------
#
# :footcite:t:`Ennis2006` showed that the tensor shape can be described by
# three mutually orthogonal invariants. :footcite:t:`Chad2021` interpret them as
# the first three moments of the eigenvalue distribution:
#
# - the mean diffusivity (MD), the first moment, describes the overall magnitude
#   of diffusion;
# - the norm of anisotropy (NA), the second moment, is the norm of the
#   deviatoric tensor, :math:`NA = \lVert \widetilde{\mathbf{D}} \rVert`. It
#   measures the spread of the eigenvalues;
# - the mode of anisotropy (MO), the third moment, describes the shape of the
#   anisotropy. It ranges from -1 (planar) through 0 (orthotropic) to +1
#   (linear).
#
# The three measures are orthogonal in the sense that their gradients with
# respect to the tensor are mutually orthogonal: changing the tensor along the
# gradient of one measure leaves the other two unchanged to first order. For
# example, adding the same
# amount to all three eigenvalues changes MD but leaves NA and MO unchanged.
# This is not true of every change: scaling all eigenvalues by the same factor
# scales both MD and NA, and leaves MO unchanged.
#
# NA is related to FA, which is NA normalized by the norm of the tensor,
# :math:`FA = \sqrt{3/2} \, NA / \lVert \mathbf{D} \rVert`. Because the norm
# of the tensor depends on MD, FA changes with a uniform shift of the
# eigenvalues, while NA does not. Conversely, FA is unchanged by a uniform
# scaling of the eigenvalues, while NA scales with it.
#
# :footcite:t:`Chad2021` suggest that degeneration of regions with a single
# fiber population appears as decreased NA, while selective degeneration of
# secondary crossing fibers appears as increased NA.

MD = tenfit.md
NA = tenfit.na
MO = tenfit.mode

###############################################################################
# Maps of the metrics
# -------------------
#
# Each scalar metric is a volume, so it is usually read as a map of the brain.
# ``compare_maps`` shows one slice of several maps in gray levels. It needs a fit
# of a single slice, so we fit the slice that holds the ellipsoids above, this
# time inside the brain mask. AD, RD and MD share one intensity range, so their
# panels can be compared directly.

slice_fit = tenmodel.fit(maskdata[:, :, 28:29], mask=mask[:, :, 28:29])

diffusivity_range = {"vmin": 0, "vmax": 2.5e-3}
compare_maps(
    [slice_fit],
    ["ad", "rd", "ga"],
    fit_labels=[""],
    map_labels=["AD", "RD", "GA"],
    map_kwargs=[diffusivity_range, diffusivity_range, {"vmin": 0, "vmax": 1.5}],
    filename="tensor_maps_ad_rd_ga.png",
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Maps of AD, RD and GA in one axial slice.
#
#
# The first two orthogonal moments can be shown in the same way.

compare_maps(
    [slice_fit],
    ["md", "na"],
    fit_labels=[""],
    map_labels=["MD", "NA"],
    map_kwargs=[diffusivity_range, {"vmin": 0, "vmax": 1e-3}],
    filename="tensor_maps_md_na.png",
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Maps of MD and NA in one axial slice.
#
#
# The third moment, MO, describes the shape of the tensor, so we draw it on the
# ellipsoids instead of as a map, with planar tensors in blue
# and linear tensors in red. MO is unstable in nearly isotropic tensors, so we only
# draw voxels with an FA above 0.2.

anisotropic = mask[roi] & (tenfit.fa[roi] > 0.2)

mo_bar = actor.scalar_bar(
    clim=(-1, 1),
    cmap="RdBu_r",
    title="MO",
    tick_format=".2g",
    orientation="vertical",
    position=(30, 90),
)
mo_colors = mo_bar.map_values(MO[roi])
scene = window.Scene()
scene.add(
    actor.tensor_slicer(evals, evecs, mask=anisotropic, colors=mo_colors, scale=0.4)
)
scene.ui_scene.add(mo_bar)
show_manager = window.ShowManager(title="tensor_mo", size=(600, 600), scene=scene)
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Tensor ellipsoids with an FA above 0.2, colored by the mode of anisotropy.
#
#
# References
# ----------
#
# .. footbibliography::
#
