"""
==================================================================
Continuous and analytical diffusion signal modelling with 3D-SHORE
==================================================================

We show how to model the diffusion signal as a linear combination
of continuous functions from the SHORE basis :footcite:p:`Merlet2013`,
:footcite:p:`Ozarslan2008`, :footcite:p:`Ozarslan2009`. We also compute the
analytical Orientation Distribution Function (ODF).

First import the necessary modules:
"""

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.shore import ShoreModel
from dipy.viz.skyline import skyline

###############################################################################
# Download and read the data for this tutorial.

# ``fetch_isbi2013_2shell()`` provides data from the `ISBI HARDI contest 2013
# <https://hardi.epfl.ch/static/events/2013_ISBI/>`_ acquired for two shells at
# b-values 1500 $s/mm^2$ and 2500 $s/mm^2$.

# The six parameters of these two functions define the ROI where to reconstruct
# the data. They respectively correspond to ``(xmin,xmax,ymin,ymax,zmin,zmax)``
# with x, y, z and the three axis defining the spatial positions of the voxels.

fraw, fbval, fbvec = get_fnames(name="isbi2013_2shell")

data, affine = load_nifti(fraw)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs=bvecs)
data_small = data[10:40, 22, 10:40]

print(f"data.shape {data.shape}")

###############################################################################
# ``data`` contains the voxel data and ``gtab`` contains a ``GradientTable``
# object (gradient information e.g. b-values). For example, to show the
# b-values it is possible to write::
#
#     print(gtab.bvals)
#
# Instantiate the SHORE Model.
#
# ``radial_order`` is the radial order of the SHORE basis.
#
# ``zeta`` is the scale factor of the SHORE basis.
#
# ``lambdaN`` and ``lambdaL`` are the radial and angular regularization
# constants, respectively.
#
# For details regarding these four parameters see :footcite:p:`Cheng2011` and
# :footcite:p:`Merlet2013`.

radial_order = 6
zeta = 700
lambdaN = 1e-8
lambdaL = 1e-8
asm = ShoreModel(
    gtab, radial_order=radial_order, zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL
)

###############################################################################
# Fit the SHORE model to the data

asmfit = asm.fit(data_small)

###############################################################################
# Load an odf reconstruction sphere

sphere = get_sphere(name="repulsion724")

###############################################################################
# Compute the ODFs

odf = asmfit.odf(sphere)
print(f"odf.shape {odf.shape}")

###############################################################################
# Display the analytical ODF harmonics, not an approximate refit of the SHORE
# signal. Normalize only the display coefficients to keep glyphs visible.

sh_coeffs = asmfit.odf_sh()
odf_max = np.max(np.abs(odf), axis=-1, keepdims=True)
odf_scale = np.divide(0.5, odf_max, out=np.zeros_like(odf_max), where=odf_max > 0)
sh_coeffs *= odf_scale
sh_coeffs = sh_coeffs[:, None, :, :]

###############################################################################
# This voxel-space display rotation turns the coronal slab face-on; it is not
# the acquisition affine.

display_affine = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=float,
)
_skyline_obj = skyline(
    sh_coeffs=[(sh_coeffs, display_affine, "SHORE ODFs", "descoteaux07_legacy")],
    out_stealth_png="odfs.png",
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Analytical orientation distribution functions, colored by orientation:
# red for left-right, green for anterior-posterior, and blue for superior-inferior.
#
#
# References
# ----------
#
# .. footbibliography::
#
