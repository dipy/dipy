"""
=====================
Gradients and Spheres
=====================

This example shows how you can create gradient tables and sphere objects using
DIPY_.

Usually, as we saw in
:ref:`sphx_glr_examples_built_quick_start_quick_start.py`,
you load your b-values and b-vectors from disk and then you can create your own
gradient table. But this time let's say that you are an MR physicist and you
want to design a new gradient scheme or you are a scientist who wants to
simulate many different gradient schemes.

Now let's assume that you are interested in creating a multi-shell
acquisition with 2-shells, one at b=1000 $s/mm^2$ and one at b=2500 $s/mm^2$.
For both shells let's say that we want a specific number of gradients (64) and
we want to have the points on the sphere evenly distributed.

This is possible using the :func:`dipy.core.sphere.disperse_charges` which is
an implementation of electrostatic repulsion :footcite:t:`Jones1999` .

Let's start by importing the necessary modules.
"""

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, disperse_charges
from dipy.viz.plotting import plot_gradient_sphere

###############################################################################
# Generating directions
# ---------------------
# Start with 64 random directions on a hemisphere.

rng = np.random.default_rng(42)
n_pts = 64
theta = np.pi * rng.random(n_pts)
phi = 2 * np.pi * rng.random(n_pts)
initial_sphere = HemiSphere(theta=theta, phi=phi)

###############################################################################
# Spreading the directions
# ------------------------
# Minimize electrostatic potential energy over 5000 iterations. Compare the
# initial directions in red with the relaxed directions in green.

relaxed_sphere, potential = disperse_charges(initial_sphere, 5000)
initial_scene = plot_gradient_sphere(initial_sphere, colors=(1, 0, 0))
plot_gradient_sphere(
    relaxed_sphere,
    scene=initial_scene,
    colors=(0, 1, 0),
    filename="initial_vs_updated.png",
    size=(300, 300),
    show=True,
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Illustration of electrostatic repulsion of red points which become green points.
#
# Completing the sphere
# ---------------------
# Mirror the hemisphere to include the antipodal direction of every point.

full_sphere = relaxed_sphere.mirror()
plot_gradient_sphere(
    full_sphere, filename="full_sphere.png", size=(300, 300), show=True
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Full sphere.
#
# Building the acquisition
# ------------------------
# B-vectors give directions; b-values give diffusion weighting. Use the same
# directions for both shells, with one b0 measurement at each end.

directions = relaxed_sphere.vertices
bvecs = np.vstack((np.zeros((1, 3)), directions, directions, np.zeros((1, 3))))
bvals = np.concatenate(([0.0], np.full(n_pts, 1000.0), np.full(n_pts, 2500.0), [0.0]))
gtab = gradient_table(bvals, bvecs=bvecs)
print(f"{len(gtab.bvals)} measurements, {np.count_nonzero(gtab.b0s_mask)} b0s")

###############################################################################
# Plotting gradients
# ------------------
# Plotted radii are proportional to b-values normalized by the largest value.
# The b0 measurements are black, the low shell is blue, and the high shell is
# cyan.
# Use ``size=(300, 300)`` for matching window and saved-image dimensions in pixels.

plot_gradient_sphere(gtab, filename="gradients.png", size=(300, 300), show=True)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Diffusion gradients.
#
# References
# ----------
#
# .. footbibliography::
