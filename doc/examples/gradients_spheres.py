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
from dipy.core.sphere import HemiSphere, Sphere, disperse_charges
from dipy.viz.plotting import plot_gradient_sphere

###############################################################################
# We can first create some random points on a ``HemiSphere`` using spherical
# polar coordinates.

rng = np.random.default_rng()
n_pts = 64
theta = np.pi * rng.random(n_pts)
phi = 2 * np.pi * rng.random(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)

###############################################################################
# Next, we call :func:`dipy.core.sphere.disperse_charges` which will
# iteratively move the points so that the electrostatic potential energy is
# minimized.

hsph_updated, potential = disperse_charges(hsph_initial, 5000)

###############################################################################
# In ``hsph_updated`` we have the updated ``HemiSphere`` with the points nicely
# distributed on the hemisphere. Let's visualize them.


initial_scene = plot_gradient_sphere(hsph_initial, colors=(1, 0, 0))
plot_gradient_sphere(
    hsph_updated,
    scene=initial_scene,
    colors=(0, 1, 0),
    filename="initial_vs_updated.png",
    show=True,
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Illustration of electrostatic repulsion of red points which become
# green points.
#
#
# We can also create a sphere from the hemisphere and show it in the
# following way.

sph = Sphere(xyz=np.vstack((hsph_updated.vertices, -hsph_updated.vertices)))

plot_gradient_sphere(sph, colors=(0, 1, 0), filename="full_sphere.png", show=True)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Full sphere.
#
#
# It is time to create the Gradients. For this purpose we will use the
# function ``gradient_table`` and fill it with the ``hsph_updated`` vectors
# that we created above.

vertices = hsph_updated.vertices
values = np.ones(vertices.shape[0])

###############################################################################
# We need two stacks of ``vertices``, one for every shell, and we need two sets
# of b-values, one at 1000 $s/mm^2$, and one at 2500 $s/mm^2$, as we discussed
# previously.

bvecs = np.vstack((vertices, vertices))
bvals = np.hstack((1000 * values, 2500 * values))

###############################################################################
# We can also add some b0s. Let's add one at the beginning and one at the end.

bvecs = np.insert(bvecs, (0, bvecs.shape[0]), np.array([0, 0, 0]), axis=0)
bvals = np.insert(bvals, (0, bvals.shape[0]), 0)

print(bvals)
print(bvecs)

###############################################################################
# Both b-values and b-vectors look correct. Let's now create the
# ``GradientTable``.

gtab = gradient_table(bvals, bvecs=bvecs)


###############################################################################
# We can also visualize the gradients. The b0 points are black, the lowest
# shell is blue, and higher shells interpolate from blue to cyan.

plot_gradient_sphere(gtab, filename="gradients.png", show=True)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Diffusion gradients.
#
#
# References
# ----------
#
# .. footbibliography::
