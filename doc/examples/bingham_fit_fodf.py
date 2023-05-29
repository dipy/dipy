"""
=======================================
Fitting Bingham distributions to an ODF
=======================================

This example demonstrates how to decompose an ODF into a combination
of Bingham distributions.

First, load useful imports.
"""
from dipy.reconst.bingham import (bingham_fit_sf, bingham_to_sf,
                                  bingham_to_fd, bingham_to_fs,
                                  fd_to_ff)
from dipy.sims.voxel import multi_tensor_odf
from dipy.data import get_sphere
from dipy.viz import window, actor
import numpy as np

"""
Then, we simulate an ODF consisting of 2 fiber populations using
dipy.sims.voxel.multi_tensor_odf. The resulting ODF is displayed below.
"""
sphere = get_sphere('repulsion724').subdivide(2)
mevals = np.array([[0.0015, 0.00050, 0.00010],
                   [0.0015, 0.00015, 0.00015]])
angles = [(0, 0), (60, 0)]
odf = multi_tensor_odf(sphere.vertices, mevals, angles, [50, 50])

# Render the simulated ODF.
odf_actor = actor.odf_slicer(odf[None, None, None, :], sphere=sphere,
                             colormap=(255, 255, 255), norm=False)
scene = window.Scene()
scene.add(odf_actor)

window.snapshot(scene, 'bingham_in_odf.png')

"""
.. figure:: bingham_in_odf.png
   :align: center

We can now use the method `bingham_fit_sf` to fit a Bingham distribution
to each fiber population describing our ODF. The output of the method is a
list of tuples, each of which contains the parameters describing a Bingham
distribution.
"""

# list of coefficients; one per ODF lobe
fits = bingham_fit_sf(odf, sphere, max_search_angle=15,
                      min_sep_angle=15, rel_th=0.1)

# print the parameters of each Bingham distribution
for i, (f0, k1, k2, mu1, mu2) in enumerate(fits):
    print('Fiber population {}:'.format(i))
    print('  Max amplitude (f0):', f0)
    print('  Concentration parameters (k1, k2):', k1, k2)
    print('  Major dispersion axis (mu1):\n', mu1)
    print('  Minor dispersion axis (mu2):\n', mu2)
    print('')

"""
From the fitted Bingham distributions, we can compute fiber specific metrics
such as the fiber density (fd), fiber spread (fs) and fiber fraction (ff).
"""
fd = bingham_to_fd(fits)
print('Fiber density:', fd)

fs = bingham_to_fs(fits)
print('Fiber spread:', fs)

ff = fd_to_ff(fd)
print('Fiber fraction:', ff)

"""
Visualisation
"""
actors = []
odf_actor = actor.odf_slicer(odf[None, None, None, :], sphere=sphere,
                             opacity=0.5, colormap=(255, 255, 255), norm=False)
for f0, k1, k2, mu1, mu2 in fits:
    sf = bingham_to_sf(f0, k1, k2, mu1, mu2, sphere.vertices)
    actors.append(actor.odf_slicer(sf[None, None, ...], sphere=sphere,
                                   opacity=0.9, norm=False))

s = window.Scene()
s.add(odf_actor, *actors)

window.show(s)
