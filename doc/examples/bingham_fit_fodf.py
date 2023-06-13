"""
=======================================
Fitting Bingham distributions to an ODF
=======================================

This example demonstrates how to decompose an ODF into a combination
of Bingham distributions [Riffert2014]_.

First, load useful imports.
"""
from dipy.reconst.bingham import (bingham_fit_sf, bingham_to_sf,
                                  bingham_to_fiber_density,
                                  bingham_to_fiber_spread)
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
                             norm=False)
odf_actor.RotateX(90)

scene = window.Scene()
scene.add(odf_actor)

window.snapshot(scene, 'bingham_in_odf.png')
scene.clear()

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
such as the fiber density (FD) and fiber spread (FS). FD corresponds to the
integral of the Bingham distribution over the sphere and describes the apparent
quantity of fibers passing through an ODF lobe. FS describes the spread of the
ODF lobe. A high FS corresponds to a wide lobe and a low FS, to a sharp one.
"""
fd = bingham_to_fiber_density(fits)
print('Fiber density:', fd)

fs = bingham_to_fiber_spread(fits)
print('Fiber spread:', fs)

"""
Finally, we can validate the quality of our Bingham fit by visualizing the
Bingham distributions overlaid on the input ODF.
"""
actors = []
actors.append(actor.odf_slicer(odf[None, None, None, :], sphere=sphere,
                               opacity=0.5, colormap=(255, 255, 255),
                               norm=False))
for f0, k1, k2, mu1, mu2 in fits:
    sf = bingham_to_sf(f0, k1, k2, mu1, mu2, sphere.vertices)
    actors.append(actor.odf_slicer(sf[None, None, ...], sphere=sphere,
                                   opacity=0.7, norm=False))

# rotate all actors by 90 degrees for better visualization.
for a in actors:
    a.RotateX(90)

scene = window.Scene()
scene.add(*actors)

window.snapshot(scene, 'bingham_fit_overlay.png')
scene.clear()
"""
.. figure:: bingham_fit_overlay.png
   :align: center


References
----------
.. [Riffert2014] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond
                 fractional anisotropy: Extraction of bundle-specific
                 structural metrics from crossing fiber models. NeuroImage.
                 2014 Oct 15;100:176-91.
"""
