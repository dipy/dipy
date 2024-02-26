"""
=======================================
Fitting Bingham distributions to an ODF
=======================================

This example demonstrates how to decompose an ODF into a combination
of Bingham distributions [Riffert2014]_.

First, load useful imports.
"""

import numpy as np
from dipy.direction.bingham import (bingham_fit_odf, bingham_odf,
                                    bingham_fiber_density,
                                    bingham_fiber_spread,
                                    bingham_orientation_dispersion,
                                    bingham_from_sh)

from dipy.sims.voxel import multi_tensor_odf
from dipy.data import get_sphere
from dipy.viz import window, actor

sh = '/Users/julio/Documents/IGC/CAMCAN/sub-CC121795_RumbaCSD_iso_SH6.nii.gz'
mask = '/Users/julio/Documents/IGC/CAMCAN/sub-CC121795_desc-eddy_mask_final.nii.gz'
sh_order = 6
npeaks = 3
sphere = 'symmetric642'

bingham_from_sh(sh, mask, sh_order, npeaks, sphere)

# """
# Then, we simulate an ODF consisting of two fiber populations using
# dipy.sims.voxel.multi_tensor_odf. The resulting ODF is displayed below. The
# red lobe is elongated while the blue lobe is flat.
# """
# # symmetric642 gives 10242 vertices after subdividing, as in [Riffert2014]_.
# # sphere = get_sphere('symmetric642').subdivide(2)
# sphere = get_sphere('repulsion724').subdivide(2)
# mevals = np.array([[0.0015, 0.00050, 0.00010],
#                    [0.0015, 0.00015, 0.00015]])
# angles = [(0, 0), (60, 0)]
# odf = multi_tensor_odf(sphere.vertices, mevals, angles, [50, 50])

# # Render the simulated ODF.
# odf_actor = actor.odf_slicer(odf[None, None, None, :], sphere=sphere,
#                              norm=False)
# odf_actor.RotateX(90)
# odf_actor.RotateZ(-40)

# scene = window.Scene()
# scene.add(odf_actor)

# window.snapshot(scene, 'bingham_in_odf.png')  # Maybe a confusing name for the image?
# scene.clear()

# """
# .. figure:: bingham_in_odf.png
#    :align: center

# We can now use the method `bingham_fit_odf` to fit a Bingham distribution
# to each fiber population describing our ODF. The output of the method is a
# list of tuples, each of which contains the parameters describing a Bingham
# distribution. Below, the parameters for both lobes are printed. The
# concentration parameters k1 and k2 describe how sharp the distribution is along
# its principal axes. As expected, the concentration parameters (k1 and k2) of
# the red lobe are equal, whereas the blue lobe has two different concentration
# parameters.
# """

# # list of coefficients; one per ODF lobe
# [fits, _] = bingham_fit_odf(odf, sphere, npeaks=3, max_search_angle=6,
#                             min_sep_angle=60, rel_th=0.1)

# # print the parameters of each Bingham distribution
# colors = ['red', 'blue']
# for i, (f0, k1, k2, mu0, mu1, mu2) in enumerate(fits):
#     print('Fiber population {} ({} lobe):'.format(i, colors[i]))
#     print('  Max amplitude (f0):', f0)  # Have to decide about a correct name for this. AFD as in Mrtrix?
#     print('  Concentration parameters (k1, k2):', k1, k2)
#     print('')

# """
# From the fitted Bingham distributions, we can compute fiber specific metrics
# such as the fiber density (FD) and fiber spread (FS). FD corresponds to the
# integral of the Bingham distribution over the sphere and describes the apparent
# quantity of fibers passing through an ODF lobe. FS describes the spread of the
# ODF lobe. A high FS corresponds to a wide lobe and a low FS, to a sharp one.
# """
# fd = bingham_fiber_density(fits)
# fs = bingham_fiber_spread(fits)
# odi = bingham_orientation_dispersion(fits)

# for it, (fdi, fsi, odii) in enumerate(zip(fd, fs, odi)):
#     print('Fiber population {} ({} lobe)'.format(it, colors[it]))
#     print('  Fiber density:', fdi)
#     print('  Fiber spread:', fsi)
#     print('  Orientation dispersion indexes (odi_1, odi_2):', odii[0], odii[1])
#     print('')

# """
# Finally, we can validate the quality of our Bingham fit by visualizing the
# Bingham distributions overlaid on the input ODF.
# """
# actors = []
# actors.append(actor.odf_slicer(odf[None, None, None, :], sphere=sphere,
#                                opacity=0.5, colormap=(255, 255, 255),
#                                norm=False))
# for f0, k1, k2, mu0, mu1, mu2 in fits:
#     sf = bingham_odf(f0, k1, k2, mu1, mu2, sphere.vertices)
#     actors.append(actor.odf_slicer(sf[None, None, ...], sphere=sphere,
#                                    opacity=0.7, norm=False))

# # rotate all actors by 90 degrees for better visualization.
# for a in actors:
#     a.RotateX(90)
#     a.RotateZ(-40)

# scene = window.Scene()
# scene.add(*actors)

# window.snapshot(scene, 'bingham_fit_overlay.png')
# scene.clear()
# """
# .. figure:: bingham_fit_overlay.png
#    :align: center


# References
# ----------
# .. [Riffert2014] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond
#                  fractional anisotropy: Extraction of bundle-specific
#                  structural metrics from crossing fiber models. NeuroImage.
#                  2014 Oct 15;100:176-91.
# """
