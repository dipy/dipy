"""
========================
DSI Deconvolution vs DSI
========================

An alternative method to DSI is the method proposed by [Canales10]_ which is
called DSI with Deconvolution. This algorithm is using Lucy-Richardson
deconvolution in the diffusion propagator with the goal to create sharper ODFs
with higher angular resolution.

In this example we will show with simulated data how this method's ODF
performs against standard DSI ODF and a ground truth multi tensor ODF.
"""

import numpy as np
from dipy.sims.voxel import multi_tensor, multi_tensor_odf
from dipy.data import get_fnames, get_sphere
from dipy.core.gradients import gradient_table
from dipy.reconst.dsi import (DiffusionSpectrumDeconvModel,
                              DiffusionSpectrumModel)
from dipy.viz import window, actor

###############################################################################
# For the simulation we will use a standard DSI acquisition scheme with 514
# gradient directions and 1 S0.

btable = np.loadtxt(get_fnames('dsi515btable'))

gtab = gradient_table(btable[:, 0], btable[:, 1:])

###############################################################################
# Let's create a multi tensor with 2 fiber directions at 60 degrees.

evals = np.array([[0.0015, 0.0003, 0.0003],
                  [0.0015, 0.0003, 0.0003]])

directions = [(-30, 0), (30, 0)]

fractions = [50, 50]

signal, _ = multi_tensor(gtab, evals, 100, angles=directions,
                         fractions=fractions, snr=None)

sphere = get_sphere('repulsion724').subdivide(1)

odf_gt = multi_tensor_odf(sphere.vertices, evals, angles=directions,
                          fractions=fractions)

###############################################################################
# Perform the reconstructions with standard DSI and DSI with deconvolution.

dsi_model = DiffusionSpectrumModel(gtab)

dsi_odf = dsi_model.fit(signal).odf(sphere)

dsid_model = DiffusionSpectrumDeconvModel(gtab)

dsid_odf = dsid_model.fit(signal).odf(sphere)

###############################################################################
# Finally, we can visualize the ground truth ODF, together with the DSI and DSI
# with deconvolution ODFs and observe that with the deconvolved method it is
# easier to resolve the correct fiber directions because the ODF is sharper.

# Enables/disables interactive visualization
interactive = False


scene = window.Scene()

# concatenate data as 4D array
odfs = np.vstack((odf_gt, dsi_odf, dsid_odf))[:, None, None]
odf_actor = actor.odf_slicer(odfs, sphere=sphere, scale=0.5, colormap='plasma')

odf_actor.display(y=0)
odf_actor.RotateX(90)
scene.add(odf_actor)
window.record(scene, out_path='dsid.png', size=(300, 300))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Ground truth ODF (left), DSI ODF (middle), DSI with Deconvolution ODF (right).
#
#
# References
# ----------
# .. [Canales10] Canales-Rodriguez et al., Deconvolution in Diffusion Spectrum
#                Imaging, Neuroimage, vol 50, no 1, p. 136-149, 2010.
