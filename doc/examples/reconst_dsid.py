"""
==================================
Reconstruct with DSI Deconvolution
==================================

An alternative method to DSI is the method proposed by [Canales10]_ which is
called DSI with Deconvolution. This algorithm is using Lucy-Richardson
deconvolution in the diffusion propagator with the goal to create sharper ODFs
with higher angular resolution.

In this example we will show how this method performs against standard DSI and
a ground truth multi tensor ODF.
"""

from dipy.sims.voxel import multi_tensor, multi_tensor_odf
from dipy.data import get_data, get_sphere
from dipy.core.gradients import gradient_table
from dipy.reconst.dsi import (DiffusionSpectrumDeconvModel,
                              DiffusionSpectrumModel)

"""
For the simulation we will use a standard DSI acqusition scheme with 514
gradient directions and 1 b-value=0.
"""

btable = np.loadtxt(get_data('dsi515btable'))

gtab = gradient_table(btable[:, 0], btable[:, 1:])

"""
Let's create a multi tensor with 2 fiber directions at 60 degrees.
"""

evals = np.array([[0.0015, 0.0003, 0.0003],
                  [0.0015, 0.0003, 0.0003]])

directions = [(-30, 0), (30, 0)]

fractions = [50, 50]

signal, _ = multi_tensor(gtab, evals, 100, angles=directions,
                         fractions=fractions, snr=None)

sphere = get_sphere('symmetric724').subdivide(1)

odf_gt = multi_tensor_odf(sphere.vertices, evals, angles=directions,
                          fractions=fractions)

"""
Now, we can fit
"""

dsi_model = DiffusionSpectrumModel(gtab)

dsi_odf = dsi_model.fit(signal).odf(sphere)


dsid_model = DiffusionSpectrumDeconvModel(gtab)

dsid_odf = dsid_model.fit(signal).odf(sphere)


from dipy.viz import fvtk

ren = fvtk.ren()

odfs = np.vstack((odf_gt, dsi_odf, dsid_odf))[:, None, None]

odf_actor = fvtk.sphere_funcs(odfs, sphere)
odf_actor.RotateX(90)
fvtk.add(ren, odf_actor)

fvtk.show(ren)
fvtk.record(ren, path_out='dsid.png', size=(600, 600))

"""
.. figure:: dsid.png
    :align: center

    **Ground truth ODF (left), DSI ODF (middle), DSI with Deconvolution ODF(right)**

.. [Canales10] Canales-Rodriguez et al., Deconvolution in Diffusion Spectrum Imaging,
			   Neuroimage, vol 50, no 1, 136-149, 2010.

"""
