"""
==================================
Reconstruct with DSI Deconvolution
==================================

An alternative method to DSI is the method proposed by [Canales10]_ which is
called DSI
"""

from dipy.sims.voxel import multi_tensor, multi_tensor_odf

from dipy.data import get_data

btable = np.loadtxt(get_data('dsi515btable'))

from dipy.core.gradients import gradient_table

gtab = gradient_table(btable[:, 0], btable[:, 1:])



evals = np.array([[0.0015, 0.0003, 0.0003],
                  [0.0015, 0.0003, 0.0003]])

directions = [(-30, 0), (30, 0)]

fractions = [50, 50]

SNR = 100

signal, _ = multi_tensor(gtab, evals, 100, angles=directions,
                         fractions=fractions, snr=SNR)

from dipy.data import get_sphere

sphere = get_sphere('symmetric724').subdivide(1)

odf_gt = multi_tensor_odf(sphere.vertices, evals, angles=directions,
                          fractions=fractions)


from dipy.reconst.dsi import (DiffusionSpectrumDeconvModel,
                              DiffusionSpectrumModel)

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


"""
.. [Canales10] Canales-Rodriguez et al., Deconvolution in Diffusion Spectrum Imaging,
			   Neuroimage, vol 50, no 1, 136-149, 2010.

"""
