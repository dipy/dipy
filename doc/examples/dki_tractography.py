"""
===============================================================
Exploring the 3D information of DKI and use it for Tractography
===============================================================

In this example we show how to analyze the 3D information of the tensors
estimated from diffusion kurtosis imaging (DKI) and how DKI can be used for
tractography. This example is based on the work done by [Raf2015]_.

All relevant modules for this example are imported below:
"""
import numpy as np
import dipy.reconst.dki as dki
from dipy.sims.voxel import multi_tensor_dki
from dipy.data import get_sphere
from dipy.viz import fvtk
# in future we have to import the right data
from dipy.data import fetch_sherbrooke_3shell
from dipy.data import read_sherbrooke_3shell

"""
For this example we use data ... :
"""

# in future we have to import the right data
fetch_sherbrooke_3shell()
img, gtab = read_sherbrooke_3shell()
data = img.get_data()

"""
We first illustrate the 3D information provided by DKI on simulates. For this,
we use the GradientTable of the loaded real data sample and simulate the signal
of five voxels of two fibers with increasing crossing angle. (Add reference to
the DKI simulation example for more details)
"""

mevals = np.array([[0.00099, 0, 0],
                   [0.00226, 0.00087, 0.00087],
                   [0.00099, 0, 0],
                   [0.00226, 0.00087, 0.00087]])
angles = [(90, 0), (90, 0), (20, 0), (20, 0)]
fie = 0.49  # intra axonal water fraction
fractions = [fie*50, (1 - fie)*50, fie*50, (1 - fie)*50]
increasing_angles = np.array([[(90, 0), (90, 0), (90, 0), (90, 0)],
                              [(90, 0), (90, 0), (60, 0), (60, 0)],
                              [(90, 0), (90, 0), (45, 0), (45, 0)],
                              [(90, 0), (90, 0), (30, 0), (30, 0)],
                              [(90, 0), (90, 0), (0, 0), (0, 0)]])

angles = increasing_angles[-2]

signal_dki, dt, kt = multi_tensor_dki(gtab, mevals, S0=200, angles=angles,
                                      fractions=fractions, snr=None)

""" (Fit DKI to simulates) Similar to the example for estimating kurtosis
statistical measures, we fit the data with a defined Diffusion Kurtosis Model
which is reconstructed using data's GradientTable """

dkimodel = dki.DiffusionKurtosisModel(gtab)
dkifit = dkimodel.fit(signal_dki)

""" (Visualization the directional distribution of apparent diffusivity values,
apparent kurtosis values and the DKI based ODF).
"""

sphere = get_sphere('symmetric724').subdivide(1)

ADC = dki.apparent_diffusion_coef(dkifit.quadratic_form, sphere)
AKC = dki.apparent_kurtosis_coef(dkifit.model_params, sphere)
ODF = dkifit.dki_odf(sphere)


ren = fvtk.ren()
odfs = np.vstack((ADC, AKC, ODF))
odf_actor = fvtk.sphere_funcs(odfs, sphere)
odf_actor.RotateX(90)
fvtk.add(ren, odf_actor)
fvtk.record(ren, out_path='dki_geometries.png', size=(300, 300))

""" (Add figure directive, explain what we see in the figure)
(Introduce the real data fitting)
"""

""" (Visualize the DKI-ODF of real data similar to the example reconst_csd)

dkiodf = dkifit.odf(sphere)
"""

"""
References:

.. [NetoHe2015] Neto Henriques R, Correia MM, Nunes RG, Ferreira HA (2015).
                Exploring the 3D geometry of the diffusion kurtosis tensor -
                Impact on the development of robust tractography procedures and
                novel biomarkers, NeuroImage 111: 85-99

.. include:: ../links_names.inc

"""

