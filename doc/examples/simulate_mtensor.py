"""
==================================
Simulations using Multiple Tensors
==================================
"""

import numpy as np
from dipy.sims.voxel import (multi_tensor,
                             multi_tensor_odf,
                             all_tensor_evecs)
from dipy.core.gradients import gradient_table
from dipy.reconst.peaks import peak_directions
from dipy.data import get_sphere, get_data


SNR = None
S0 = 100

# two equal fibers (creating a very sharp odf)
mevals = np.array([[0.0015, 0.0003, 0.0003],
                   [0.0015, 0.0003, 0.0003]])
angles = [(0, 45), (-45, 0)]
fractions = [50, 50]

_, fbvals, fbvecs = get_data('small_64D')

bvals = np.load(fbvals)
bvecs = np.load(fbvecs)

gtab = gradient_table(bvals, bvecs)

S, sticks = multi_tensor(gtab, mevals, S0, angles=angles,
                         fractions=fractions, snr=SNR)

sphere = get_sphere('symmetric724').subdivide(2)

mevecs = []
for i in range(sticks.shape[0]):
    mevecs += [all_tensor_evecs(sticks[i]).T]

fractions = np.array(fractions)
fracts = fractions / 100.

odf_gt = multi_tensor_odf(sphere.vertices, fracts, mevals, mevecs)

directions, values, indices = peak_directions(odf_gt, sphere, .5, 25.)


from dipy.viz import fvtk

ren = fvtk.ren()

odf_actor = fvtk.sphere_funcs(odf_gt, sphere)
odf_actor.RotateX(90)

fvtk.add(ren, odf_actor)

fvtk.record(ren, out_path='multi_tensor_simulation.png', size=(300, 300))

"""
.. figure:: multi_tensor_simulation.png
   :align: center

   **Simulating a MultiTensor**.
"""
