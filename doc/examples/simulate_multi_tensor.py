"""
======================
MultiTensor Simulation
======================

In this example we show how someone can simulate the signal and the ODF of a
single voxel using a MultiTensor.
"""

import numpy as np
from dipy.sims.voxel import (multi_tensor,
                             multi_tensor_odf,
                             single_tensor_odf,
                             all_tensor_evecs)
from dipy.data import get_sphere

"""
For the simulation we will need a GradientTable with the b-values and b-vectors
Here we use the one we created in :ref:`example_gradients_spheres`.
"""

from gradients_spheres import gtab

"""
In ``mevals`` we save the eigenvalues of each tensor.
"""

mevals = np.array([[0.0015, 0.0003, 0.0003],
                   [0.0015, 0.0003, 0.0003]])

"""
In ``angles`` we save in polar coordinates (:math:`\theta, \phi`) the principal
axis of each tensor.
"""

angles = [(0, 0), (60, 0)]

"""
In ``fractions`` we save the percentage of the contribution of each tensor.
"""

fractions = [50, 50]

"""
The function ``multi_tensor`` will return the simulated signal and an array
with the principal axes of the tensors in cartesian coordinates.
"""

signal, sticks = multi_tensor(gtab, mevals, S0=100, angles=angles,
                         fractions=fractions, snr=None)

"""
We can also add rician noise with a specific SNR.
"""

signal_noisy, sticks = multi_tensor(gtab, mevals, S0=100, angles=angles,
                         fractions=fractions, snr=20)


import matplotlib.pyplot as plt

plt.plot(signal, label='noiseless')

plt.plot(signal_noisy, label='with noise')
plt.legend()
plt.show()
plt.savefig('simulated_signal.png')

"""
.. figure:: simulated_signal.png
   :align: center

   **Simulated MultiTensor signal**
"""

"""
For the ODF simulation we will need a sphere. Because we are interested in a
simulation of only a single voxel, we can use a sphere with very high
resolution. We generate that by subdividing the triangles of one of Dipy's
cached spheres, which we can read in the following way.
"""

sphere = get_sphere('symmetric724')
sphere = sphere.subdivide(2)

odf = multi_tensor_odf(sphere.vertices, mevals, angles, fractions)

from dipy.viz import fvtk

ren = fvtk.ren()

odf_actor = fvtk.sphere_funcs(odf, sphere)
odf_actor.RotateX(90)

fvtk.add(ren, odf_actor)

print('Saving illustration as multi_tensor_simulation')
fvtk.record(ren, out_path='multi_tensor_simulation.png', size=(300, 300))

"""
.. figure:: multi_tensor_simulation.png
   :align: center

   **Simulating a MultiTensor ODF**
"""
