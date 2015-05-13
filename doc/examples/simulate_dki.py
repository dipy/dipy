"""
==========================
DKI MultiTensor Simulation
==========================

In this example we show how someone can simulate the diffusion tensor,
diffusion kurtosis tensor and signals of a single voxel by taking into account 
different cellular media using a multi-compartment simulatations. Simulations 
are preformed according to: 

[1] R. Neto Henriques et al., "Exploring the 3D geometriy of the diffusion 
    kurtosis tensor - Impact on the development of robust tractography 
    procedures and novel biomarkers", NeuroImage (2015) 111, 85-99.
"""

import numpy as np
from dipy.sims.voxel import multi_tensor_dki

"""
For the simulation we will need a GradientTable with the b-values and b-vectors
Here we use the one we created in :ref:`example_gradients_spheres`.
"""

from gradients_spheres import gtab

"""
In ``mevals`` we save the eigenvalues of each tensor. To simulate crossing 
fibers with two different media (representing intra and extra-cellular media)
4 tissue components are taken in to account (the first two compartments 
correspond to the intra and extra cellular media for the first fiber population
while the others correspond to the media of the second fiber population)
"""

mevals = np.array([[0.00099, 0, 0],
                   [0.00226, 0.00087, 0.00087],
                   [0.00099, 0, 0],
                   [0.00226, 0.00087, 0.00087],])

"""
In ``angles`` we save in polar coordinates (:math:`\theta, \phi`) the principal
axis of each compartment tensor. To simulate crossing fibres at 70 degrees
the compartments corresponding to the first fibre are aligned to the x-axis 
while the compartments corresponding to the second fibre are aligned to the
x-z plane with a angular deviation of 70 degrees from the first one.
"""

angles = [(90, 0), (90, 0), (20, 0), (20, 0)]

"""
In ``fractions`` we save the percentage of the contribution of each tensor, 
which is computed by multipling the percentage of contribution of each fiber 
populatuion and the water fraction corresponding each fiber media 
"""

fie = 0.49 # intra axonal water fraction
fractions = [fie*50, (1-fie)*50, fie*50, (1-fie)*50]

"""
Having defined the parameters for all tissue compartments, the elements of the
diffusion tensor (dt), the elements of the kurtosis tensor (KT) and the DW 
signals simulated from the DKI model can be obtain using the function 
``multi_tensor_dki``.
"""

signal, dt, kt = multi_tensor_dki(gtab, mevals, S0=200, angles=angles,
                         fractions=fractions, snr=None)

"""
We can also add rician noise with a specific SNR.
"""

signal_noisy, dt, kt  = multi_tensor_dki(gtab, mevals, S0=200, 
                         angles=angles, fractions=fractions, snr=10)


import matplotlib.pyplot as plt

plt.plot(signal, label='noiseless')

plt.plot(signal_noisy, label='with noise')
plt.legend()
plt.show()
plt.savefig('simulated_signal.png')