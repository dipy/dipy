# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:37:41 2018

@author: Shreyas
"""
"""
======================
MultiTensor Simulation
======================

In this example we show how someone can simulate the signal three voxels using a MultiTensor.
"""
import numpy as np
from dipy.sims.voxel import sticks_and_ball
                             

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
signal, sticks_ball = sticks_and_ball(gtab, d=0.0015, S0=100, angles=angles,
                         fractions=fractions, snr=None)

"""
We can also add Rician noise with a specific SNR.
"""

signal_noisy, sticks_ball = sticks_and_ball(gtab, d=0.0015, S0=100, angles=angles,
                         fractions=fractions, snr=20)


import matplotlib.pyplot as plt

plt.plot(signal, label='noiseless')

plt.plot(signal_noisy, label='with noise')
plt.legend()
plt.show()
# plt.savefig('simulated_signal.png')
