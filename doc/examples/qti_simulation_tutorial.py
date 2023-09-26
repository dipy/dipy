"""
============================
Q-SpaceTrajectory Simulation
============================

In this tutorial, we demonstrate how to generate synthetic signals using
b-tensor valued encodings. These are invaluable for acquiring synthetic signals
from advanced diffusion encodings, working under the premise that tissues can
be represented as a summation of multiple Gaussian components.

QTI is designed to handle a single gtab at a time. In this tutorial, we will
start by combining two gtabs into a single btensor.

First, let's import all the relevant libraries and modules for this simulation.
"""
import matplotlib.pyplot as plt
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.core.sphere import disperse_charges, HemiSphere
from dipy.sims.voxel import multi_tensor

"""
We'll take the case of two gtabs and combine them into a single btensor.
"""
n_pts = 30
theta = np.pi * np.random.rand(n_pts)
phi = 2 * np.pi * np.random.rand(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)
hsph_updated, potential = disperse_charges(hsph_initial, 5000)
# These will correspond to the main directions of the btensors.
bvecs = np.concatenate([hsph_updated.vertices] * 4)
bvals = np.array([1000] * 30 + [2000] * 30 + [1000] * 30 + [2000] * 30)

# Define b-tensor encodings
btens = np.array(['LTE' for i in range(60)] + ['PTE' for i in range(60)])

# Create the gradient table with b-tensor encodings
gtab = gradient_table(bvals, bvecs, btens=btens)

# Define the eigenvalues for the tensors
mevals = np.array([[0.0015, 0.0003, 0.0003],
                   [0.0015, 0.0003, 0.0003]])

# Define the principal axis of each tensor in polar coordinates (θ, ϕ)
angles = [(0, 0), (60, 0)]

# Define the contribution percentage of each tensor
fractions = [50, 50]

# Generate the simulated signal using the multi_tensor function
signal, sticks = multi_tensor(gtab, mevals, S0=100, angles=angles,
                              fractions=fractions, snr=None)

plt.plot(signal, label='signals')
plt.legend()
plt.show()
