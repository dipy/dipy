# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:37:41 2018

@author: Shreyas
"""
"""
======================
MultiTensor Simulation
======================

In this example we show how someone can simulate the signal three voxels using a Stick and Ball Model.
"""
from dipy.sims.voxel import sticks_and_ball, SingleTensor
import dipy.reconst.dti as dti

# for testing 
from numpy.testing import assert_array_equal, assert_array_almost_equal
                             

"""
For the simulation we will need a GradientTable with the b-values and b-vectors
Here we use the one we created in :ref:`example_gradients_spheres`.
"""

from gradients_spheres import gtab

"""
In ``angles`` we save in polar coordinates (:math:`\theta, \phi`) the principal
axis of each tensor.
"""

angles = [(0, 0)]

"""
In ``fractions`` we save the percentage of the contribution of each tensor.
"""

fractions = [100]

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


"""

Now we try to fit the data using DTIFIT
"""
dtimodel = dti.TensorModel(gtab)
dtifit = dtimodel.fit(signal)

"""
Below we extract the fractional anisotropy (FA) and the mean diffusivity (MD) of the diffusion tensor.
"""
dti_FA = dtifit.fa
dti_MD = dtifit.md

print("The values obtained after the fit are: ", dtifit.evals, ", which is almost equal to the diffusivity index that we had declared = 0.0015. Note that we only care about the 1st eigenvalue since we are assigning 100% of the volume fraction to it.")

"""
Testing using assertions.
"""
def test_sticks_and_ball():
    d = 0.0015
    S, sticks = sticks_and_ball(gtab, d=d, S0=1, angles=[(0, 0), ],
                                fractions=[100], snr=None)
    assert_array_equal(sticks, [[0, 0, 1]])
    S_st = SingleTensor(gtab, 1, evals=[d, 0, 0], evecs=[[0, 0, 0],
                                                         [0, 0, 0],
                                                         [1, 0, 0]])
    assert_array_almost_equal(S, S_st)
    
test_sticks_and_ball()
print("-----------------------------------------------------Test Case Passed!")