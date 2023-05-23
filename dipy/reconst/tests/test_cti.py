import matplotlib.pyplot as plt
import numpy as np
import math

from dipy.reconst.tests.test_qti import  _anisotropic_DTD, _isotropic_DTD 
from dipy.core.gradients import gradient_table 
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
from dipy.sims.voxel import multi_tensor

# Simulation: signals of two crossing fibers are simulated
n_pts = 20 #points are assumed to be on a sphere
theta = np.pi * np.random.rand(n_pts) #theta: angle betn point P and z-axis
phi = 2 * np.pi * np.random.rand(n_pts) #value ranges between 0 to n
hsph_initial = HemiSphere(theta=theta, phi=phi)
hsph_updated, potential = disperse_charges(hsph_initial, 5000) 
#defining bvecs1, bvals1
bvecs1 = np.concatenate([hsph_updated.vertices] * 4) #total 4 x 20 + 1 = 81 vectors 
bvecs1 = np.append(bvecs1 , [[0, 0, 0]], axis=0)
bvals1 = np.array([2] * 20 + [1] * 20 + [1] * 20 + [1] * 20 + [0])
#deifining DDE protocol for CTI
btens = np.array(['LTE' for i in range(20)] + ['PTE' for i in range(20)] + ['STE' for i in range(20)] + ['CTE' for i in range(21)])

#on providing btens, (bvals1,bvecs1) is ignored.
gtab = gradient_table(bvals1, bvecs1, btens = btens) #gradients (81,3) will be defined, earlier on, not sure
#we've isotropic and anisotropic diffusion tensor distribution (DTD)
 
angles = [(0, 0), (90, 0)] #both tesnsors are perpendicular to each other: to simulate the presence of crossing fibers
fractions = [50, 50] #each tensor contribution
#now we can have the isotropic and the anisotropic signals. 

