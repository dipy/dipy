"""
============================
Q-SpaceTrajectory Simulation
============================

The most basic assumption of qti states that tissues can be denoted by multiple gaussian component.
QTI can only handle a single gtab at a time, so in this tutorial we'll start by taking 2 gtabs then combine them into a single btensor. 
"""

import matplotlib.pyplot as plt
import numpy as np
import math

from dipy.reconst.tests.test_qti import  _anisotropic_DTD, _isotropic_DTD 
from dipy.core.gradients import gradient_table 
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
from dipy.sims.voxel import multi_tensor

"""
We'll take the case of 2 gtabs and then combine them into a single btensor as for qti model, we need a single gtab.
"""

""" 
Creating the first gtab 
""" 

n_pts = 20
theta = np.pi * np.random.rand(n_pts) 
phi = 2 * np.pi * np.random.rand(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)
hsph_updated, potential = disperse_charges(hsph_initial, 5000) 
bvecs1 = np.concatenate([hsph_updated.vertices] * 4)
bvecs1 = np.append(bvecs1 , [[0, 0, 0]], axis=0)
bvals1 = np.array([2] * 20 + [1] * 20 + [1] * 20 + [1] * 20 + [0])
gtab1 = gradient_table(bvals1, bvecs1)

""" 
For the second gradient table, we'll consider a set of directions which are perpendicular relative to the generated hsph_updated. We'll call it hsph_updated90.
Using the function perpendicular_directions, we can get the perpendicular tensors.
"""

def perpendicular_directions(v, num=20, half=False):
    v = np.array(v, dtype=np.float64)
    v = v.T 
    er = np.finfo(v[0].dtype).eps * 1e3
    if half is True:
        a = np.linspace(0., math.pi, num=num, endpoint=False)
    else:
        a = np.linspace(0., 2 * math.pi, num=num, endpoint=False)
    cosa = np.cos(a) #(20,)
    sina = np.sin(a)
    if np.any(abs(v[0] - 1.) > er):
        sq = np.sqrt(v[1]**2 + v[2]**2)
        psamples = np.array([- sq*sina, (v[0]*v[1]*sina - v[2]*cosa) / sq,
                             (v[0]*v[2]*sina + v[1]*cosa) / sq])
    else:
        sq = np.sqrt(v[0]**2 + v[2]**2)
        psamples = np.array([- (v[2]*cosa + v[0]*v[1]*sina) / sq, sina*sq,
                             (v[0]*cosa - v[2]*v[1]*sina) / sq])
    return psamples.T
    

"""
Using the above function to get the perpendicular vectors.
""" 

hsph_updated90 = perpendicular_directions(hsph_updated.vertices)

"""
We can check that the verctors are indeed perpendicular by taking dot product.
"""

dot_product = np.sum(hsph_updated.vertices * hsph_updated90, axis=1)
are_perpendicular = np.isclose(dot_product, 0)

bvecs2 = np.concatenate(([hsph_updated.vertices] * 2) + [hsph_updated90] + ([hsph_updated.vertices]))
bvecs2 = np.append(bvecs2, [[0, 0, 0]], axis=0)

bvals2 =  np.array([0] * 20 + [1] * 20 + [1] * 20 + [0] * 20 + [0])
gtab2 = gradient_table(bvals2, bvecs2)

"""
Creating the btensor.
Defining btensor helps to capture more complex tissue microstructure information than either DTI or DKI. It also simplifies the information as now rather than having to consider 2 gtabs, we need to consider only one. 
"""

e1 = bvecs1 
e2 = bvecs2
e3 = np.cross(e1,e2) 
V = np.stack((e1, e2, e3), axis=-1)
V_transpose = np.transpose(V, axes=(0, 2, 1))

"""
Defining the Btensor
"""
B = np.zeros((81, 3, 3)) #initializing a btensor
b = np.zeros((3, 3))
for i in range(81):
    b[0, 0] = bvals1[i]
    b[1, 1] = bvals2[i]
    B[i] = np.matmul(V[i], np.matmul(b, V_transpose[i]))
    
"""
Creating the equivalent gradient table
"""
gtab = gradient_table(bvals1, bvecs1, btens = B) 
angles = [(0, 0), (90, 0)] #both of them are perpendicular to each other
fractions = [50, 50] 

# Generate anisotropic signal
anisotropic_DTD = _anisotropic_DTD()
isotropic_DTD = _isotropic_DTD() 
signal_aniso, sticks = multi_tensor(gtab, anisotropic_DTD, S0=100, angles=angles, fractions=fractions, snr=None)
# Generate isotropic signal
signal_iso, sticks = multi_tensor(gtab, isotropic_DTD, S0=100, angles = angles, fractions= fractions, snr=None)


plt.plot(signal_aniso, label='anisotropic_signals')
plt.plot(signal_iso, label = 'isotropic_signals')
plt.legend()
plt.show()
















