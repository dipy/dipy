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
For the ODF simulation we will need a sphere. Because, here we are
interested in a single voxel simulation we can use a sphere at very high
resolution by subdividing the triangles of one of the Dipy's default spheres.
"""

sphere = get_sphere('symmetric724')
sphere = sphere.subdivide(2)



def multi_tensor_odf_new(odf_verts, mf, mevals, angles):
    r'''Simulate a Multi-Tensor ODF.

    Parameters
    ----------
    odf_verts : (N,3) ndarray
        Vertices of the reconstruction sphere.
    mf : sequence of floats,
        Percentages of the fractions for each tensor.
    mevals : sequence of 1D arrays,
        Eigen-values for each tensor.
    angles : sequence of 2d tuples,
        principal directions for each tensor

    Returns
    -------
    ODF : (N,) ndarray
        Orientation distribution function.

    Examples
    --------
    Simulate a MultiTensor ODF with two peaks and calculate its exact ODF.

    >>> import numpy as np
    >>> from dipy.sims.voxel import multi_tensor_odf, all_tensor_evecs
    >>> from dipy.data import get_sphere
    >>> sphere = get_sphere('symmetric724')
    >>> vertices, faces = sphere.vertices, sphere.faces
    >>> mevals = np.array(([0.0015, 0.0003, 0.0003],[0.0015, 0.0003, 0.0003]))
    >>> angles = [(0, 0), (90, 0)]
    >>> odf = multi_tensor_odf(vertices, [50, 50], mevals, angles)

    '''
    from dipy.core.sphere import sphere2cart

    mf = [f / 100. for f in fractions]

    angles = np.array(angles)
    if angles.shape[-1] == 3:
        sticks = angles
    else:
        sticks = [sphere2cart(1, np.deg2rad(pair[0]), np.deg2rad(pair[1]))
                  for pair in angles]
        sticks = np.array(sticks)

    odf = np.zeros(len(sphere.vertices))

    mevecs = []
    for s in sticks:
        mevecs += [all_tensor_evecs(s).T]

    for (j, f) in enumerate(mf):
        odf += f * single_tensor_odf(sphere.vertices,
                                     evals=mevals[j], evecs=mevecs[j])
    return odf


odf = multi_tensor_odf_new(sphere, fractions, mevals, angles)


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
