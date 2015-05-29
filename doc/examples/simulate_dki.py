"""
==========================
DKI MultiTensor Simulation
==========================

In this example we show how to simulate the diffusion kurtosis imaging (DKI)
data of a single voxel. DKI captures information about tissue heterogeneity.
Therefore DKI simulations have to take into account different tissue
compartments with different diffusion properties. For example, here diffusion
heterogeneity is taken into account by modeling different compartments for the
intra- and extra-cellular media. These simulations are performed according to
[RNH2015]_.

We first import all relevant modules.
"""

import numpy as np
import matplotlib.pyplot as plt
from dipy.sims.voxel import multi_tensor_dki
from dipy.data import get_data
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

"""
For the simulation we will need a Gradient Table with the b-values and
b-vectors. Here we use the Gradient table of the sample Dipy dataset
'small_64D'.
"""

fimg, fbvals, fbvecs = get_data('small_64D')
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)

"""
DKI requires data from more than one non-zero b-value. Since the dataset
'small_64D' was acquired with one non-zero bvalue we artificialy produce a
second non-zero b-value.
"""

bvals = np.concatenate((bvals, bvals * 2), axis=0)
bvecs = np.concatenate((bvecs, bvecs), axis=0)

"""
The b-values and gradient directions are then converted to Dipy's Gradient
Table format.
"""

gtab = gradient_table(bvals, bvecs)

"""
In ``mevals`` we save the eigenvalues of each tensor. To simulate crossing
fibers with two different media (representing intra and extra-cellular media),
a total of four components have to be taken in to account (i.e. the first two
compartments correspond to the intra and extra cellular media for the first
fiber population while the others correspond to the media of the second fiber
population)
"""

mevals = np.array([[0.00099, 0, 0],
                   [0.00226, 0.00087, 0.00087],
                   [0.00099, 0, 0],
                   [0.00226, 0.00087, 0.00087]])

"""
In ``angles`` we save in polar coordinates (:math:`\theta, \phi`) the principal
axis of each compartment tensor. To simulate crossing fibers at 70 degrees
the compartments of the first fiber are aligned to the x-axis while the
compartments of the second fiber are aligned to the x-z plane with an angular
deviation of 70 degrees from the first one.
"""

angles = [(90, 0), (90, 0), (20, 0), (20, 0)]

"""
In ``fractions`` we save the percentage of the contribution of each
compartment, which is computed by multiplying the percentage of contribution
of each fiber population and the water fraction of each different medium
"""

fie = 0.49  # intra axonal water fraction
fractions = [fie*50, (1 - fie)*50, fie*50, (1 - fie)*50]

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

signal_noisy, dt, kt = multi_tensor_dki(gtab, mevals, S0=200,
                                        angles=angles, fractions=fractions,
                                        snr=10)

"""
Finally, we can visualize the values of the signal simulated for the DKI for
all assumed gradient directions and bvalues.
"""

plt.plot(signal, label='noiseless')

plt.plot(signal_noisy, label='with noise')
plt.legend()
plt.show()
plt.savefig('simulated_signal.png')


"""
.. figure:: simulated_dki_signal.png
   :align: center
   **Simulated signals obtain from the DKI model**.


References:

[RNH2015] R. Neto Henriques et al., "Exploring the 3D geometry of the diffusion
          kurtosis tensor - Impact on the development of robust tractography
          procedures and novel biomarkers", NeuroImage (2015) 111, 85-99.
"""
