"""
=================================================
Calculate SHORE metrics
=================================================

We show how to calculate two SHORE-based scalar metrics: return to origin 
probability (rtop) [Descoteaux2011]_ and mean square displacement (msd) 
[Wu2007]_, [Wu2008]_ on your dataset.

First import the necessary modules:
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi, get_sphere
from dipy.data import get_data, dsi_voxels
from dipy.reconst.shore import ShoreModel

"""
Download and read the data for this tutorial.
"""

fetch_taiwan_ntu_dsi()
img, gtab = read_taiwan_ntu_dsi()

"""
img contains a nibabel Nifti1Image object (data) and gtab contains a GradientTable
object (gradient information e.g. b-values). For example to read the b-values
it is possible to write print(gtab.bvals).

Load the raw diffusion data and the affine.
"""

data = img.get_data()
affine = img.get_affine()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
Instantiate the Model.
"""

asm = ShoreModel(gtab)

"""
Lets just use one slice only from the data.
"""

dataslice = data[30:70, 20:80, data.shape[2] / 2]

"""
Normalize the signal by the b0
"""

dataslice = dataslice / (dataslice[..., 0, None]).astype(np.float)

"""
Fit the signal with the model and calculate the SHORE coefficients.
"""

asmfit = asm.fit(dataslice)

"""
Calculate the analytical return to origin probability on the signal 
that corresponds to the integral of the signal.
"""

print('Calculating... rtop_signal')
rtop_signal = asmfit.rtop_signal()

"""
Now we calculate the analytical return to origin probability on the propagator, 
that corresponds to its central value.
"""

print('Calculating... rtop_pdf')
rtop_pdf = asmfit.rtop_pdf()
"""
In theory, these two measures must be equal, 
to show that we calculate the mean square error on this two measures.
"""

mse = np.sum((rtop_signal - rtop_pdf) ** 2) / rtop_signal.size
print("mse = %f" % mse)

""" 
mse = 0.000000

Let's calculate the analytical mean square displacement on the propagator.
"""

print('Calculating... msd')
msd = asmfit.msd()

"""
Show the metrics images and save them in SHORE_metrics.png.
"""

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(2, 2, 1, title='rtop_signal')
ax1.set_axis_off()
ind = ax1.imshow(rtop_signal.T, interpolation='nearest', origin='lower')
plt.colorbar(ind)
ax2 = fig.add_subplot(2, 2, 2, title='rtop_pdf')
ax2.set_axis_off()
ind = ax2.imshow(rtop_pdf.T, interpolation='nearest', origin='lower')
plt.colorbar(ind)
ax3 = fig.add_subplot(2, 2, 3, title='msd')
ax3.set_axis_off()
ind = ax3.imshow(msd.T, interpolation='nearest', origin='lower', vmin=0)
plt.colorbar(ind)
plt.savefig('SHORE_metrics.png')

"""
.. figure:: SHORE_metrics.png
   :align: center

   **rtop and msd calculated using the SHORE model**.


.. [Descoteaux2011] Descoteaux M. et. al , "Multiple q-shell diffusion 
					propagator imaging", Medical Image Analysis, vol 15,
					No. 4, p. 603-621, 2011.

.. [Wu2007] Wu Y. et. al, "Hybrid diffusion imaging", NeuroImage, vol 36,
        	p. 617-629, 2007.

.. [Wu2008] Wu Y. et. al, "Computation of Diffusion Function Measures
			in q -Space Using Magnetic Resonance Hybrid Diffusion Imaging",
			IEEE TRANSACTIONS ON MEDICAL IMAGING, vol. 27, No. 6, p. 858-865,
			2008

.. include:: ../links_names.inc

"""