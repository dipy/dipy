"""
=================================================
Calculate DSI metrics
=================================================

We show how to calculate DSI metrics return to origin probability (rtop) 
and mean square displacement (MSD) on your dataset.

First import the necessary modules:
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi, get_sphere
from dipy.data import get_data, dsi_voxels
from dipy.reconst.dsi import DiffusionSpectrumModel

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
Instantiate the Model and apply it to the data.
"""

dsmodel = DiffusionSpectrumModel(
    gtab, qgrid_size=35, r_start=0.4 * 17, r_end=0.7 * 17, filter_width=18.5)

"""
Lets just use one slice only from the data.
"""

dataslice = data[30:70, 20:80, data.shape[2] / 2]

"""
Normalize the signal by the b0
"""

dataslice = dataslice / (dataslice[..., 0, None]).astype(np.float)

"""
Calculate the return to origin probability on the signal 
that corresponds to the integral of the signal.
"""

print('Calculating... rtop_signal')
rtop_signal = dsmodel.fit(dataslice).rtop_signal()

"""
Now we calculate the return to origin probability on the propagator, 
that corresponds to its central value. 
By default the propagator is divided by its sum in order to obtain a pdf,
however this normalization changes the values of rtop, in order to compare it
with the rtop previously calculated on the signal we turn the normalized parameter to false.
"""

print('Calculating... rtop_pdf')
rtop_pdf = dsmodel.fit(dataslice).rtop_pdf(normalized=False)

"""
Following the theory this two measures must be equal, 
to proof that we calculate the mean square error on this two measures.
"""

mse = np.sum((rtop_signal - rtop_pdf) ** 2) / rtop_signal.size
print("mse = %f" % mse)

""" 
mse = 0.000000

Leaving normalized parameter to the default changes the values of the 
rtop but not the contrast between the voxels.
"""

print('Calculating... rtop_pdf_norm')
rtop_pdf_norm = dsmodel.fit(dataslice).rtop_pdf()

"""
Lets calculate the mean square displacement on the normalized propagator.
"""

print('Calculating... MSD_norm')
MSD_norm = dsmodel.fit(dataslice).msd_discrete()

"""
Turning the normalized parameter to false is possible to calculate 
the mean square displacement on the propagator without normalization.
"""

print('Calculating... MSD')
MSD = dsmodel.fit(dataslice).msd_discrete(normalized=False)

"""
Save the rtop images in rtop.png in order to compare the contrast. 
"""

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title('rtop_signal')
ind = ax1.imshow(rtop_signal.T, interpolation='nearest', origin='lower')
plt.colorbar(ind)
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('rtop_pdf_norm')
ind = ax2.imshow(rtop_pdf_norm.T, interpolation='nearest', origin='lower')
plt.colorbar(ind)
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('rtop_pdf')
ind = ax3.imshow(rtop_pdf.T, interpolation='nearest', origin='lower')
plt.colorbar(ind)
plt.savefig('rtop.png')

"""
Save the MSD images in MSD.png in order to compare the contrast. 
"""

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('MSD_norm')
ind = ax1.imshow(MSD_norm.T, interpolation='nearest', origin='lower')
plt.colorbar(ind)
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('MSD')
ind = ax2.imshow(MSD.T, interpolation='nearest', origin='lower')
plt.colorbar(ind)
plt.savefig('MSD.png')
