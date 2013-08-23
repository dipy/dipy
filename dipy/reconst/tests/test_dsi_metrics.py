import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi, get_sphere
from dipy.align.aniso2iso import resample
from dipy.viz import fvtk
from dipy.core.ndindex import ndindex
from dipy.data import get_sphere
import dipy.reconst.dti as dti
from dipy.data import get_data, dsi_voxels
from dipy.reconst.dsi import (DiffusionSpectrumModel,DiffusionSpectrumDeconvModel)
from dipy.core.sphere import Sphere
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs

fetch_taiwan_ntu_dsi()
img, gtab = read_taiwan_ntu_dsi()
data = img.get_data()
affine = img.get_affine()

print('data.shape (%d, %d, %d, %d)' % data.shape)

dsmodel = DiffusionSpectrumModel(gtab, qgrid_size=35, r_start=0.4*17, r_end=0.7*17, filter_width=18.5)

dataslice = data[30:70, 20:80, data.shape[2] / 2]
mask = dataslice[..., 0] > 50

# for the signal the normalization means that is divided by the b0
print('Calculating... rtz_signal_norm') 
rtz_signal_norm=dsmodel.fit(dataslice).rt0p_signal()
print('Calculating... rtz_signal')
rtz_signal=dsmodel.fit(dataslice).rt0p_signal(normalized=False)

# for the pdf the normalization means that is divided by pdf.sum()
print('Calculating... rtz_pdf_norm')
rtz_pdf_norm=dsmodel.fit(dataslice).rt0p_pdf()
print('Calculating... rtz_pdf')
rtz_pdf=dsmodel.fit(dataslice).rt0p_pdf(normalized=False)

# for the pdf the normalization means that is divided by pdf.sum()
print('Calculating... MSD_norm')
MSD_norm=dsmodel.fit(dataslice).MSD_discrete()
print('Calculating... MSD')
MSD=dsmodel.fit(dataslice).MSD_discrete(normalized=False)

# plot rt0p
fig=plt.figure()
ax1=fig.add_subplot(2,2,1)
ax1.set_title('rtz_signal_norm')
ind=ax1.imshow(rtz_signal_norm.T,interpolation='nearest',origin='lower')
plt.colorbar(ind)
ax2=fig.add_subplot(2,2,2)
ax2.set_title('rtz_signal')
ind=ax2.imshow(rtz_signal.T,interpolation='nearest',origin='lower')
plt.colorbar(ind)
ax3=fig.add_subplot(2,2,3)
ax3.set_title('rtz_pdf_norm')
ind=ax3.imshow(rtz_pdf_norm.T,interpolation='nearest',origin='lower')
plt.colorbar(ind)
ax4=fig.add_subplot(2,2,4)
ax4.set_title('rtz_pdf')
ind=ax4.imshow(rtz_pdf.T,interpolation='nearest',origin='lower')
plt.colorbar(ind)
plt.show()

# plot MSD
fig=plt.figure()
ax1=fig.add_subplot(2,1,1)
ax1.set_title('MSD_norm')
ind=ax1.imshow(MSD_norm.T,interpolation='nearest',origin='lower')
plt.colorbar(ind)
ax2=fig.add_subplot(2,1,2)
ax2.set_title('MSD')
ind=ax2.imshow(MSD.T,interpolation='nearest',origin='lower')
plt.colorbar(ind)
plt.show()