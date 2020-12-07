import numpy as np
import fwdti_custom as fwdti
import dipy.reconst.dti as dti
import matplotlib.pyplot as plt
from dipy.data import fetch_cenir_multib
from dipy.data import read_cenir_multib
from dipy.segment.mask import median_otsu

fetch_cenir_multib(with_raw=False)

# single-shell data
bvals = [1000]

img, gtab = read_cenir_multib(bvals)

data = np.asarray(img.dataobj)

affine = img.affine

# masking
maskdata, mask = median_otsu(data, vol_idx=[0, 1], median_radius=4, numpass=2,
                             autocrop=False, dilate=1)

axial_slice = 40

mask_roi = np.zeros(data.shape[:-1], dtype=bool)
mask_roi[:, :, axial_slice] = mask[:, :, axial_slice]

S0 = np.mean(data[..., gtab.b0s_mask], axis=-1)

# checking S0 to choose St and Sw for FERNET method
plt.figure()
plt.imshow(S0[:, :, axial_slice].T, origin='lower', cmap='gray', vmin=0, vmax=15000)
plt.show()

# Running routine
# FW-DTI
fwdtimodel = fwdti.FreeWaterTensorModel(gtab, St=5000, Sw=14000, method='hy')
fwdtifit = fwdtimodel.fit(data, mask=mask_roi)
FA = fwdtifit.fa
MD = fwdtifit.md
F = fwdtifit.f

# DTI
dtimodel = dti.TensorModel(gtab)
dtifit = dtimodel.fit(data, mask=mask_roi)
dti_FA = dtifit.fa
dti_MD = dtifit.md

# correcting bad voxels
FA[F > 0.7] = 0
dti_FA[F > 0.7] = 0
MD[F > 0.7] = 0
dti_MD[F > 0.7] = 0

fig1, ax = plt.subplots(2, 4, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.05)
ax.flat[0].imshow(FA[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[0].set_title('A) fwDTI FA')
ax.flat[1].imshow(dti_FA[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[1].set_title('B) standard DTI FA')

FAdiff = abs(FA[:, :, axial_slice] - dti_FA[:, :, axial_slice])
ax.flat[2].imshow(FAdiff.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax.flat[2].set_title('C) FA difference')

ax.flat[3].axis('off')

ax.flat[4].imshow(MD[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[4].set_title('D) fwDTI MD')
ax.flat[5].imshow(dti_MD[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[5].set_title('E) standard DTI MD')

MDdiff = abs(MD[:, :, axial_slice] - dti_MD[:, :, axial_slice])
ax.flat[6].imshow(MDdiff.T, origin='lower', cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[6].set_title('F) MD difference')

ax.flat[7].imshow(F[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[7].set_title('G) free water volume')

plt.show()
fig1.savefig('In_vivo_free_water_DTI_single_shell.png')
