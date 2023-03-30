import numpy as np
import dipy.reconst.fwdti as fwdti
import matplotlib.pyplot as plt
from dipy.data import fetch_cenir_multib
from dipy.data import read_cenir_multib
from dipy.segment.mask import median_otsu

fetch_cenir_multib(with_raw=False)

# single-shell data
bvals = [1000]
img, gtab_single = read_cenir_multib(bvals)
data_single = np.asarray(img.dataobj)

# multi-shell data
bvals = [1000, 2000]
img, gtab_multi = read_cenir_multib(bvals)
data_multi = np.asarray(img.dataobj)

# masking
maskdata_s, mask = median_otsu(data_single, vol_idx=[0, 1],
                               autocrop=False)
maskdata_m, mask = median_otsu(data_multi, vol_idx=[0, 1],
                               autocrop=False)

axial_slice = 40

mask_roi = np.zeros(maskdata_m.shape[:-1], dtype=bool)
mask_roi[:, :, axial_slice] = mask[:, :, axial_slice]

S0 = np.mean(data_multi[..., gtab_multi.b0s_mask], axis=-1) * mask_roi

# checking S0 to choose St and Sw for FERNET method, sometimes choosing the
# right values is hard, is there a better way?
plt.figure()
plt.subplot(121)
plt.imshow(S0[:, :, axial_slice].T,
           origin='lower',
           cmap='gray',
           vmin=0,
           vmax=15000)

plt.subplot(122)
plt.hist(S0[mask_roi], bins=50)

# FW-DTI
fwdtimodel = fwdti.FreeWaterTensorModel(gtab_single, St=1000, Sw=15000,
                                        fit_method='hy')

fwdtifit = fwdtimodel.fit(data_single, mask=mask_roi)

FA = fwdtifit.fa
MD = fwdtifit.md
F = fwdtifit.f

# multi-shell FW-DTI
dtimodel = fwdti.FreeWaterTensorModel(gtab_multi)
dtifit = dtimodel.fit(maskdata_m, mask=mask_roi)
dti_FA = dtifit.fa
dti_MD = dtifit.md
dti_F = dtifit.f

# correcting bad voxels
FA[F > 0.7] = 0
dti_FA[dti_F > 0.7] = 0
MD[F > 0.7] = 0
dti_MD[dti_F > 0.7] = 0

fig1, ax = plt.subplots(2, 4, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.05)
ax.flat[0].imshow(FA[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[0].set_title('A) single-shell fwDTI FA')
ax.flat[1].imshow(dti_FA[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[1].set_title('B) multi-shell fwDTI FA')

FAdiff = abs(FA[:, :, axial_slice] - dti_FA[:, :, axial_slice])
ax.flat[2].imshow(FAdiff.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax.flat[2].set_title('C) FA difference')

Fdiff = abs(F[:, :, axial_slice] - dti_F[:, :, axial_slice])
ax.flat[3].imshow(Fdiff.T, origin='lower', cmap='gray', vmin=0, vmax=1)
ax.flat[3].set_title('H) FW difference')

ax.flat[4].imshow(MD[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[4].set_title('D) single-shell fwDTI MD')
ax.flat[5].imshow(dti_MD[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[5].set_title('E) multi-shell fwDTI MD')

MDdiff = abs(MD[:, :, axial_slice] - dti_MD[:, :, axial_slice])
ax.flat[6].imshow(MDdiff.T, origin='lower', cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[6].set_title('F) MD difference')

ax.flat[7].imshow(F[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[7].set_title('G) single-shell free water volume')

plt.show()
# fig1.savefig('In_vivo_free_water_DTI_single_shell.png')
