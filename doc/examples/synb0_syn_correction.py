"""
=========================================================
Susceptibility Distortion Correction with Synb0-SyN
=========================================================

Diffusion MRI is acquired with echo-planar imaging (EPI), which is fast but
sensitive to off-resonance fields caused by the difference in magnetic
susceptibility between tissue and air. These fields shift the signal along
the phase-encoding (PE) direction, so the images are stretched or compressed
near the sinuses and ear canals and no longer line up with the anatomy.

The usual remedy is to acquire a fieldmap, or a second b0 with the opposite PE
polarity (as used by TOPUP). Many datasets have neither. Synb0-SyN
:footcite:p:`Chigurupati2024` addresses this case with only a T1-weighted
image of the same subject:

1. The T1-weighted image and the distorted b0 are brought to the MNI space
   with an affine transform.
2. The Synb0 network :footcite:p:`Schilling2019`, :footcite:p:`Schilling2020`
   synthesizes from them an *undistorted* b0, with the contrast of a b0 and
   the geometry of the T1-weighted image.
3. The synthetic b0 is brought back to the DWI space and the distorted b0 is
   registered to it with symmetric diffeomorphic registration (SyN)
   :footcite:p:`Avants2008`. Only the displacement along the PE axis is kept,
   since susceptibility distortions only occur in that direction.
4. The displacement is applied to every DWI volume.

Synb0 requires PyTorch (or TensorFlow); the network weights are downloaded the
first time it is used.
"""

import matplotlib.pyplot as plt
import numpy as np

from dipy.align.disco import synb0_syn
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti

###############################################################################
# We use the Stanford HARDI dataset, which comes with a T1-weighted image of
# the same subject and tissue probability maps from which we build a brain
# mask for the T1. The DWI was acquired with axial slices and phase encoding
# along the anterior-posterior axis, which is voxel axis 1.

dwi_fname, bval_fname, bvec_fname = get_fnames(name="stanford_hardi")
t1_fname = get_fnames(name="stanford_t1")
csf_fname, gm_fname, wm_fname = get_fnames(name="stanford_pve_maps")

dwi, dwi_affine = load_nifti(dwi_fname)
t1, t1_affine = load_nifti(t1_fname)
pve = [load_nifti(f)[0] for f in (csf_fname, gm_fname, wm_fname)]
t1_mask = np.sum(pve, axis=0) > 0.5

bvals = np.loadtxt(bval_fname)
b0_index = int(np.flatnonzero(bvals < 50)[0])
print(f"DWI shape: {dwi.shape}, b0 volume index: {b0_index}")

###############################################################################
# ``synb0_syn`` estimates the distortion from the b0 volume ``b0_index`` and
# corrects all volumes. ``pe_axis`` is the voxel axis of the DWI along which
# phase encoding was performed; the returned ``field`` is the displacement, in
# mm, along that axis.

corrected, field = synb0_syn(
    dwi,
    t1,
    dwi_affine,
    t1_affine,
    b0_index=b0_index,
    T1_mask=t1_mask,
    pe_axis=1,
    return_field=True,
)

print(f"Displacement range (mm): {field.min():.2f} to {field.max():.2f}")

###############################################################################
# To judge the correction, we overlay the outline of the brain in the
# T1-weighted image (which is free of distortion) on the b0 before and after
# correction. The frontal lobe, above the sinuses, is where the distortion is
# largest.

b0 = dwi[..., b0_index]
b0_corrected = corrected[..., b0_index]
sl = 35
edge = t1_mask[:, :, sl].T


def show(ax, img, title):
    vmax = np.percentile(img, 99)
    ax.imshow(img[:, :, sl].T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
    ax.contour(edge, levels=[0.5], colors="r", linewidths=0.8)
    ax.set_title(title)
    ax.set_axis_off()


fig, axes = plt.subplots(1, 3, figsize=(12, 5))
show(axes[0], b0, "b0 before correction")
show(axes[1], b0_corrected, "b0 after Synb0-SyN")
lim = np.abs(field).max()
im = axes[2].imshow(
    field[:, :, sl].T, cmap="RdBu_r", origin="lower", vmin=-lim, vmax=lim
)
axes[2].set_title("Displacement along PE (mm)")
axes[2].set_axis_off()
fig.colorbar(im, ax=axes[2], fraction=0.046)
fig.tight_layout()
fig.savefig("synb0_syn_correction.png", bbox_inches="tight")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Axial slice of the b0 before (left) and after (middle) correction, with the
# brain outline from the T1-weighted image in red, and the estimated
# displacement along the phase-encoding axis (right).
#
#
# The corrected data and the displacement can be saved for later use:

save_nifti("dwi_synb0_syn.nii.gz", corrected.astype(np.float32), dwi_affine)
save_nifti("synb0_syn_field.nii.gz", field.astype(np.float32), dwi_affine)

###############################################################################
# The same correction is available from the command line::
#
#     dipy_correct_synb0_syn dwi.nii.gz t1.nii.gz --pe_axis 1 --out_dir out
#
# If the T1-weighted image is not skull-stripped, a brain mask can be given
# with ``--t1_mask_file``, or computed with SynthSeg by adding
# ``--apply_skull_strip``.
#
#
# References
# ----------
#
# .. footbibliography::
#
