"""
==================================================================
Fieldmap-less Susceptibility Distortion Correction with SyN
==================================================================

Diffusion MRI is acquired with echo-planar imaging (EPI), which is fast but
sensitive to off-resonance fields caused by the difference in magnetic
susceptibility between tissue and air. These fields displace the signal along
the phase-encoding (PE) direction, so the images are stretched or compressed
near the sinuses and ear canals and no longer line up with the anatomy.

When neither a fieldmap nor a b0 with the opposite PE polarity was acquired,
the distortion can still be estimated from an undistorted anatomical image of
the same subject :footcite:p:`Huang2008`, :footcite:p:`Wang2017`. In DIPY,
:func:`~dipy.align.sdc.syn_sdc`:

1. rigidly aligns the anatomical image to the distorted b0;
2. maps the anatomical image to the b0 contrast: a T1-weighted (T1w) image is
   intensity-inverted, so that cerebrospinal fluid is bright and white matter
   dark as in a b0, while the intensities of a T2-weighted (T2w) image, whose
   contrast already resembles a b0, are mapped to the b0 through their joint
   histogram inside the brain;
3. registers the b0 to this b0-like reference with symmetric diffeomorphic
   registration (SyN) :footcite:p:`Avants2008`, with every update restricted
   to the PE axis, since susceptibility distortions only occur in that
   direction;
4. repeats the rigid alignment and the registration on the corrected b0, so
   that the distortion does not bias the coregistration;
5. applies the estimated displacement to every DWI volume.
"""

import matplotlib.pyplot as plt
import numpy as np

from dipy.align.sdc import syn_sdc
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti

###############################################################################
# We use the Stanford HARDI dataset, which comes with a T1w image of the same
# subject and tissue probability maps from which we build a brain mask for the
# T1w image. The DWI was acquired with axial slices and phase encoding along
# the anterior-posterior axis, which is voxel axis 1.

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
# ``syn_sdc`` estimates the distortion from the b0 volume ``b0_index`` and
# corrects all volumes. ``pe_axis`` is the voxel axis of the DWI along which
# phase encoding was performed (its polarity does not matter) and
# ``anat_contrast`` the contrast of the anatomical image. The anatomical image
# should be skull-stripped, or a brain mask given in ``anat_mask``. The
# returned ``field`` is the displacement, in voxels, along the PE axis.

corrected, field = syn_sdc(
    dwi,
    dwi_affine,
    t1,
    t1_affine,
    pe_axis=1,
    anat_contrast="t1",
    anat_mask=t1_mask,
    b0_index=b0_index,
    return_field=True,
)

print(f"Displacement range (voxels): {field.min():.2f} to {field.max():.2f}")

###############################################################################
# To judge the correction, we overlay the outline of the brain in the T1w
# image, which is free of distortion, on the b0 before and after correction.
# The distortion is largest in the frontal lobe, above the sinuses.

b0 = dwi[..., b0_index]
b0_corrected = corrected[..., b0_index]
sl = 40
edge = np.rot90(t1_mask[sl])


def show(ax, img, title):
    vmax = np.percentile(img, 99)
    ax.imshow(np.rot90(img[sl]), cmap="gray", vmin=0, vmax=vmax)
    ax.contour(edge, levels=[0.5], colors="r", linewidths=0.8)
    ax.set_title(title)
    ax.set_axis_off()


fig, axes = plt.subplots(1, 3, figsize=(13, 4))
show(axes[0], b0, "b0 before correction")
show(axes[1], b0_corrected, "b0 after SyN-SDC")
lim = np.abs(field).max()
im = axes[2].imshow(np.rot90(field[sl]), cmap="RdBu_r", vmin=-lim, vmax=lim)
axes[2].set_title("Displacement along PE (voxels)")
axes[2].set_axis_off()
fig.colorbar(im, ax=axes[2], fraction=0.046)
fig.tight_layout()
fig.savefig("syn_sdc_correction.png", bbox_inches="tight")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Sagittal slice of the b0 before (left) and after (middle) correction, with
# the brain outline from the T1w image in red, and the estimated displacement
# along the phase-encoding axis (right).
#
#
# Where the signal is lost altogether (here, the dark region above the
# sphenoid sinus at the bottom right), the b0 contains nothing that can be
# matched to the anatomy, so the displacement estimated there is unreliable;
# registration-based correction cannot recover lost signal.
#
# With a T2w image the call is the same, with ``anat_contrast="t2"``. The
# corrected data and the displacement can be saved for later use:

save_nifti("dwi_syn_sdc.nii.gz", corrected.astype(np.float32), dwi_affine)
save_nifti("syn_sdc_field.nii.gz", field, dwi_affine)

###############################################################################
# The same correction is available from the command line::
#
#     dipy_correct_syn_sdc dwi.nii.gz t1w_brain.nii.gz --pe_axis 1 \
#         --anat_contrast t1 --out_dir out
#
# References
# ----------
#
# .. footbibliography::
#
