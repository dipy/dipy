"""
=====================================================================
Auto-correcting a DWI gradient table with the Aganj-2018 criterion
=====================================================================

The b-vectors that ship with a DWI dataset are usually correct, but they are
also one of the most fragile pieces of metadata in the pipeline. Different
scanners, file converters and pre-processing tools disagree on which axis is
which and what sign each axis should carry; the wrong convention silently
corrupts every downstream result -- tractography, microstructure, registration
-- in a way that is hard to spot by eye.

DIPY ships a tractography-free verifier based on the **fiber-continuity error**
of :footcite:p:`Aganj2018`. The intuition: in fibrous tissue the orientation
distribution function (ODF) varies smoothly *along* fibers and sharply *across*
them, so at every voxel the spatial gradient of the ODF should be perpendicular
to the local fiber orientation. Concretely, the criterion is

.. math::

   \\varepsilon(T) = \\int_{\\Omega} \\int_{S^2}
       \\bigl( T(\\hat n) \\cdot \\nabla_x \\psi(x, \\hat n) \\bigr)^2
       \\, d\\hat n \\, dx,

where :math:`T` is a candidate axis permutation/flip applied to the gradient
table, :math:`\\psi` is the CSA-ODF reconstructed from the (possibly wrong)
gradient table, and :math:`\\Omega` is a fibrous-tissue mask. The minimizing
:math:`T^*` is the transform that brings the gradient frame into agreement
with the voxel frame; applying it to the b-vectors yields a corrected table.

There are 24 candidates in total (six axis permutations times four canonical
flips -- flipping all three axes is equivalent to no flip thanks to the
antipodal symmetry :math:`\\psi(\\hat n) = \\psi(-\\hat n)`), and the verifier
scores all of them with a single CSA-ODF reconstruction.
"""

import numpy as np

from dipy.core.gradient_check import (
    apply_transform,
    check_gradient_table,
    correct_bvecs,
    correct_gradient_table,
    transform_label,
)
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti

###############################################################################
# Load the Stanford HARDI dataset
# --------------------------------
# DIPY ships the Stanford HARDI dataset, which has 160 diffusion-weighted
# volumes plus several b0 volumes acquired at 2 mm isotropic resolution. We
# load the DWI volume and the bval/bvec files separately so we keep the raw
# b-vectors around for the corruption demo below.

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
data, _ = load_nifti(hardi_fname)
data = data.astype(np.float32)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

print(f"Data shape : {data.shape}  (X, Y, Z, volumes)")
print(f"b0 volumes : {gtab.b0s_mask.sum()} out of {data.shape[-1]}")

###############################################################################
# Verify the original gradient table
# -----------------------------------
# ``check_gradient_table`` returns the recommended ``(perm, flip)`` transform.
# When the b-vectors are already aligned with the image frame it should hand
# back the identity ``((0, 1, 2), (1, 1, 1))``.
#
# Two optional masks tune cost and quality:
#
# * ``brain_mask=`` restricts the CSA-ODF fit, which is the dominant cost on
#   full-volume HARDI. Pass a brain mask (e.g. from
#   :func:`~dipy.segment.mask.median_otsu`) for a large speed-up.
# * ``mask=`` is the fibrous-tissue mask used for scoring. When omitted the
#   verifier builds its own ADC/GFA-based white-matter approximation inside
#   ``brain_mask`` (or runs ``median_otsu`` itself if neither is supplied).

from dipy.segment.mask import median_otsu

b0_idx = np.where(gtab.b0s_mask)[0].tolist()
_, brain = median_otsu(data, vol_idx=b0_idx, median_radius=4, numpass=4)

perm, flip = check_gradient_table(data, gtab, brain_mask=brain)
print(f"\nOriginal gradient table -> {transform_label(perm, flip)}")
assert (perm, flip) == ((0, 1, 2), (1, 1, 1))

###############################################################################
# Inject a corruption and recover it
# -----------------------------------
# A common real-world failure mode is the "Aarhus-style" corruption: the
# converter swaps the x and y axes and flips the y sign. We apply that to the
# bvecs, rebuild the gradient table, and ask the verifier to recover it.

corrupt_perm = (1, 0, 2)
corrupt_flip = (1, -1, 1)
bvecs_corrupt = apply_transform(gtab.bvecs, corrupt_perm, corrupt_flip)
gtab_corrupt = gradient_table(
    gtab.bvals, bvecs=bvecs_corrupt, b0_threshold=gtab.b0_threshold
)
perm_rec, flip_rec = check_gradient_table(data, gtab_corrupt, brain_mask=brain)
print(f"Corrupted gradient table -> {transform_label(perm_rec, flip_rec)}")

###############################################################################
# The recovered transform composed with the corruption returns to the original
# canonical frame. Because of the ODF's antipodal symmetry the verifier may
# legitimately pick either the corruption-inverse or its global negation; both
# describe the same physical correction.

recovered_bvecs = correct_bvecs(bvecs_corrupt, perm_rec, flip_rec)
diff_pos = np.max(np.abs(recovered_bvecs - gtab.bvecs))
diff_neg = np.max(np.abs(recovered_bvecs + gtab.bvecs))
print(f"max|recovered - original| = {diff_pos:.3e}")
print(f"max|recovered + original| = {diff_neg:.3e}")
print(f"sign-agnostic error       = {min(diff_pos, diff_neg):.3e}")

###############################################################################
# Apply the correction in one shot
# ---------------------------------
# :func:`~dipy.core.gradient_check.correct_gradient_table` is a convenience
# wrapper that runs the verifier and returns a fresh ``GradientTable`` whose
# bvecs have been transformed. When the result is the identity the input
# ``gtab`` is returned unchanged so you do not get a confusing "rebuilt but
# identical" object.

gtab_fixed, perm, flip = correct_gradient_table(data, gtab_corrupt, brain_mask=brain)
print(f"\ncorrect_gradient_table applied {transform_label(perm, flip)}")
print(f"bvals unchanged?  {np.allclose(gtab_fixed.bvals, gtab_corrupt.bvals)}")

###############################################################################
# Command-line equivalent
# ------------------------
# The ``dipy_correct_bvecs`` CLI runs the same algorithm without writing any
# Python. The fastest invocation passes a precomputed brain mask so the
# CSA-ODF fit is limited to brain voxels::
#
#     dipy_correct_bvecs path/to/dwi.nii.gz path/to/dwi.bval path/to/dwi.bvec \\
#         --brain_mask_files path/to/brain_mask.nii.gz \\
#         --out_dir out/ --out_bvecs dwi_corrected.bvec
#
# Without ``--brain_mask_files`` the workflow fits CSA over the entire volume
# and runs an internal median-otsu brain extraction itself, which is several
# times slower on full-brain HARDI. The recommended transform is logged at
# INFO level. If your dataset already has aligned bvecs the workflow logs
# that and writes them through unchanged.
#
# Summary
# -------
#
# * The verifier scores 24 candidate axis-permutation/flip configurations of
#   the gradient table against the CSA-ODF and returns the one that best
#   satisfies the fiber-continuity criterion :footcite:p:`Aganj2018`.
# * A *clean* gradient table yields the identity transform; a corrupted one
#   yields the (inverse) corruption -- modulo a global sign flip that is a
#   no-op under the antipodal symmetry of the ODF.
# * Use :func:`~dipy.core.gradient_check.correct_gradient_table` for a
#   one-line fix, or the ``dipy_correct_bvecs`` workflow on the command line.
# * The default mask works on typical full-brain DWI; for unusual acquisitions
#   (very small volumes, mostly-CSF crops) pass an explicit ``mask=`` derived
#   from your own segmentation.
#
# References
# ----------
#
# .. footbibliography::
#
