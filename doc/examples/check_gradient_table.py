"""
=====================================================================
Correcting DWI gradient tables: Aganj 2018 and Schilling 2019
=====================================================================

B-vectors are one of the most fragile pieces of DWI metadata.  Different
scanners, DICOM converters and preprocessing tools disagree on axis ordering
and sign conventions, silently corrupting every downstream result —
tractography, microstructure modelling, registration — in ways that are hard
to spot by eye.

DIPY provides two independent methods to detect and correct these errors:

**Method 1 — Aganj 2018 fiber-continuity criterion** (:footcite:p:`Aganj2018`)
  Requires only the raw DWI volume (no prior fitting).  Reconstructs
  orientation distribution functions (ODFs) with a CSA model and scores all
  24 axis-permutation/flip candidates; applies the one that *minimises* the
  fiber-continuity error:

  .. math::

     \\varepsilon(T) = \\int_{\\Omega} \\int_{S^2}
         \\bigl( T(\\hat n) \\cdot \\nabla_x \\psi(x, \\hat n) \\bigr)^2
         \\, d\\hat n \\, dx

**Method 2 — Schilling 2019 fiber-coherence index** (:footcite:p:`Schilling2019b`)
  Requires pre-computed peak directions and an FA map.  Scores all 24
  candidates by how well fiber directions agree with the inter-voxel
  displacement vector between neighbouring voxels; applies the one that
  *maximises* the coherence index.  Substantially faster than method 1 if
  peaks and FA are already available.

When to use which
-----------------

.. list-table::
   :header-rows: 1

   * - Criterion
     - Aganj 2018
     - Schilling 2019
   * - Inputs needed
     - raw DWI + bvals + bvecs
     - peaks + FA + mask
   * - Prior fitting required
     - No (fits ODF internally)
     - Yes (DTI or CSD run first)
   * - Speed
     - Slower (ODF reconstruction)
     - Fast when peaks are ready
   * - Python entry point
     - ``dipy.core.gradient_check``
     - ``dipy.reconst.utils``
   * - CLI flag
     - ``--method aganj2018`` (default)
     - ``--method schilling2019``
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
# ============================================================
# Part 1 — Aganj 2018 (fiber-continuity criterion)
# ============================================================
#
# Load the Stanford HARDI dataset
# --------------------------------
# DIPY ships the Stanford HARDI dataset (160 DWI volumes at 2 mm isotropic).
# We keep the raw b-vectors to demonstrate a corruption/recovery cycle.

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
# When b-vectors are already aligned the result is the identity transform
# ``((0, 1, 2), (1, 1, 1))``.
#
# Passing a brain mask to ``brain_mask=`` restricts the CSA-ODF fit to brain
# voxels, which gives a significant speed-up on full-brain acquisitions.

from dipy.segment.mask import median_otsu

b0_idx = np.where(gtab.b0s_mask)[0].tolist()
_, brain = median_otsu(data, vol_idx=b0_idx, median_radius=4, numpass=4)

perm, flip = check_gradient_table(data, gtab, brain_mask=brain)
print(f"\nOriginal gradient table -> {transform_label(perm, flip)}")
assert (perm, flip) == ((0, 1, 2), (1, 1, 1))

###############################################################################
# Inject a corruption and recover it
# -----------------------------------
# A common real-world failure is a converter that swaps x/y and flips y.
# We apply that, then ask the verifier to detect and undo it.

corrupt_perm = (1, 0, 2)
corrupt_flip = (1, -1, 1)
bvecs_corrupt = apply_transform(gtab.bvecs, corrupt_perm, corrupt_flip)
gtab_corrupt = gradient_table(
    gtab.bvals, bvecs=bvecs_corrupt, b0_threshold=gtab.b0_threshold
)
perm_rec, flip_rec = check_gradient_table(data, gtab_corrupt, brain_mask=brain)
print(f"Corrupted table  -> {transform_label(perm_rec, flip_rec)}")

recovered_bvecs = correct_bvecs(bvecs_corrupt, perm_rec, flip_rec)
diff_pos = np.max(np.abs(recovered_bvecs - gtab.bvecs))
diff_neg = np.max(np.abs(recovered_bvecs + gtab.bvecs))
print(f"max|recovered - original| = {diff_pos:.3e}")
print(f"sign-agnostic error       = {min(diff_pos, diff_neg):.3e}")

###############################################################################
# One-shot convenience wrapper
# ----------------------------
# :func:`~dipy.core.gradient_check.correct_gradient_table` runs the verifier
# and returns a corrected ``GradientTable``.  When the result is the identity
# the *same* object is returned (no unnecessary copy).

gtab_fixed, perm, flip = correct_gradient_table(data, gtab_corrupt, brain_mask=brain)
print(f"\ncorrect_gradient_table applied: {transform_label(perm, flip)}")
print(f"bvals unchanged?  {np.allclose(gtab_fixed.bvals, gtab_corrupt.bvals)}")

###############################################################################
# ============================================================
# Part 2 — Schilling 2019 (fiber-coherence index)
# ============================================================
#
# The Schilling method works on pre-computed peaks and FA, so we first fit a
# DTI model to the uncorrupted HARDI data to obtain those maps.

from dipy.reconst import dti
from dipy.reconst.utils import compute_coherence_table_for_gradient_transforms

dti_model = dti.TensorModel(gtab)
dti_fit = dti_model.fit(data, mask=brain)

peaks = dti_fit.evecs[..., 0]  # principal eigenvector
fa = dti_fit.fa

print(f"\nPeaks shape : {peaks.shape}")
print(f"FA    range : {fa[brain].min():.3f} – {fa[brain].max():.3f}")

###############################################################################
# Score all 24 candidates with the coherence index
# -------------------------------------------------
# :func:`~dipy.reconst.utils.compute_coherence_table_for_gradient_transforms`
# evaluates the Schilling 2019 coherence index for every (permutation, flip)
# candidate and returns the list of scores together with the corresponding
# transformation matrices.
#
# We mask peaks and FA to white-matter tissue before scoring so that CSF and
# GM voxels do not dilute the signal.

tissue_mask = (fa > 0.2) & brain
masked_peaks = peaks * tissue_mask[..., np.newaxis]
masked_fa = fa * tissue_mask

coherence_values, transforms = compute_coherence_table_for_gradient_transforms(
    masked_peaks, masked_fa
)

best_idx = int(np.argmax(coherence_values))
best_t = transforms[best_idx]
print(f"\nBest transform (Schilling):\n{best_t}")
print(f"Coherence at best : {coherence_values[best_idx]:.4f}")
print(f"Identity selected : {np.allclose(best_t, np.eye(3))}")

###############################################################################
# Apply the selected transform to the b-vectors
# -----------------------------------------------
# Multiply b-vectors by the transpose of the best transform.  When the
# identity is selected the b-vectors are returned unchanged.

if np.allclose(best_t, np.eye(3)):
    corrected_schilling = gtab.bvecs.copy()
    print("b-vectors already aligned (Schilling 2019).")
else:
    corrected_schilling = np.dot(gtab.bvecs, best_t.T)
    print("b-vectors corrected (Schilling 2019).")

###############################################################################
# ============================================================
# Part 3 — Unified CLI
# ============================================================
#
# A single CLI command, ``dipy_correct_bvecs``, exposes both methods via the
# ``--method`` flag.  The first positional argument is always the b-vector
# file.
#
# **Aganj 2018** (default — needs only raw DWI)::
#
#     dipy_correct_bvecs path/to/dwi.bvec \
#         --method aganj2018 \
#         --dwi_files path/to/dwi.nii.gz \
#         --bvalues_files path/to/dwi.bval \
#         --brain_mask_files path/to/brain_mask.nii.gz \
#         --out_dir out/ --out_bvecs corrected.bvec
#
# **Schilling 2019** (needs pre-computed peaks + FA)::
#
#     dipy_correct_bvecs path/to/dwi.bvec \
#         --method schilling2019 \
#         --fa_files path/to/fa.nii.gz \
#         --peaks_files path/to/peaks.pam5 \
#         --mask_files path/to/mask.nii.gz \
#         --out_dir out/ --out_bvecs corrected.bvec
#
# The workflow logs the detected transform at INFO level.  If the b-vectors
# are already aligned it writes them through unchanged.
#
# ============================================================
# Part 4 — Summary
# ============================================================
#
# * **Aganj 2018**: no prior fitting required; scores all 24 candidates by
#   minimising the fiber-continuity error; use when you have only raw DWI.
# * **Schilling 2019**: faster when peaks and FA are already available; scores
#   all 24 candidates by maximising the fiber-coherence index.
# * Both methods evaluate all 24 distinct (permutation, flip) configurations
#   and select the best one automatically.
# * The unified CLI ``dipy_correct_bvecs`` selects the algorithm via
#   ``--method aganj2018`` (default) or ``--method schilling2019``.
#
# References
# ----------
#
# .. footbibliography::
#
