"""

=================================================
Between-volumes Motion Correction on DWI datasets
=================================================

During a dMRI acquisition, the subject motion inevitable. This motion implies
a misalignment between N volumes on a dMRI dataset. A common way to solve this
issue is to perform a registration on each acquired volume to a
reference b = 0. [JenkinsonSmith01]_

This preprocessing is an highly recommended step that should be executed before
any dMRI dataset analysis.


Let's import some essential functions.
"""

from dipy.align import motion_correction
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs

###############################################################################
# We choose one of the data from the datasets in dipy_. However, you can
# replace the following line with the path of your image.

dwi_fname, dwi_bval_fname, dwi_bvec_fname = get_fnames('sherbrooke_3shell')

###############################################################################
# We load the image and the affine of the image. The affine is the
# transformation matrix which maps image coordinates to world (mm) coordinates.
# We also load the b-values and b-vectors.

data, affine = load_nifti(dwi_fname)
bvals, bvecs = read_bvals_bvecs(dwi_bval_fname, dwi_bvec_fname)

###############################################################################
# This data has 193 volumes. For this demo purpose, we decide to reduce the
# number of volumes to 3. However, we do not recommended to perform a motion
# correction with less than 10 volumes.

data_small = data[..., :3]
bvals_small = bvals[:3]
bvecs_small = bvecs[:3]
gtab = gradient_table(bvals_small, bvecs_small)

###############################################################################
# Start motion correction of our reduced DWI dataset(between-volumes motion
# correction).

data_corrected, reg_affines = motion_correction(data_small, gtab, affine)

###############################################################################
# Save our DWI dataset corrected to a new Nifti file.

save_nifti('motion_correction.nii.gz', data_corrected.get_fdata(),
           data_corrected.affine)

###############################################################################
# References
# ----------
#
# .. [JenkinsonSmith01] Jenkinson, M., Smith, S., 2001. A global optimisation
#    method for robust affine registration of brain images. Med Image Anal 5
#    (2), 143–56.
#
# .. include:: ../links_names.inc
