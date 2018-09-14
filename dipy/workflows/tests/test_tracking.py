import numpy as np
from numpy.testing import assert_equal
from dipy.testing import assert_false
from os.path import join

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

from dipy.data import get_fnames
from dipy.io.image import save_nifti
from dipy.workflows.mask import MaskFlow
from dipy.workflows.reconst import ReconstCSDFlow
from dipy.workflows.tracking import (LocalFiberTrackingPAMFlow,
                                     PFTrackingPAMFlow)


def test_local_fiber_track():
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames('small_64D')
        vol_img = nib.load(data_path)
        volume = vol_img.get_data()
        mask = np.ones_like(volume[:, :, :, 0])
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), vol_img.affine)
        mask_path = join(out_dir, 'tmp_mask.nii.gz')
        nib.save(mask_img, mask_path)

        reconst_csd_flow = ReconstCSDFlow()
        reconst_csd_flow.run(data_path, bval_path, bvec_path, mask_path,
                             out_dir=out_dir, extract_pam_values=True)

        pam_path = reconst_csd_flow.last_generated_outputs['out_pam']
        gfa_path = reconst_csd_flow.last_generated_outputs['out_gfa']

        # Create seeding mask by thresholding the gfa
        mask_flow = MaskFlow()
        mask_flow.run(gfa_path, 0.8, out_dir=out_dir)
        seeds_path = mask_flow.last_generated_outputs['out_mask']

        # Put identity in gfa path to prevent impossible to use
        # local tracking because of affine containing shearing.
        gfa_img = nib.load(gfa_path)
        save_nifti(gfa_path, gfa_img.get_data(), np.eye(4), gfa_img.header)

        # Test tracking with pam no sh
        lf_track_pam = LocalFiberTrackingPAMFlow()
        assert_equal(lf_track_pam.get_short_name(), 'lf_track')
        lf_track_pam.run(pam_path, gfa_path, seeds_path)
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))

        # Test tracking with pam with sh
        lf_track_pam.run(pam_path, gfa_path, seeds_path, use_sh=True)
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))

        # Test tracking with pam with sh and deterministic getter
        lf_track_pam.run(pam_path, gfa_path, seeds_path, use_sh=True,
                         sh_strategy="deterministic")
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))

        # Test tracking with pam with sh and probabilistic getter
        lf_track_pam.run(pam_path, gfa_path, seeds_path, use_sh=True,
                         sh_strategy="probabilistic")
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))

        # Test tracking with pam with sh and closestpeaks getter
        lf_track_pam.run(pam_path, gfa_path, seeds_path, use_sh=True,
                         sh_strategy="closestpeaks")
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))


def is_tractogram_empty(tractogram_path):
    tractogram_file = \
        nib.streamlines.load(tractogram_path)

    return len(tractogram_file.tractogram) == 0


if __name__ == '__main__':
    test_local_fiber_track()
