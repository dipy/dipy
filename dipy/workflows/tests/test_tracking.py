import numpy as np
from numpy.testing import assert_equal
from dipy.testing import assert_false
from os.path import join

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

from dipy.data import (fetch_stanford_hardi, fetch_stanford_pve_maps,
                       fetch_stanford_labels, get_fnames)
from dipy.io.image import save_nifti
from dipy.workflows.mask import MaskFlow
from dipy.workflows.reconst import ReconstCSDFlow
from dipy.workflows.tracking import (LocalFiberTrackingPAMFlow,
                                     PFTrackingPAMFlow)
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter,
                            ClosestPeakDirectionGetter)


def test_particule_filtering_traking_workflows():
    with TemporaryDirectory() as out_dir:
        dwi_path, bval_path, bvec_path = get_fnames('small_64D')
        vol_img = nib.load(dwi_path)
        volume = vol_img.get_data()

        # Create some mask
        mask = np.ones_like(volume[:, :, :, 0])
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), vol_img.affine)
        mask_path = join(out_dir, 'tmp_mask.nii.gz')
        nib.save(mask_img, mask_path)

        simple_wm = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                              [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                              [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
                              [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              ])
        simple_wm = np.dstack([np.zeros(simple_wm.shape),
                               np.zeros(simple_wm.shape),
                               simple_wm, simple_wm, simple_wm,
                               simple_wm, simple_wm, simple_wm,
                               np.zeros(simple_wm.shape),
                               np.zeros(simple_wm.shape)])
        simple_gm = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                              [0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                              [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                              ])
        simple_gm = np.dstack([np.zeros(simple_gm.shape),
                               np.zeros(simple_gm.shape),
                               simple_gm, simple_gm, simple_gm,
                               simple_gm, simple_gm, simple_gm,
                               np.zeros(simple_gm.shape),
                               np.zeros(simple_gm.shape)])
        simple_csf = np.ones(simple_wm.shape) - simple_wm - simple_gm

        wm_path = join(out_dir, 'tmp_wm.nii.gz')
        gm_path = join(out_dir, 'tmp_gm.nii.gz')
        csf_path = join(out_dir, 'tmp_csf.nii.gz')

        for path, arr in zip([wm_path, gm_path, csf_path],
                             [simple_wm, simple_gm, simple_csf]):
            nib.save(nib.Nifti1Image(arr.astype(np.uint8), vol_img.affine),
                     path)

        # CSD Reconstruction
        reconst_csd_flow = ReconstCSDFlow()
        reconst_csd_flow.run(dwi_path, bval_path, bvec_path, mask_path,
                             out_dir=out_dir, extract_pam_values=True)

        pam_path = reconst_csd_flow.last_generated_outputs['out_pam']
        gfa_path = reconst_csd_flow.last_generated_outputs['out_gfa']

        # Create seeding mask by thresholding the gfa
        mask_flow = MaskFlow()
        mask_flow.run(gfa_path, 0.8, out_dir=out_dir)
        seeds_path = mask_flow.last_generated_outputs['out_mask']

        # Test tracking
        pf_track_pam = PFTrackingPAMFlow()
        assert_equal(pf_track_pam.get_short_name(), 'track_pft')
        pf_track_pam.run(pam_path, wm_path, gm_path, csf_path, seeds_path)
        tractogram_path = \
            pf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))


def test_local_fiber_tracking_workflow():
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
        lf_track_pam._force_overwrite = True
        assert_equal(lf_track_pam.get_short_name(), 'track_local')
        lf_track_pam.run(pam_path, gfa_path, seeds_path)
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))

        # Test tracking with pam with sh
        lf_track_pam = LocalFiberTrackingPAMFlow()
        lf_track_pam._force_overwrite = True
        lf_track_pam.run(pam_path, gfa_path, seeds_path,
                         tracking_method="eudx")
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))

        # Test tracking with pam with sh and deterministic getter
        lf_track_pam = LocalFiberTrackingPAMFlow()
        lf_track_pam._force_overwrite = True
        lf_track_pam.run(pam_path, gfa_path, seeds_path,
                         tracking_method="deterministic")
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))

        # Test tracking with pam with sh and probabilistic getter
        lf_track_pam = LocalFiberTrackingPAMFlow()
        lf_track_pam._force_overwrite = True
        lf_track_pam.run(pam_path, gfa_path, seeds_path,
                         tracking_method="probabilistic")
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))

        # Test tracking with pam with sh and closestpeaks getter
        lf_track_pam = LocalFiberTrackingPAMFlow()
        lf_track_pam._force_overwrite = True
        lf_track_pam.run(pam_path, gfa_path, seeds_path,
                         tracking_method="closestpeaks")
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))


def is_tractogram_empty(tractogram_path):
    tractogram_file = \
        nib.streamlines.load(tractogram_path)

    return len(tractogram_file.tractogram) == 0


if __name__ == '__main__':
    test_local_fiber_tracking_workflow()
    # test_particule_filtering_traking_workflows()
