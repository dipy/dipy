from os.path import join
from tempfile import TemporaryDirectory
import warnings

import nibabel as nib
import numpy as np
from numpy.testing import assert_equal
from dipy.testing import assert_false, assert_true


from dipy.data import get_fnames
from dipy.io.image import save_nifti, load_nifti
from dipy.io.streamline import load_tractogram
from dipy.reconst.shm import descoteaux07_legacy_msg
from dipy.workflows.mask import MaskFlow
from dipy.workflows.reconst import ReconstCSDFlow
from dipy.workflows.tracking import (LocalFiberTrackingPAMFlow,
                                     PFTrackingPAMFlow)


def test_particle_filtering_tracking_workflows():
    with TemporaryDirectory() as out_dir:
        dwi_path, bval_path, bvec_path = get_fnames('small_64D')
        volume, affine = load_nifti(dwi_path)

        # Create some mask
        mask = np.ones_like(volume[:, :, :, 0], dtype=np.uint8)
        mask_path = join(out_dir, 'tmp_mask.nii.gz')
        save_nifti(mask_path, mask, affine)

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
            save_nifti(path, arr.astype(np.uint8), affine)

        # CSD Reconstruction
        reconst_csd_flow = ReconstCSDFlow()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            pf_track_pam.run(pam_path, wm_path, gm_path, csf_path, seeds_path)
        tractogram_path = \
            pf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))

        # Test that tracking returns seeds
        pf_track_pam = PFTrackingPAMFlow()
        pf_track_pam._force_overwrite = True
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            pf_track_pam.run(pam_path,
                             wm_path,
                             gm_path,
                             csf_path,
                             seeds_path,
                             save_seeds=True)
        tractogram_path = \
            pf_track_pam.last_generated_outputs['out_tractogram']
        assert_true(tractogram_has_seeds(tractogram_path))
        assert_true(seeds_are_same_space_as_streamlines(tractogram_path))


def test_local_fiber_tracking_workflow():
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames('small_64D')
        volume, affine = load_nifti(data_path)
        mask = np.ones_like(volume[:, :, :, 0], dtype=np.uint8)
        mask_path = join(out_dir, 'tmp_mask.nii.gz')
        save_nifti(mask_path, mask, affine)

        reconst_csd_flow = ReconstCSDFlow()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            reconst_csd_flow.run(data_path, bval_path, bvec_path, mask_path,
                                 out_dir=out_dir, extract_pam_values=True)

        pam_path = reconst_csd_flow.last_generated_outputs['out_pam']
        gfa_path = reconst_csd_flow.last_generated_outputs['out_gfa']

        # Create seeding mask by thresholding the gfa
        mask_flow = MaskFlow()
        mask_flow.run(gfa_path, 0.8, out_dir=out_dir)
        seeds_path = mask_flow.last_generated_outputs['out_mask']
        mask_path = mask_flow.last_generated_outputs['out_mask']

        gfa_img, gfa_affine = load_nifti(gfa_path)
        save_nifti(gfa_path, gfa_img, gfa_affine)

        # Test tracking with pam no sh
        lf_track_pam = LocalFiberTrackingPAMFlow()
        lf_track_pam._force_overwrite = True
        assert_equal(lf_track_pam.get_short_name(), 'track_local')
        lf_track_pam.run(pam_path, gfa_path, seeds_path)
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))

        # Test tracking with binary stopping criterion
        lf_track_pam = LocalFiberTrackingPAMFlow()
        lf_track_pam._force_overwrite = True
        lf_track_pam.run(pam_path, mask_path, seeds_path,
                         use_binary_mask=True)

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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            lf_track_pam.run(pam_path, gfa_path, seeds_path,
                             tracking_method="deterministic")
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))

        # Test tracking with pam with sh and probabilistic getter
        lf_track_pam = LocalFiberTrackingPAMFlow()
        lf_track_pam._force_overwrite = True
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            lf_track_pam.run(pam_path, gfa_path, seeds_path,
                             tracking_method="probabilistic")
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))

        # Test tracking with pam with sh and closest peaks getter
        lf_track_pam = LocalFiberTrackingPAMFlow()
        lf_track_pam._force_overwrite = True
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            lf_track_pam.run(pam_path, gfa_path, seeds_path,
                             tracking_method="closestpeaks")
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_false(is_tractogram_empty(tractogram_path))

        # Test that tracking returns seeds
        lf_track_pam = LocalFiberTrackingPAMFlow()
        lf_track_pam._force_overwrite = True
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            lf_track_pam.run(pam_path, gfa_path, seeds_path,
                             tracking_method="deterministic",
                             save_seeds=True)
        tractogram_path = \
            lf_track_pam.last_generated_outputs['out_tractogram']
        assert_true(tractogram_has_seeds(tractogram_path))
        assert_true(seeds_are_same_space_as_streamlines(tractogram_path))


def is_tractogram_empty(tractogram_path):
    tractogram_file = \
        nib.streamlines.load(tractogram_path)

    return len(tractogram_file.tractogram) == 0


def tractogram_has_seeds(tractogram_path):
    tractogram = \
        nib.streamlines.load(tractogram_path).tractogram

    return len(tractogram.data_per_streamline['seeds']) > 0


def seeds_are_same_space_as_streamlines(tractogram_path):
    sft = load_tractogram(tractogram_path, 'same', bbox_valid_check=False)
    seeds = sft.data_per_streamline['seeds']
    streamlines = sft.streamlines

    for seed, streamline in zip(seeds, streamlines):
        map_res = list(map(lambda x: np.allclose(seed, x,
                                                 atol=1e-2,
                                                 rtol=1e-4), streamline))
        # If no point is close to the seed, it likely means that the seed is
        # not in the same space as the streamline
        if not np.any(map_res):
            return False

    return True
