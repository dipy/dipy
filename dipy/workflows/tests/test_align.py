from os.path import join as pjoin
import os.path
from tempfile import TemporaryDirectory

import numpy.testing as npt
import numpy as np
import nibabel as nib
import pytest
from dipy.utils.optpkg import optional_package
from dipy.align.tests.test_imwarp import get_synthetic_warped_circle
from dipy.align.tests.test_parzenhist import setup_random_transform
from dipy.align.transforms import regtransforms
from dipy.data import get_fnames
from dipy.io.image import save_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.streamline import Streamlines
from dipy.workflows.align import (ImageRegistrationFlow, SynRegistrationFlow,
                                  ApplyTransformFlow, ResliceFlow,
                                  SlrWithQbxFlow, MotionCorrectionFlow,
                                  BundleWarpFlow)
from dipy.testing.decorators import set_random_number_generator

_, have_pd, _ = optional_package("pandas")


def test_reslice():

    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames('small_25')
        volume = load_nifti_data(data_path)

        reslice_flow = ResliceFlow()
        reslice_flow.run(data_path, [1.5, 1.5, 1.5], out_dir=out_dir)

        out_path = reslice_flow.last_generated_outputs['out_resliced']
        resliced = load_nifti_data(out_path)

        npt.assert_equal(resliced.shape[0] > volume.shape[0], True)
        npt.assert_equal(resliced.shape[1] > volume.shape[1], True)
        npt.assert_equal(resliced.shape[2] > volume.shape[2], True)
        npt.assert_equal(resliced.shape[-1], volume.shape[-1])


def test_slr_flow():
    with TemporaryDirectory() as out_dir:
        data_path = get_fnames('fornix')

        fornix = load_tractogram(data_path, 'same',
                                 bbox_valid_check=False).streamlines

        f = Streamlines(fornix)
        f1 = f.copy()

        f1_path = pjoin(out_dir, "f1.trk")
        sft = StatefulTractogram(f1, data_path, Space.RASMM)
        save_tractogram(sft, f1_path, bbox_valid_check=False)

        f2 = f1.copy()
        f2._data += np.array([50, 0, 0])

        f2_path = pjoin(out_dir, "f2.trk")
        sft = StatefulTractogram(f2, data_path, Space.RASMM)
        save_tractogram(sft, f2_path, bbox_valid_check=False)

        slr_flow = SlrWithQbxFlow(force=True)
        slr_flow.run(f1_path, f2_path)

        out_path = slr_flow.last_generated_outputs['out_moved']

        npt.assert_equal(os.path.isfile(out_path), True)


@set_random_number_generator(1234)
def test_image_registration(rng):
    with TemporaryDirectory() as temp_out_dir:

        static, moving, static_g2w, moving_g2w, smask, mmask, M\
            = setup_random_transform(transform=regtransforms[('AFFINE', 3)],
                                     rfactor=0.1, rng=rng)

        save_nifti(pjoin(temp_out_dir, 'b0.nii.gz'), data=static,
                   affine=static_g2w)
        save_nifti(pjoin(temp_out_dir, 't1.nii.gz'), data=moving,
                   affine=moving_g2w)

        static_image_file = pjoin(temp_out_dir, 'b0.nii.gz')
        moving_image_file = pjoin(temp_out_dir, 't1.nii.gz')

        image_registration_flow = ImageRegistrationFlow()

        def read_distance(qual_fname):
            with open(pjoin(temp_out_dir, qual_fname), 'r') as f:
                return float(f.readlines()[-1])

        def test_com():

            out_moved = pjoin(temp_out_dir, "com_moved.nii.gz")
            out_affine = pjoin(temp_out_dir, "com_affine.txt")

            image_registration_flow._force_overwrite = True
            image_registration_flow.run(static_image_file,
                                        moving_image_file,
                                        transform='com',
                                        out_dir=temp_out_dir,
                                        out_moved=out_moved,
                                        out_affine=out_affine)
            check_existence(out_moved, out_affine)

        def test_translation():

            out_moved = pjoin(temp_out_dir, "trans_moved.nii.gz")
            out_affine = pjoin(temp_out_dir, "trans_affine.txt")

            image_registration_flow._force_overwrite = True
            image_registration_flow.run(static_image_file,
                                        moving_image_file,
                                        transform='trans',
                                        out_dir=temp_out_dir,
                                        out_moved=out_moved,
                                        out_affine=out_affine,
                                        save_metric=True,
                                        level_iters=[100, 10, 1],
                                        out_quality='trans_q.txt')

            dist = read_distance('trans_q.txt')
            npt.assert_almost_equal(float(dist), -0.42097809101318934, 1)
            check_existence(out_moved, out_affine)

        def test_rigid():

            out_moved = pjoin(temp_out_dir, "rigid_moved.nii.gz")
            out_affine = pjoin(temp_out_dir, "rigid_affine.txt")

            image_registration_flow._force_overwrite = True
            image_registration_flow.run(static_image_file,
                                        moving_image_file,
                                        transform='rigid',
                                        out_dir=temp_out_dir,
                                        out_moved=out_moved,
                                        out_affine=out_affine,
                                        save_metric=True,
                                        level_iters=[100, 10, 1],
                                        out_quality='rigid_q.txt')

            dist = read_distance('rigid_q.txt')
            npt.assert_almost_equal(dist, -0.6900534794005155, 1)
            check_existence(out_moved, out_affine)

        def test_rigid_isoscaling():

            out_moved = pjoin(temp_out_dir, "rigid_isoscaling_moved.nii.gz")
            out_affine = pjoin(temp_out_dir, "rigid_isoscaling_affine.txt")

            image_registration_flow._force_overwrite = True
            image_registration_flow.run(static_image_file,
                                        moving_image_file,
                                        transform='rigid_isoscaling',
                                        out_dir=temp_out_dir,
                                        out_moved=out_moved,
                                        out_affine=out_affine,
                                        save_metric=True,
                                        level_iters=[100, 10, 1],
                                        out_quality='rigid_isoscaling_q.txt')

            dist = read_distance('rigid_isoscaling_q.txt')
            npt.assert_almost_equal(dist, -0.6960044668271375, 1)
            check_existence(out_moved, out_affine)

        def test_rigid_scaling():

            out_moved = pjoin(temp_out_dir, "rigid_scaling_moved.nii.gz")
            out_affine = pjoin(temp_out_dir, "rigid_scaling_affine.txt")

            image_registration_flow._force_overwrite = True
            image_registration_flow.run(static_image_file,
                                        moving_image_file,
                                        transform='rigid_scaling',
                                        out_dir=temp_out_dir,
                                        out_moved=out_moved,
                                        out_affine=out_affine,
                                        save_metric=True,
                                        level_iters=[100, 10, 1],
                                        out_quality='rigid_scaling_q.txt')

            dist = read_distance('rigid_scaling_q.txt')
            npt.assert_almost_equal(dist, -0.698688892993124, 1)
            check_existence(out_moved, out_affine)

        def test_affine():

            out_moved = pjoin(temp_out_dir, "affine_moved.nii.gz")
            out_affine = pjoin(temp_out_dir, "affine_affine.txt")

            image_registration_flow._force_overwrite = True
            image_registration_flow.run(static_image_file,
                                        moving_image_file,
                                        transform='affine',
                                        out_dir=temp_out_dir,
                                        out_moved=out_moved,
                                        out_affine=out_affine,
                                        save_metric=True,
                                        level_iters=[100, 10, 1],
                                        out_quality='affine_q.txt')

            dist = read_distance('affine_q.txt')
            npt.assert_almost_equal(dist, -0.7670650775914811, 1)
            check_existence(out_moved, out_affine)

        # Creating the erroneous behavior
        def test_err():
            image_registration_flow._force_overwrite = True
            npt.assert_raises(ValueError, image_registration_flow.run,
                              static_image_file,
                              moving_image_file,
                              transform='notransform')

            image_registration_flow._force_overwrite = True
            npt.assert_raises(ValueError, image_registration_flow.run,
                              static_image_file,
                              moving_image_file,
                              metric='wrong_metric')

        def check_existence(movedfile, affine_mat_file):
            assert os.path.exists(movedfile)
            assert os.path.exists(affine_mat_file)
            return True

        test_com()
        test_translation()
        test_rigid()
        test_rigid_isoscaling()
        test_rigid_scaling()
        test_affine()
        test_err()


def test_apply_transform_error():
    flow = ApplyTransformFlow()
    npt.assert_raises(ValueError, flow.run,
                      'my_fake_static.nii.gz', 'my_fake_moving.nii.gz',
                      'my_fake_map.nii.gz', transform_type='wrong_type')


def test_apply_affine_transform():
    with TemporaryDirectory() as temp_out_dir:

        factors = {
            ('TRANSLATION', 3): (2.0, None, np.array([2.3, 4.5, 1.7])),
            ('RIGID', 3): (0.1, None, np.array([0.1, 0.15, -0.11, 2.3, 4.5,
                                                1.7])),
            ('RIGIDISOSCALING', 3): (0.1, None, np.array([0.1, 0.15, -0.11,
                                                         2.3, 4.5, 1.7,
                                                          0.8])),
            ('RIGIDSCALING', 3): (0.1, None, np.array([0.1, 0.15, -0.11, 2.3,
                                                       4.5, 1.7, 0.8, 0.9,
                                                       1.1])),
            ('AFFINE', 3): (0.1, None, np.array([0.99, -0.05, 0.03, 1.3,
                                                 0.05, 0.99, -0.10, 2.5,
                                                 -0.07, 0.10, 0.99, -1.4]))}

        image_registration_flow = ImageRegistrationFlow()
        apply_trans = ApplyTransformFlow()

        for i in factors.keys():
            static, moving, static_g2w, moving_g2w, smask, mmask, M = \
                setup_random_transform(transform=regtransforms[i],
                                       rfactor=factors[i][0])

            stat_file = str(i[0]) + '_static.nii.gz'
            mov_file = str(i[0]) + '_moving.nii.gz'

            save_nifti(pjoin(temp_out_dir, stat_file), data=static,
                       affine=static_g2w)

            save_nifti(pjoin(temp_out_dir, mov_file), data=moving,
                       affine=moving_g2w)

            static_image_file = pjoin(temp_out_dir,
                                      str(i[0]) + '_static.nii.gz')
            moving_image_file = pjoin(temp_out_dir,
                                      str(i[0]) + '_moving.nii.gz')

            out_moved = pjoin(temp_out_dir,
                              str(i[0]) + "_moved.nii.gz")
            out_affine = pjoin(temp_out_dir,
                               str(i[0]) + "_affine.txt")

            if str(i[0]) == "TRANSLATION":
                transform_type = "trans"
            elif str(i[0]) == "RIGIDISOSCALING":
                transform_type = "rigid_isoscaling"
            elif str(i[0]) == "RIGIDSCALING":
                transform_type = "rigid_scaling"
            else:
                transform_type = str(i[0]).lower()

            image_registration_flow.run(static_image_file, moving_image_file,
                                        transform=transform_type,
                                        out_dir=temp_out_dir,
                                        out_moved=out_moved,
                                        out_affine=out_affine,
                                        level_iters=[1, 1, 1],
                                        save_metric=False)

            # Checking for the created moved file.
            assert os.path.exists(out_moved)
            assert os.path.exists(out_affine)

        images = pjoin(temp_out_dir, '*moving*')
        apply_trans.run(static_image_file, images,
                        out_dir=temp_out_dir,
                        transform_map_file=out_affine)

        # Checking for the transformed file.
        assert os.path.exists(pjoin(temp_out_dir, "transformed.nii.gz"))


def test_motion_correction():
    data_path, fbvals_path, fbvecs_path = get_fnames('small_64D')
    volume = load_nifti_data(data_path)

    with TemporaryDirectory() as out_dir:
        # Use an abbreviated data-set:
        img = nib.load(data_path)
        data = img.get_fdata()[..., :10]
        nib.save(nib.Nifti1Image(data, img.affine),
                 os.path.join(out_dir, 'data.nii.gz'))
        # Save a subset:
        bvals = np.loadtxt(fbvals_path)
        bvecs = np.loadtxt(fbvecs_path)
        np.savetxt(os.path.join(out_dir, 'bvals.txt'), bvals[:10])
        np.savetxt(os.path.join(out_dir, 'bvecs.txt'), bvecs[:10])

        motion_correction_flow = MotionCorrectionFlow()

        motion_correction_flow._force_overwrite = True
        motion_correction_flow.run(os.path.join(out_dir, 'data.nii.gz'),
                                   os.path.join(out_dir, 'bvals.txt'),
                                   os.path.join(out_dir, 'bvecs.txt'),
                                   out_dir=out_dir)
        out_path = motion_correction_flow.last_generated_outputs['out_moved']
        corrected = load_nifti_data(out_path)

        npt.assert_equal(corrected.shape, data.shape)
        npt.assert_equal(corrected.min(), data.min())
        npt.assert_equal(corrected.max(), data.max())


def test_syn_registration_flow():
    moving_data, static_data = get_synthetic_warped_circle(40)
    moving_data[..., :10] = 0
    moving_data[..., -1:-11:-1] = 0
    static_data[..., :10] = 0
    static_data[..., -1:-11:-1] = 0

    syn_flow = SynRegistrationFlow()

    with TemporaryDirectory() as out_dir:
        static_img = nib.Nifti1Image(static_data.astype(float), np.eye(4))
        fname_static = pjoin(out_dir, 'tmp_static.nii.gz')
        nib.save(static_img, fname_static)

        moving_img = nib.Nifti1Image(moving_data.astype(float), np.eye(4))
        fname_moving = pjoin(out_dir, 'tmp_moving.nii.gz')
        nib.save(moving_img, fname_moving)

        positional_args = [fname_static, fname_moving]

        # Test the cc metric
        metric_optional_args = {'metric': 'cc',
                                'mopt_sigma_diff': 2.0,
                                'mopt_radius': 4
                                }
        optimizer_optional_args = {'level_iters': [10, 10, 5],
                                   'step_length': 0.25,
                                   'opt_tol': 1e-5,
                                   'inv_iter': 20,
                                   'inv_tol': 1e-3,
                                   'ss_sigma_factor': 0.2
                                   }

        all_args = dict(metric_optional_args, **optimizer_optional_args)
        syn_flow.run(*positional_args,
                     out_dir=out_dir,
                     **all_args
                     )

        warped_path = syn_flow.last_generated_outputs['out_warped']
        npt.assert_equal(os.path.isfile(warped_path), True)
        warped_map_path = syn_flow.last_generated_outputs['out_field']
        npt.assert_equal(os.path.isfile(warped_map_path), True)

        # Test the ssd metric
        metric_optional_args = {'metric': 'ssd',
                                'mopt_smooth': 4.0,
                                'mopt_inner_iter': 5,
                                'mopt_step_type': 'demons',
                                }
        optimizer_optional_args = {'level_iters': [200, 100, 50, 25],
                                   'step_length': 0.5,
                                   'opt_tol': 1e-4,
                                   'inv_iter': 40,
                                   'inv_tol': 1e-3,
                                   'ss_sigma_factor': 0.2
                                   }

        all_args = dict(metric_optional_args, **optimizer_optional_args)
        syn_flow.run(*positional_args,
                     out_dir=out_dir,
                     **all_args
                     )

        warped_path = syn_flow.last_generated_outputs['out_warped']
        npt.assert_equal(os.path.isfile(warped_path), True)
        warped_map_path = syn_flow.last_generated_outputs['out_field']
        npt.assert_equal(os.path.isfile(warped_map_path), True)

        # Test the em metric
        metric_optional_args = {'metric': 'em',
                                'mopt_smooth': 1.0,
                                'mopt_inner_iter': 5,
                                'mopt_step_type': 'gauss_newton',
                                }
        optimizer_optional_args = {'level_iters': [200, 100, 50, 25],
                                   'step_length': 0.5,
                                   'opt_tol': 1e-4,
                                   'inv_iter': 40,
                                   'inv_tol': 1e-3,
                                   'ss_sigma_factor': 0.2
                                   }

        all_args = dict(metric_optional_args, **optimizer_optional_args)
        syn_flow.run(*positional_args,
                     out_dir=out_dir,
                     **all_args
                     )

        warped_path = syn_flow.last_generated_outputs['out_warped']
        npt.assert_equal(os.path.isfile(warped_path), True)
        warped_map_path = syn_flow.last_generated_outputs['out_field']
        npt.assert_equal(os.path.isfile(warped_map_path), True)


@pytest.mark.skipif(not have_pd, reason='Requires pandas')
def test_bundlewarp_flow():
    with TemporaryDirectory() as out_dir:

        data_path = get_fnames('fornix')

        fornix = load_tractogram(data_path, 'same',
                                 bbox_valid_check=False).streamlines

        f = Streamlines(fornix)
        f1 = f.copy()

        f1_path = pjoin(out_dir, "f1.trk")
        sft = StatefulTractogram(f1, data_path, Space.RASMM)
        save_tractogram(sft, f1_path, bbox_valid_check=False)

        f2 = f1.copy()
        f2._data += np.array([50, 0, 0])

        f2_path = pjoin(out_dir, "f2.trk")
        sft = StatefulTractogram(f2, data_path, Space.RASMM)
        save_tractogram(sft, f2_path, bbox_valid_check=False)

        bw_flow = BundleWarpFlow(force=True)
        bw_flow.run(f1_path, f2_path, out_dir=out_dir)

        out_linearly_moved = pjoin(out_dir, "linearly_moved.trk")
        out_nonlinearly_moved = pjoin(out_dir, "nonlinearly_moved.trk")

        assert os.path.exists(out_linearly_moved)
        assert os.path.exists(out_nonlinearly_moved)
