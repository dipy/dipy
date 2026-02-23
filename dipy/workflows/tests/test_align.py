from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from dipy.align.tests.test_imwarp import get_synthetic_warped_circle
from dipy.align.tests.test_parzenhist import setup_random_transform
from dipy.align.transforms import regtransforms
from dipy.data import get_fnames
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.testing.decorators import set_random_number_generator
from dipy.tracking.streamline import Streamlines
from dipy.utils.optpkg import optional_package
from dipy.workflows.align import (
    ApplyTransformFlow,
    BundleWarpFlow,
    ImageRegistrationFlow,
    MotionCorrectionFlow,
    ResliceFlow,
    SlrWithQbxFlow,
    SynRegistrationFlow,
)

_, have_pd, _ = optional_package("pandas")


def test_reslice():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames(name="small_25")
        volume = load_nifti_data(data_path)

        reslice_flow = ResliceFlow()
        reslice_flow.run(data_path, [1.5, 1.5, 1.5], out_dir=out_dir)

        out_path = reslice_flow.last_generated_outputs["out_resliced"]
        resliced = load_nifti_data(out_path)

        npt.assert_equal(resliced.shape[0] > volume.shape[0], True)
        npt.assert_equal(resliced.shape[1] > volume.shape[1], True)
        npt.assert_equal(resliced.shape[2] > volume.shape[2], True)
        npt.assert_equal(resliced.shape[-1], volume.shape[-1])


def test_reslice_auto_voxsize(caplog):
    """Test ResliceFlow with automatic voxel size calculation."""
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames(name="small_25")
        volume = load_nifti_data(data_path)

        reslice_flow = ResliceFlow()
        reslice_flow.run(data_path, out_dir=out_dir)

        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) > 0, "Expected WARNING level log message"
        assert any(
            "new_vox_size not provided" in record.message for record in warning_records
        ), "Expected warning about auto-calculation"
        assert any(
            "vox_factor=0.14" in record.message for record in warning_records
        ), "Expected warning to include vox_factor value"

        out_path = reslice_flow.last_generated_outputs["out_resliced"]
        resliced = load_nifti_data(out_path)

        npt.assert_equal(resliced.shape[-1], volume.shape[-1])


def test_reslice_custom_voxfactor(caplog):
    """Test ResliceFlow with custom vox_factor parameter."""
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames(name="small_25")
        volume = load_nifti_data(data_path)

        reslice_flow = ResliceFlow()
        custom_factor = 0.5
        reslice_flow.run(data_path, vox_factor=custom_factor, out_dir=out_dir)

        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) > 0, "Expected WARNING level log message"
        assert any(
            f"vox_factor={custom_factor}" in record.message
            for record in warning_records
        ), f"Expected warning to include vox_factor={custom_factor}"

        out_path = reslice_flow.last_generated_outputs["out_resliced"]
        resliced = load_nifti_data(out_path)

        npt.assert_equal(resliced.shape[-1], volume.shape[-1])


def test_reslice_skip_when_matching(caplog):
    """Test ResliceFlow skips reslicing when voxel size already matches."""
    import logging

    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames(name="small_25")
        volume = load_nifti_data(data_path)
        _, _, zooms = load_nifti(data_path, return_voxsize=True)

        with caplog.at_level(logging.INFO, logger="dipy"):
            reslice_flow = ResliceFlow()
            reslice_flow.run(data_path, new_vox_size=list(zooms[:3]), out_dir=out_dir)

        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        assert any("Skipping reslicing" in record.message for record in info_records), (
            "Expected INFO message about skipping. "
            f"Found: {[r.message for r in info_records]}"
        )
        assert any(
            "already matches target" in record.message for record in info_records
        ), "Expected INFO message about matching voxel size"

        out_path = reslice_flow.last_generated_outputs["out_resliced"]
        resliced = load_nifti_data(out_path)

        npt.assert_equal(resliced.shape, volume.shape)


def test_slr_flow(caplog):
    with TemporaryDirectory() as out_dir:
        data_path = get_fnames(name="fornix")

        sft = load_tractogram(data_path, "same", bbox_valid_check=False)
        sft.streamlines._data += np.array([50, 0, 0])
        moved_path = Path(out_dir) / "moved.trx"
        save_tractogram(sft, moved_path, bbox_valid_check=False)

        slr_flow = SlrWithQbxFlow(force=True)
        slr_flow.run(data_path, moved_path, out_dir=out_dir, bbox_valid_check=False)

        out_path = slr_flow.last_generated_outputs["out_moved"]

        npt.assert_equal(Path(out_path).is_file(), True)

        sft = sft.from_sft(np.array([]), sft)
        empty_path = Path(out_dir) / "empty.trk"
        save_tractogram(sft, empty_path, bbox_valid_check=False)

        slr_flow = SlrWithQbxFlow(force=True)

        # Test empty static file
        slr_flow.run(
            empty_path,
            moved_path,
            out_dir=out_dir,
            bbox_valid_check=False,
        )

        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_records) > 0, "Expected ERROR level log message"
        error_msg = f"Static file {empty_path} is empty"
        assert any(err.msg in error_msg for err in error_records)

        caplog.clear()

        # Test empty moving file
        slr_flow.run(
            data_path,
            empty_path,
            out_dir=out_dir,
            bbox_valid_check=False,
        )

        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_records) > 0, "Expected ERROR level log message"
        error_msg = f"Moving file {empty_path} is empty"
        assert any(err.msg in error_msg for err in error_records)


def test_slr_flow_empty_after_length_filtering(caplog):
    """Test that SlrWithQbxFlow logs error when all streamlines are filtered
    out by length constraints.
    """
    with TemporaryDirectory() as out_dir:
        data_path = get_fnames(name="fornix")

        sft = load_tractogram(data_path, "same", bbox_valid_check=False)
        moved_path = Path(out_dir) / "moved.trx"
        save_tractogram(sft, moved_path, bbox_valid_check=False)

        slr_flow = SlrWithQbxFlow(force=True)

        caplog.clear()
        slr_flow.run(
            data_path,
            moved_path,
            out_dir=out_dir,
            bbox_valid_check=False,
            greater_than=1000,
            less_than=np.inf,
        )

        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_records) > 0, "Expected ERROR level log message"
        assert any("SLR with QBX failed" in err.message for err in error_records)

        caplog.clear()
        slr_flow.run(
            data_path,
            moved_path,
            out_dir=out_dir,
            bbox_valid_check=False,
            greater_than=0,
            less_than=1,
        )

        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_records) > 0, "Expected ERROR level log message"
        assert any("SLR with QBX failed" in err.message for err in error_records)


@set_random_number_generator(1234)
def test_image_registration(rng):
    with TemporaryDirectory() as temp_out_dir:
        static, moving, static_g2w, moving_g2w, smask, mmask, M = (
            setup_random_transform(
                transform=regtransforms[("AFFINE", 3)], rfactor=0.1, rng=rng
            )
        )

        save_nifti(Path(temp_out_dir) / "b0.nii.gz", data=static, affine=static_g2w)
        save_nifti(Path(temp_out_dir) / "t1.nii.gz", data=moving, affine=moving_g2w)
        # simulate three direction DWI by repeating b0 three times
        save_nifti(
            Path(temp_out_dir) / "dwi.nii.gz",
            data=np.repeat(static[..., None], 3, axis=-1),
            affine=static_g2w,
        )

        static_image_file = Path(temp_out_dir) / "b0.nii.gz"
        moving_image_file = Path(temp_out_dir) / "t1.nii.gz"
        dwi_image_file = Path(temp_out_dir) / "dwi.nii.gz"

        image_registration_flow = ImageRegistrationFlow()
        apply_trans = ApplyTransformFlow()

        def read_distance(qual_fname):
            with open(Path(temp_out_dir) / qual_fname, "r") as f:
                return float(f.readlines()[-1])

        def test_com():
            out_moved = Path(temp_out_dir) / "com_moved.nii.gz"
            out_affine = Path(temp_out_dir) / "com_affine.txt"

            image_registration_flow._force_overwrite = True
            image_registration_flow.run(
                static_image_file,
                moving_image_file,
                transform="com",
                out_dir=temp_out_dir,
                out_moved=out_moved,
                out_affine=out_affine,
            )
            check_existence(out_moved, out_affine)

        def test_translation():
            out_moved = Path(temp_out_dir) / "trans_moved.nii.gz"
            out_affine = Path(temp_out_dir) / "trans_affine.txt"

            image_registration_flow._force_overwrite = True
            image_registration_flow.run(
                static_image_file,
                moving_image_file,
                transform="trans",
                out_dir=temp_out_dir,
                out_moved=out_moved,
                out_affine=out_affine,
                save_metric=True,
                level_iters=[100, 10, 1],
                out_quality="trans_q.txt",
            )

            dist = read_distance("trans_q.txt")
            npt.assert_almost_equal(float(dist), -0.42097809101318934, 1)
            check_existence(out_moved, out_affine)

        def test_rigid():
            out_moved = Path(temp_out_dir) / "rigid_moved.nii.gz"
            out_affine = Path(temp_out_dir) / "rigid_affine.txt"

            image_registration_flow._force_overwrite = True
            image_registration_flow.run(
                static_image_file,
                moving_image_file,
                transform="rigid",
                out_dir=temp_out_dir,
                out_moved=out_moved,
                out_affine=out_affine,
                save_metric=True,
                level_iters=[100, 10, 1],
                out_quality="rigid_q.txt",
            )

            dist = read_distance("rigid_q.txt")
            npt.assert_almost_equal(dist, -0.6900534794005155, 1)
            check_existence(out_moved, out_affine)

        def test_rigid_isoscaling():
            out_moved = Path(temp_out_dir) / "rigid_isoscaling_moved.nii.gz"
            out_affine = Path(temp_out_dir) / "rigid_isoscaling_affine.txt"

            image_registration_flow._force_overwrite = True
            image_registration_flow.run(
                static_image_file,
                moving_image_file,
                transform="rigid_isoscaling",
                out_dir=temp_out_dir,
                out_moved=out_moved,
                out_affine=out_affine,
                save_metric=True,
                level_iters=[100, 10, 1],
                out_quality="rigid_isoscaling_q.txt",
            )

            dist = read_distance("rigid_isoscaling_q.txt")
            npt.assert_almost_equal(dist, -0.6960044668271375, 1)
            check_existence(out_moved, out_affine)

        def test_rigid_scaling():
            out_moved = Path(temp_out_dir) / "rigid_scaling_moved.nii.gz"
            out_affine = Path(temp_out_dir) / "rigid_scaling_affine.txt"

            image_registration_flow._force_overwrite = True
            image_registration_flow.run(
                static_image_file,
                moving_image_file,
                transform="rigid_scaling",
                out_dir=temp_out_dir,
                out_moved=out_moved,
                out_affine=out_affine,
                save_metric=True,
                level_iters=[100, 10, 1],
                out_quality="rigid_scaling_q.txt",
            )

            dist = read_distance("rigid_scaling_q.txt")
            npt.assert_almost_equal(dist, -0.698688892993124, 1)
            check_existence(out_moved, out_affine)

        def test_affine():
            out_moved = Path(temp_out_dir) / "affine_moved.nii.gz"
            out_affine = Path(temp_out_dir) / "affine_affine.txt"

            image_registration_flow._force_overwrite = True
            image_registration_flow.run(
                static_image_file,
                moving_image_file,
                transform="affine",
                out_dir=temp_out_dir,
                out_moved=out_moved,
                out_affine=out_affine,
                save_metric=True,
                level_iters=[100, 10, 1],
                out_quality="affine_q.txt",
            )

            dist = read_distance("affine_q.txt")
            npt.assert_almost_equal(dist, -0.7670650775914811, 1)
            check_existence(out_moved, out_affine)

        # Creating the erroneous behavior
        def test_err():
            image_registration_flow._force_overwrite = True
            npt.assert_raises(
                ValueError,
                image_registration_flow.run,
                static_image_file,
                moving_image_file,
                transform="notransform",
            )

            image_registration_flow._force_overwrite = True
            npt.assert_raises(
                ValueError,
                image_registration_flow.run,
                static_image_file,
                moving_image_file,
                metric="wrong_metric",
            )

        def check_existence(movedfile, affine_mat_file):
            assert Path(movedfile).exists()
            assert Path(affine_mat_file).exists()
            return True

        def test_4D_static():
            out_moved = Path(temp_out_dir) / "trans_moved.nii.gz"
            out_affine = Path(temp_out_dir) / "trans_affine.txt"

            image_registration_flow._force_overwrite = True
            kwargs = {
                "static_image_files": dwi_image_file,
                "moving_image_files": moving_image_file,
                "transform": "trans",
                "out_dir": temp_out_dir,
                "out_moved": out_moved,
                "out_affine": out_affine,
                "save_metric": True,
                "level_iters": [100, 10, 1],
                "out_quality": "trans_q.txt",
            }
            with pytest.raises(ValueError, match="Dimension mismatch"):
                image_registration_flow.run(**kwargs)

            image_registration_flow.run(static_vol_idx=0, **kwargs)

            dist = read_distance("trans_q.txt")
            npt.assert_almost_equal(float(dist), -0.42097809101318934, 1)
            check_existence(out_moved, out_affine)

            apply_trans.run(
                static_image_files=dwi_image_file,
                moving_image_files=moving_image_file,
                out_dir=temp_out_dir,
                transform_map_file=out_affine,
            )

            # Checking for the transformed volume shape
            volume = load_nifti_data(Path(temp_out_dir) / "transformed.nii.gz")
            assert volume.ndim == 3

        def test_4D_moving():
            out_moved = Path(temp_out_dir) / "trans_moved.nii.gz"
            out_affine = Path(temp_out_dir) / "trans_affine.txt"

            image_registration_flow._force_overwrite = True

            kwargs = {
                "static_image_files": static_image_file,
                "moving_image_files": dwi_image_file,
                "transform": "trans",
                "out_dir": temp_out_dir,
                "out_moved": out_moved,
                "out_affine": out_affine,
                "save_metric": True,
                "level_iters": [100, 10, 1],
                "out_quality": "trans_q.txt",
            }
            with pytest.raises(ValueError, match="Dimension mismatch"):
                image_registration_flow.run(**kwargs)

            image_registration_flow.run(moving_vol_idx=0, **kwargs)

            dist = read_distance("trans_q.txt")
            npt.assert_almost_equal(float(dist), -1.0002607616786339, 1)
            check_existence(out_moved, out_affine)

            apply_trans.run(
                static_image_files=static_image_file,
                moving_image_files=dwi_image_file,
                out_dir=temp_out_dir,
                transform_map_file=out_affine,
                out_file="transformed2.nii.gz",
            )

            # Checking for the transformed volume shape
            volume = load_nifti_data(Path(temp_out_dir) / "transformed2.nii.gz")
            assert volume.ndim == 4

        test_com()
        test_translation()
        test_rigid()
        test_rigid_isoscaling()
        test_rigid_scaling()
        test_affine()
        test_err()
        test_4D_static()
        test_4D_moving()


def test_apply_transform_type_error():
    flow = ApplyTransformFlow()
    npt.assert_raises(
        ValueError,
        flow.run,
        "my_fake_static.nii.gz",
        "my_fake_moving.nii.gz",
        "my_fake_map.nii.gz",
        transform_type="wrong_type",
    )


def test_apply_transform_interp_error():
    flow = ApplyTransformFlow()
    npt.assert_raises(
        ValueError,
        flow.run,
        "my_fake_static.nii.gz",
        "my_fake_moving.nii.gz",
        "my_fake_map.nii.gz",
        transform_type="affine",
        interpolation="wrong_interp",
    )


def test_apply_affine_transform():
    with TemporaryDirectory() as temp_out_dir:
        factors = {
            ("TRANSLATION", 3): (2.0, None, np.array([2.3, 4.5, 1.7])),
            ("RIGID", 3): (0.1, None, np.array([0.1, 0.15, -0.11, 2.3, 4.5, 1.7])),
            ("RIGIDISOSCALING", 3): (
                0.1,
                None,
                np.array([0.1, 0.15, -0.11, 2.3, 4.5, 1.7, 0.8]),
            ),
            ("RIGIDSCALING", 3): (
                0.1,
                None,
                np.array([0.1, 0.15, -0.11, 2.3, 4.5, 1.7, 0.8, 0.9, 1.1]),
            ),
            ("AFFINE", 3): (
                0.1,
                None,
                np.array(
                    [
                        0.99,
                        -0.05,
                        0.03,
                        1.3,
                        0.05,
                        0.99,
                        -0.10,
                        2.5,
                        -0.07,
                        0.10,
                        0.99,
                        -1.4,
                    ]
                ),
            ),
        }

        image_registration_flow = ImageRegistrationFlow()
        apply_trans = ApplyTransformFlow()

        for i in factors.keys():
            static, moving, static_g2w, moving_g2w, smask, mmask, M = (
                setup_random_transform(
                    transform=regtransforms[i], rfactor=factors[i][0]
                )
            )

            stat_file = str(i[0]) + "_static.nii.gz"
            mov_file = str(i[0]) + "_moving.nii.gz"

            save_nifti(Path(temp_out_dir) / stat_file, data=static, affine=static_g2w)

            save_nifti(Path(temp_out_dir) / mov_file, data=moving, affine=moving_g2w)

            static_image_file = Path(temp_out_dir) / str(i[0] + "_static.nii.gz")
            moving_image_file = Path(temp_out_dir) / str(i[0] + "_moving.nii.gz")

            out_moved = Path(temp_out_dir) / str(i[0] + "_moved.nii.gz")
            out_affine = Path(temp_out_dir) / str(i[0] + "_affine.txt")

            if str(i[0]) == "TRANSLATION":
                transform_type = "trans"
            elif str(i[0]) == "RIGIDISOSCALING":
                transform_type = "rigid_isoscaling"
            elif str(i[0]) == "RIGIDSCALING":
                transform_type = "rigid_scaling"
            else:
                transform_type = str(i[0]).lower()

            image_registration_flow.run(
                static_image_file,
                moving_image_file,
                transform=transform_type,
                out_dir=temp_out_dir,
                out_moved=out_moved,
                out_affine=out_affine,
                level_iters=[1, 1, 1],
                save_metric=False,
            )

            # Checking for the created moved file.
            assert Path(out_moved).exists()
            assert Path(out_affine).exists()

        images = Path(temp_out_dir) / "*moving*"
        apply_trans.run(
            static_image_file,
            images,
            out_dir=temp_out_dir,
            transform_map_file=out_affine,
        )

        # Checking for the transformed file.
        assert Path(Path(temp_out_dir) / "transformed.nii.gz").exists()

        apply_trans.run(
            static_image_file,
            images,
            out_dir=temp_out_dir,
            transform_map_file=out_affine,
            out_file="transformed_linear.nii.gz",
        )

        assert Path(Path(temp_out_dir) / "transformed_linear.nii.gz").exists()

        apply_trans.run(
            static_image_file,
            images,
            out_dir=temp_out_dir,
            transform_map_file=out_affine,
            interpolation="nearest",
            out_file="transformed_nearest.nii.gz",
        )

        assert Path(Path(temp_out_dir) / "transformed_nearest.nii.gz").exists()


def test_motion_correction():
    data_path, fbvals_path, fbvecs_path = get_fnames(name="small_64D")

    with TemporaryDirectory() as out_dir:
        # Use an abbreviated data-set:
        img = nib.load(data_path)
        data = img.get_fdata()[..., :10]
        nib.save(nib.Nifti1Image(data, img.affine), Path(out_dir) / "data.nii.gz")
        # Save a subset:
        bvals = np.loadtxt(fbvals_path)
        bvecs = np.loadtxt(fbvecs_path)
        np.savetxt(Path(out_dir) / "bvals.txt", bvals[:10])
        np.savetxt(Path(out_dir) / "bvecs.txt", bvecs[:10])

        motion_correction_flow = MotionCorrectionFlow()

        motion_correction_flow._force_overwrite = True
        motion_correction_flow.run(
            str(Path(out_dir) / "data.nii.gz"),
            str(Path(out_dir) / "bvals.txt"),
            str(Path(out_dir) / "bvecs.txt"),
            out_dir=out_dir,
        )
        out_path = motion_correction_flow.last_generated_outputs["out_moved"]
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
        fname_static = Path(out_dir) / "tmp_static.nii.gz"
        nib.save(static_img, fname_static)

        moving_img = nib.Nifti1Image(moving_data.astype(float), np.eye(4))
        fname_moving = Path(out_dir) / "tmp_moving.nii.gz"
        nib.save(moving_img, fname_moving)

        positional_args = [fname_static, fname_moving]

        # Test the cc metric
        metric_optional_args = {
            "metric": "cc",
            "mopt_sigma_diff": 2.0,
            "mopt_radius": 4,
        }
        optimizer_optional_args = {
            "level_iters": [10, 10, 5],
            "step_length": 0.25,
            "opt_tol": 1e-5,
            "inv_iter": 20,
            "inv_tol": 1e-3,
            "ss_sigma_factor": 0.2,
        }

        all_args = dict(metric_optional_args, **optimizer_optional_args)
        syn_flow.run(*positional_args, out_dir=out_dir, **all_args)

        warped_path = syn_flow.last_generated_outputs["out_warped"]
        npt.assert_equal(Path(warped_path).is_file(), True)
        warped_map_path = syn_flow.last_generated_outputs["out_field"]
        npt.assert_equal(Path(warped_map_path).is_file(), True)

        # Test the ssd metric
        metric_optional_args = {
            "metric": "ssd",
            "mopt_smooth": 4.0,
            "mopt_inner_iter": 5,
            "mopt_step_type": "demons",
        }
        optimizer_optional_args = {
            "level_iters": [200, 100, 50, 25],
            "step_length": 0.5,
            "opt_tol": 1e-4,
            "inv_iter": 40,
            "inv_tol": 1e-3,
            "ss_sigma_factor": 0.2,
        }

        all_args = dict(metric_optional_args, **optimizer_optional_args)
        syn_flow.run(*positional_args, out_dir=out_dir, **all_args)

        warped_path = syn_flow.last_generated_outputs["out_warped"]
        npt.assert_equal(Path(warped_path).is_file(), True)
        warped_map_path = syn_flow.last_generated_outputs["out_field"]
        npt.assert_equal(Path(warped_map_path).is_file(), True)

        # Test the em metric
        metric_optional_args = {
            "metric": "em",
            "mopt_smooth": 1.0,
            "mopt_inner_iter": 5,
            "mopt_step_type": "gauss_newton",
        }
        optimizer_optional_args = {
            "level_iters": [200, 100, 50, 25],
            "step_length": 0.5,
            "opt_tol": 1e-4,
            "inv_iter": 40,
            "inv_tol": 1e-3,
            "ss_sigma_factor": 0.2,
        }

        all_args = dict(metric_optional_args, **optimizer_optional_args)
        syn_flow.run(*positional_args, out_dir=out_dir, **all_args)

        warped_path = syn_flow.last_generated_outputs["out_warped"]
        npt.assert_equal(Path(warped_path).is_file(), True)
        warped_map_path = syn_flow.last_generated_outputs["out_field"]
        npt.assert_equal(Path(warped_map_path).is_file(), True)


@pytest.mark.skipif(not have_pd, reason="Requires pandas")
def test_bundlewarp_flow():
    with TemporaryDirectory() as out_dir:
        data_path = get_fnames(name="fornix")

        fornix = load_tractogram(data_path, "same", bbox_valid_check=False).streamlines

        f = Streamlines(fornix)
        f1 = f.copy()

        f1_path = Path(out_dir) / "f1.trk"
        sft = StatefulTractogram(f1, data_path, Space.RASMM)
        save_tractogram(sft, f1_path, bbox_valid_check=False)

        f2 = f1.copy()
        f2._data += np.array([50, 0, 0])

        f2_path = Path(out_dir) / "f2.trk"
        sft = StatefulTractogram(f2, data_path, Space.RASMM)
        save_tractogram(sft, f2_path, bbox_valid_check=False)

        bw_flow = BundleWarpFlow(force=True)
        bw_flow.run(f1_path, f2_path, out_dir=out_dir, bbox_valid_check=False)

        out_linearly_moved = Path(out_dir) / "linearly_moved.trx"
        out_nonlinearly_moved = Path(out_dir) / "nonlinearly_moved.trx"

        assert out_linearly_moved.exists()
        assert out_nonlinearly_moved.exists()
