from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from dipy.align.streamlinear import BundleMinDistanceMetric
from dipy.data import get_fnames
from dipy.io.image import load_nifti_data, save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.segment.mask import median_otsu
from dipy.testing import assert_warns
from dipy.testing.decorators import set_random_number_generator
from dipy.tracking.streamline import Streamlines, set_number_of_points
from dipy.utils.deprecator import ArgsDeprecationWarning
from dipy.utils.optpkg import optional_package
from dipy.workflows.segment import (
    BrainMaskFlow,
    ClassifyTissueFlow,
    ClusterStreamlinesFlow,
    LabelsBundlesFlow,
    MedianOtsuFlow,
    RecoBundlesFlow,
)

sklearn, has_sklearn, _ = optional_package("sklearn")
torch, has_torch, _ = optional_package("torch")


def test_median_otsu_flow():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames(name="small_25")
        volume = load_nifti_data(data_path)
        save_masked = True
        median_radius = 3
        numpass = 3
        vol_idx = "0,"
        dilate = 0
        finalize_mask = False

        mo_flow = MedianOtsuFlow()
        mo_flow.run(
            data_path,
            out_dir=out_dir,
            save_masked=save_masked,
            median_radius=median_radius,
            numpass=numpass,
            vol_idx=vol_idx,
            dilate=dilate,
            finalize_mask=finalize_mask,
        )

        mask_name = mo_flow.last_generated_outputs["out_mask"]
        masked_name = mo_flow.last_generated_outputs["out_masked"]

        vol_idx = [
            0,
        ]
        masked, mask = median_otsu(
            volume,
            vol_idx=vol_idx,
            median_radius=median_radius,
            numpass=numpass,
            dilate=dilate,
        )

        result_mask_data = load_nifti_data(Path(out_dir) / mask_name)
        npt.assert_array_equal(result_mask_data.astype(np.uint8), mask)

        result_masked = nib.load(Path(out_dir) / masked_name)
        result_masked_data = np.asanyarray(result_masked.dataobj)

        npt.assert_array_equal(np.round(result_masked_data), masked)


def test_median_otsu_flow_autocrop_deprecated():
    """Test that autocrop parameter triggers deprecation warning."""
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames(name="small_25")

        mo_flow = MedianOtsuFlow()

        with assert_warns(ArgsDeprecationWarning):
            mo_flow.run(
                data_path,
                out_dir=out_dir,
                vol_idx="0,",
                autocrop=True,
            )


def test_median_otsu_flow_with_bvalues():
    """Test MedianOtsuFlow with bvalues_files parameter."""
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, _ = get_fnames(name="small_25")
        volume = load_nifti_data(data_path)

        mo_flow = MedianOtsuFlow()
        for arg_bval in [{"bvalues_files": bval_path}, {"bvalues_files": [bval_path]}]:
            mo_flow._force_overwrite = True
            mo_flow.run(
                data_path,
                **arg_bval,
                b0_threshold=50,
                out_dir=out_dir,
                save_masked=True,
            )

            mask_name = mo_flow.last_generated_outputs["out_mask"]
            masked_name = mo_flow.last_generated_outputs["out_masked"]

            npt.assert_equal((Path(out_dir) / mask_name).exists(), True)
            npt.assert_equal((Path(out_dir) / masked_name).exists(), True)

            result_mask_data = load_nifti_data(Path(out_dir) / mask_name)
            npt.assert_equal(result_mask_data.shape, volume.shape[:3])


def test_recobundles_flow():
    with TemporaryDirectory() as out_dir:
        data_path = get_fnames(name="fornix")

        fornix = load_tractogram(data_path, "same", bbox_valid_check=False).streamlines

        f = Streamlines(fornix)
        f1 = f.copy()

        f2 = f1[:15].copy()
        f2._data += np.array([40, 0, 0])

        f.extend(f2)

        f2_path = Path(out_dir) / "f2.trk"
        sft = StatefulTractogram(f2, data_path, Space.RASMM)
        save_tractogram(sft, f2_path, bbox_valid_check=False)

        f1_path = Path(out_dir) / "f1.trk"
        sft = StatefulTractogram(f, data_path, Space.RASMM)
        save_tractogram(sft, f1_path, bbox_valid_check=False)

        rb_flow = RecoBundlesFlow(force=True)
        rb_flow.run(
            f1_path,
            f2_path,
            greater_than=0,
            clust_thr=10,
            model_clust_thr=5.0,
            reduction_thr=10,
            out_dir=out_dir,
        )

        labels = rb_flow.last_generated_outputs["out_recognized_labels"]
        recog_trk = rb_flow.last_generated_outputs["out_recognized_transf"]

        rec_bundle = load_tractogram(
            recog_trk, "same", bbox_valid_check=False
        ).streamlines
        npt.assert_equal(len(rec_bundle) == len(f2), True)

        label_flow = LabelsBundlesFlow(force=True)
        label_flow.run(f1_path, labels, out_dir=out_dir)

        recog_bundle = label_flow.last_generated_outputs["out_bundle"]
        rec_bundle_org = load_tractogram(
            recog_bundle, "same", bbox_valid_check=False
        ).streamlines

        BMD = BundleMinDistanceMetric()
        nb_pts = 20
        static = set_number_of_points(f2, nb_points=nb_pts)
        moving = set_number_of_points(rec_bundle_org, nb_points=nb_pts)

        BMD.setup(static, moving)
        x0 = np.array([0, 0, 0, 0, 0, 0, 1.0, 1.0, 1, 0, 0, 0])  # affine
        bmd_value = BMD.distance(x0.tolist())

        npt.assert_equal(bmd_value < 1, True)


@set_random_number_generator()
def test_classify_tissue_flow_hmrf(rng=None):
    with TemporaryDirectory() as out_dir:
        # Small structured volume (30x30x5 = 4500 voxels, ~70x smaller than
        # the original 256x256x5). Three Gaussian populations with non-zero
        # variance so the HMRF E-step statistics are well-conditioned.
        data = np.zeros((30, 30, 5), dtype=np.float64)
        data[:10, :, :] = 50.0
        data[10:20, :, :] = 150.0
        data[20:, :, :] = 250.0
        data += rng.normal(0, 10.0, data.shape)
        data = np.clip(data, 0, None)
        data_path = Path(out_dir) / "data.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), data_path)

        flow = ClassifyTissueFlow()
        flow.run(
            input_files=data_path,
            method="hmrf",
            nclass=3,
            beta=0.1,
            tolerance=0,
            max_iter=3,
            out_dir=out_dir,
        )

        tissue = flow.last_generated_outputs["out_tissue"]
        pve = flow.last_generated_outputs["out_pve"]

        tissue_data = load_nifti_data(tissue)
        pve_data = load_nifti_data(pve)

        npt.assert_equal(tissue_data.shape, data.shape)
        npt.assert_equal(tissue_data.max(), 3)
        npt.assert_equal(tissue_data.min() >= 0, True)
        npt.assert_equal(pve_data.shape, data.shape + (3,))
        npt.assert_equal(pve_data.max(), 1)

        npt.assert_raises(SystemExit, flow.run, data_path)
        npt.assert_raises(SystemExit, flow.run, data_path, method="random")
        npt.assert_raises(SystemExit, flow.run, data_path, method="dam")
        npt.assert_raises(SystemExit, flow.run, data_path, method="hmrf")


@pytest.mark.skipif(not has_sklearn, reason="Requires scikit-learn")
@set_random_number_generator()
def test_classify_tissue_flow_dam(rng=None):
    with TemporaryDirectory() as out_dir:
        data = rng.uniform(low=0.0, high=100.0, size=(3, 3, 3, 7))
        bvals = np.array([0, 100, 500, 1000, 1500, 2000, 3000])
        data_path = Path(out_dir) / "data.nii.gz"
        bvals_path = Path(out_dir) / "bvals"
        np.savetxt(bvals_path, bvals)
        nib.save(nib.Nifti1Image(data, np.eye(4)), data_path)

        flow = ClassifyTissueFlow()
        flow.run(
            input_files=data_path,
            bvals_file=bvals_path,
            method="dam",
            wm_threshold=0.5,
            out_dir=out_dir,
        )

        tissue = flow.last_generated_outputs["out_tissue"]
        pve = flow.last_generated_outputs["out_pve"]

        tissue_data = load_nifti_data(tissue)
        pve_data = load_nifti_data(pve)

        npt.assert_equal(tissue_data.shape, data.shape[:-1])
        npt.assert_equal(tissue_data.max(), 2)
        npt.assert_equal(tissue_data.min(), 0)
        npt.assert_equal(pve_data.shape, data.shape[:-1] + (2,))
        npt.assert_equal(pve_data.max(), 1)


@pytest.mark.skipif(not has_torch, reason="Requires PyTorch")
@set_random_number_generator()
def test_classify_tissue_flow_synthseg(rng=None):
    with TemporaryDirectory() as out_dir:
        data = rng.uniform(low=0.0, high=1.0, size=(96, 96, 96))
        data_path = Path(out_dir) / "data.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4) * 2), data_path)

        torch.set_num_threads(1)
        flow = ClassifyTissueFlow()
        flow.run(input_files=data_path, method="synthseg", out_dir=out_dir)

        tissue = flow.last_generated_outputs["out_tissue"]
        tissue_data = load_nifti_data(tissue)

        npt.assert_equal(tissue_data.shape, data.shape)
        npt.assert_equal(tissue_data.min(), 0)
        npt.assert_equal(tissue_data.max() < 60, True)


@set_random_number_generator()
def test_brain_mask_flow(rng=None):
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames(name="small_25")
        flow = BrainMaskFlow()
        npt.assert_raises(SystemExit, flow.run, data_path, method="invalid")

    with TemporaryDirectory() as out_dir:
        data_path, bval_path, _ = get_fnames(name="small_25")
        volume = load_nifti_data(data_path)
        median_radius = 2
        numpass = 5

        flow = BrainMaskFlow()
        flow.run(
            data_path,
            method="median_otsu",
            vol_idx="0,",
            median_radius=median_radius,
            numpass=numpass,
            save_masked=True,
            out_dir=out_dir,
        )

        mask_name = flow.last_generated_outputs["out_mask"]
        masked_name = flow.last_generated_outputs["out_masked"]

        _, expected_mask = median_otsu(
            volume,
            vol_idx=[0],
            median_radius=median_radius,
            numpass=numpass,
        )
        result_mask_data = load_nifti_data(Path(out_dir) / mask_name)
        npt.assert_array_equal(result_mask_data, expected_mask.astype(np.float64))
        npt.assert_equal((Path(out_dir) / masked_name).exists(), True)

    with TemporaryDirectory() as out_dir:
        data_path, bval_path, _ = get_fnames(name="small_25")
        volume = load_nifti_data(data_path)

        flow = BrainMaskFlow()
        flow.run(
            data_path,
            method="median_otsu",
            bvalues_files=bval_path,
            out_dir=out_dir,
        )

        mask_name = flow.last_generated_outputs["out_mask"]
        result_mask_data = load_nifti_data(Path(out_dir) / mask_name)
        npt.assert_equal(result_mask_data.shape, volume.shape[:3])

    if has_torch:
        with TemporaryDirectory() as out_dir:
            file_path = get_fnames(name="evac_test_data")
            volume = np.load(file_path)["input"][0]
            temp_path = Path(out_dir) / "temp.nii.gz"
            save_nifti(temp_path, volume, np.eye(4))

            flow = BrainMaskFlow()
            flow.run(temp_path, method="evac", out_dir=out_dir)

            mask_name = flow.last_generated_outputs["out_mask"]
            mask_data = load_nifti_data(Path(out_dir) / mask_name)

            npt.assert_equal(mask_data.shape, volume.shape)
            npt.assert_equal(mask_data.min() >= 0, True)
            npt.assert_equal(mask_data.max() <= 1, True)

    if has_torch:
        with TemporaryDirectory() as out_dir:
            data = rng.uniform(low=0.0, high=1.0, size=(96, 96, 96))
            data_path = Path(out_dir) / "data.nii.gz"
            nib.save(nib.Nifti1Image(data, np.eye(4) * 2), data_path)

            flow = BrainMaskFlow()
            flow.run(data_path, method="synthseg", out_dir=out_dir)

            mask_name = flow.last_generated_outputs["out_mask"]
            mask_data = load_nifti_data(Path(out_dir) / mask_name)

            npt.assert_equal(mask_data.shape, data.shape)
            npt.assert_equal(mask_data.min() >= 0, True)
            npt.assert_equal(mask_data.max() <= 1, True)


def test_cluster_streamlines_flow():
    with TemporaryDirectory() as out_dir:
        data_path = get_fnames(name="fornix")
        sft = load_tractogram(data_path, "same", bbox_valid_check=False)
        n_streamlines = len(sft.streamlines)

        flow = ClusterStreamlinesFlow()
        npt.assert_raises(SystemExit, flow.run, data_path, method="invalid")

        flow = ClusterStreamlinesFlow(force=True)
        flow.run(data_path, method="quickbundles", threshold=10.0, out_dir=out_dir)
        centroids_sft = load_tractogram(
            Path(out_dir) / flow.last_generated_outputs["out_centroids"],
            "same",
            bbox_valid_check=False,
        )
        labels = np.load(
            Path(out_dir) / flow.last_generated_outputs["out_cluster_labels"]
        )
        npt.assert_equal(len(centroids_sft.streamlines) > 0, True)
        npt.assert_equal(labels.shape[0], n_streamlines)

        flow._force_overwrite = True
        flow.run(
            data_path,
            method="qbx_and_merge",
            thresholds="30,20,10",
            out_dir=out_dir,
        )
        centroids_sft = load_tractogram(
            Path(out_dir) / flow.last_generated_outputs["out_centroids"],
            "same",
            bbox_valid_check=False,
        )
        labels = np.load(
            Path(out_dir) / flow.last_generated_outputs["out_cluster_labels"]
        )
        npt.assert_equal(len(centroids_sft.streamlines) > 0, True)
        npt.assert_equal(labels.shape[0], n_streamlines)

        flow._force_overwrite = True
        flow.run(
            data_path,
            method="faststreamlines",
            threshold=10.0,
            max_radius=15.0,
            out_dir=out_dir,
        )
        centroids_sft = load_tractogram(
            Path(out_dir) / flow.last_generated_outputs["out_centroids"],
            "same",
            bbox_valid_check=False,
        )
        labels = np.load(
            Path(out_dir) / flow.last_generated_outputs["out_cluster_labels"]
        )
        npt.assert_equal(len(centroids_sft.streamlines) > 0, True)
        npt.assert_equal(labels.shape[0], n_streamlines)
