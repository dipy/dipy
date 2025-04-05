from os.path import join as pjoin
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import numpy.testing as npt

from dipy.align.streamlinear import BundleMinDistanceMetric
from dipy.data import get_fnames
from dipy.io.image import load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.segment.mask import median_otsu
from dipy.segment.tests.test_mrf import create_image
from dipy.tracking.streamline import Streamlines, set_number_of_points
from dipy.utils.optpkg import optional_package
from dipy.workflows.segment import (
    ClassifyTissueFlow,
    LabelsBundlesFlow,
    MedianOtsuFlow,
    RecoBundlesFlow,
)

sklearn, has_sklearn, _ = optional_package("sklearn")


def test_median_otsu_flow():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames(name="small_25")
        volume = load_nifti_data(data_path)
        save_masked = True
        median_radius = 3
        numpass = 3
        autocrop = False
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
            autocrop=autocrop,
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
            autocrop=autocrop,
            dilate=dilate,
        )

        result_mask_data = load_nifti_data(pjoin(out_dir, mask_name))
        npt.assert_array_equal(result_mask_data.astype(np.uint8), mask)

        result_masked = nib.load(pjoin(out_dir, masked_name))
        result_masked_data = np.asanyarray(result_masked.dataobj)

        npt.assert_array_equal(np.round(result_masked_data), masked)


def test_recobundles_flow():
    with TemporaryDirectory() as out_dir:
        data_path = get_fnames(name="fornix")

        fornix = load_tractogram(data_path, "same", bbox_valid_check=False).streamlines

        f = Streamlines(fornix)
        f1 = f.copy()

        f2 = f1[:15].copy()
        f2._data += np.array([40, 0, 0])

        f.extend(f2)

        f2_path = pjoin(out_dir, "f2.trk")
        sft = StatefulTractogram(f2, data_path, Space.RASMM)
        save_tractogram(sft, f2_path, bbox_valid_check=False)

        f1_path = pjoin(out_dir, "f1.trk")
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


def test_classify_tissue_flow():
    with TemporaryDirectory() as out_dir:
        data = create_image()
        data_path = pjoin(out_dir, "data.nii.gz")
        nib.save(nib.Nifti1Image(data, np.eye(4)), data_path)

        args = {
            "input_files": data_path,
            "method": "hmrf",
            "nclass": 4,
            "beta": 0.1,
            "tolerance": 0.0001,
            "max_iter": 10,
            "out_dir": out_dir,
        }
        flow = ClassifyTissueFlow()
        flow.run(**args)

        tissue = flow.last_generated_outputs["out_tissue"]
        pve = flow.last_generated_outputs["out_pve"]

        tissue_data = load_nifti_data(tissue)
        pve_data = load_nifti_data(pve)

        npt.assert_equal(tissue_data.shape, data.shape)
        npt.assert_equal(tissue_data.max(), 4)
        npt.assert_equal(tissue_data.min(), 0)

        npt.assert_equal(pve_data.shape, (data.shape) + (4,))
        npt.assert_equal(pve_data.max(), 1)

        npt.assert_raises(SystemExit, flow.run, data_path)
        npt.assert_raises(SystemExit, flow.run, data_path, method="random")
        npt.assert_raises(SystemExit, flow.run, data_path, method="dam")
        npt.assert_raises(SystemExit, flow.run, data_path, method="hmrf")

    if has_sklearn:
        with TemporaryDirectory() as out_dir:
            data = np.random.rand(3, 3, 3, 7) * 100  # Simulated random data
            bvals = np.array([0, 100, 500, 1000, 1500, 2000, 3000])
            data_path = pjoin(out_dir, "data.nii.gz")
            bvals_path = pjoin(out_dir, "bvals")
            np.savetxt(bvals_path, bvals)
            nib.save(nib.Nifti1Image(data, np.eye(4)), data_path)

            args = {
                "input_files": data_path,
                "bvals_file": bvals_path,
                "method": "dam",
                "wm_threshold": 0.5,
                "out_dir": out_dir,
            }
            flow = ClassifyTissueFlow()
            flow.run(**args)

            tissue = flow.last_generated_outputs["out_tissue"]
            pve = flow.last_generated_outputs["out_pve"]

            tissue_data = load_nifti_data(tissue)
            pve_data = load_nifti_data(pve)

            npt.assert_equal(tissue_data.shape, data.shape[:-1])
            npt.assert_equal(tissue_data.max(), 2)
            npt.assert_equal(tissue_data.min(), 0)

            npt.assert_equal(pve_data.shape, (data.shape[:-1]) + (2,))
            npt.assert_equal(pve_data.max(), 1)
