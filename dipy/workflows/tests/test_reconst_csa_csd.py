import logging
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dipy.core.gradients import generate_bvecs
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.io.peaks import load_pam
from dipy.reconst.mcsd import have_cvxpy
from dipy.reconst.shm import descoteaux07_legacy_msg, sph_harm_ind_list
from dipy.testing import assert_warns
from dipy.workflows.reconst import ReconstCSDFlow, ReconstQBallBaseFlow, ReconstSDTFlow

needs_cvxpy = pytest.mark.skipif(not have_cvxpy, reason="Requires CVXPY")

logging.getLogger().setLevel(logging.INFO)


def test_reconst_csa():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstQBallBaseFlow)


def test_reconst_opdt():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstQBallBaseFlow, method="opdt")


def test_reconst_qball():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstQBallBaseFlow, method="qball")


def test_reconst_csd():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstCSDFlow)


def test_reconst_sdt():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstSDTFlow)


@needs_cvxpy
def test_reconst_csd_msmt():
    """Test MSMT-CSD functionality."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Solution may be inaccurate.*",
            category=UserWarning,
        )
        with TemporaryDirectory() as out_dir:
            data_path, bval_path, bvec_path = get_fnames(name="small_64D")
            volume, affine = load_nifti(data_path)
            mask = np.ones_like(volume[:, :, :, 0])
            mask_path = Path(out_dir) / "tmp_mask.nii.gz"
            save_nifti(mask_path, mask.astype(np.uint8), affine)

            reconst_flow = ReconstCSDFlow()
            reconst_flow._force_overwrite = True

            reconst_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                use_msmt=True,
                iso=3,
                sh_order_max=4,
                out_dir=out_dir,
                extract_pam_values=True,
            )

            gfa_path = reconst_flow.last_generated_outputs["out_gfa"]
            npt.assert_equal(load_nifti_data(gfa_path).shape, volume.shape[:-1])

            for fname in ["wm_mask.nii.gz", "gm_mask.nii.gz", "csf_mask.nii.gz"]:
                mask_path_out = Path(out_dir) / fname
                assert mask_path_out.exists()
                npt.assert_equal(
                    load_nifti_data(str(mask_path_out)).shape, volume.shape[:-1]
                )


@needs_cvxpy
def test_reconst_csd_msmt_with_t1():
    """Test MSMT-CSD with provided T1 image (T1-based HMRF tissue segmentation).

    The anisotropic power map is used as the T1 substitute so that the HMRF
    tissue labels are consistent with the actual DWI signal properties,
    avoiding degenerate response functions.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Solution may be inaccurate.*",
            category=UserWarning,
        )
        with TemporaryDirectory() as out_dir:
            data_path, bval_path, bvec_path = get_fnames(name="small_64D")
            volume, affine = load_nifti(data_path)
            bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
            mask = np.ones_like(volume[:, :, :, 0])
            mask_path = Path(out_dir) / "tmp_mask.nii.gz"
            save_nifti(mask_path, mask.astype(np.uint8), affine)

            # Compute the anisotropic power map from DWI data and save it as a
            # "T1" so HMRF tissue labels are consistent with DWI signal.
            from dipy.core.gradients import gradient_table
            from dipy.core.sphere import HemiSphere
            from dipy.reconst.shm import (
                anisotropic_power,
                normalize_data,
                smooth_pinv,
                sph_harm_lookup,
            )

            gtab = gradient_table(bvals, bvecs=bvecs)
            dwi_mask = ~gtab.b0s_mask
            normed = normalize_data(volume, gtab.b0s_mask)[..., dwi_mask]
            normed = normed * mask[..., None]
            signal_pts = HemiSphere(xyz=gtab.bvecs[dwi_mask])
            Ba, m, n = sph_harm_lookup.get("descoteaux07")(
                4, signal_pts.theta, signal_pts.phi
            )
            invB = smooth_pinv(Ba, np.sqrt(0.0) * (-n * (n + 1)))
            shm = np.dot(normed, invB.T)
            amap = anisotropic_power(
                shm, norm_factor=0.00001, power=2, non_negative=True
            )
            t1_path = Path(out_dir) / "t1.nii.gz"
            save_nifti(str(t1_path), amap.astype(np.float32), affine)

            reconst_flow = ReconstCSDFlow()
            reconst_flow._force_overwrite = True

            reconst_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                use_msmt=True,
                t1_file=str(t1_path),
                iso=3,
                sh_order_max=4,
                out_dir=out_dir,
                extract_pam_values=True,
            )

            gfa_path = reconst_flow.last_generated_outputs["out_gfa"]
            npt.assert_equal(load_nifti_data(gfa_path).shape, volume.shape[:-1])

            for fname in ["wm_mask.nii.gz", "gm_mask.nii.gz", "csf_mask.nii.gz"]:
                mask_path_out = Path(out_dir) / fname
                assert mask_path_out.exists()
                npt.assert_equal(
                    load_nifti_data(str(mask_path_out)).shape, volume.shape[:-1]
                )


def reconst_flow_core(flow, **kwargs):
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames(name="small_64D")
        volume, affine = load_nifti(data_path)
        mask = np.ones_like(volume[:, :, :, 0])
        mask_path = Path(out_dir) / "tmp_mask.nii.gz"
        save_nifti(mask_path, mask.astype(np.uint8), affine)

        reconst_flow = flow()
        for sh_order in [4, 6, 8]:
            reconst_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                sh_order_max=sh_order,
                out_dir=out_dir,
                extract_pam_values=True,
                **kwargs,
            )

            gfa_path = reconst_flow.last_generated_outputs["out_gfa"]
            gfa_data = load_nifti_data(gfa_path)
            npt.assert_equal(gfa_data.shape, volume.shape[:-1])

            peaks_dir_path = reconst_flow.last_generated_outputs["out_peaks_dir"]
            peaks_dir_data = load_nifti_data(peaks_dir_path)
            npt.assert_equal(peaks_dir_data.shape[-1], 15)
            npt.assert_equal(peaks_dir_data.shape[:-1], volume.shape[:-1])

            peaks_idx_path = reconst_flow.last_generated_outputs["out_peaks_indices"]
            peaks_idx_data = load_nifti_data(peaks_idx_path)
            npt.assert_equal(peaks_idx_data.shape[-1], 5)
            npt.assert_equal(peaks_idx_data.shape[:-1], volume.shape[:-1])

            peaks_vals_path = reconst_flow.last_generated_outputs["out_peaks_values"]
            peaks_vals_data = load_nifti_data(peaks_vals_path)
            npt.assert_equal(peaks_vals_data.shape[-1], 5)
            npt.assert_equal(peaks_vals_data.shape[:-1], volume.shape[:-1])

            shm_path = reconst_flow.last_generated_outputs["out_shm"]
            shm_data = load_nifti_data(shm_path)
            # Test that the number of coefficients is what you would expect
            # given the order of the sh basis:
            npt.assert_equal(
                shm_data.shape[-1], sph_harm_ind_list(sh_order)[0].shape[0]
            )
            npt.assert_equal(shm_data.shape[:-1], volume.shape[:-1])

            pam = load_pam(reconst_flow.last_generated_outputs["out_pam"])
            npt.assert_allclose(
                pam.peak_dirs.reshape(peaks_dir_data.shape), peaks_dir_data
            )
            npt.assert_allclose(pam.peak_values, peaks_vals_data)
            npt.assert_allclose(pam.peak_indices, peaks_idx_data)
            npt.assert_allclose(pam.shm_coeff, shm_data)
            npt.assert_allclose(pam.gfa, gfa_data)

            bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
            bvals[0] = 5.0
            bvecs = generate_bvecs(len(bvals))

            tmp_bval_path = Path(out_dir) / "tmp.bval"
            tmp_bvec_path = Path(out_dir) / "tmp.bvec"
            np.savetxt(tmp_bval_path, bvals)
            np.savetxt(tmp_bvec_path, bvecs.T)
            reconst_flow._force_overwrite = True

            if flow.get_short_name() == "csd":
                reconst_flow = flow()
                reconst_flow._force_overwrite = True
                reconst_flow.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    out_dir=out_dir,
                    frf=[15, 5, 5],
                    **kwargs,
                )
                reconst_flow = flow()
                reconst_flow._force_overwrite = True
                reconst_flow.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    out_dir=out_dir,
                    frf="15, 5, 5",
                    **kwargs,
                )
                reconst_flow = flow()
                reconst_flow._force_overwrite = True
                reconst_flow.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    out_dir=out_dir,
                    frf=None,
                    **kwargs,
                )
                reconst_flow2 = flow()
                reconst_flow2._force_overwrite = True
                reconst_flow2.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    out_dir=out_dir,
                    frf=None,
                    roi_center=[5, 5, 5],
                    **kwargs,
                )
            else:
                with npt.assert_raises(BaseException):
                    assert_warns(
                        UserWarning,
                        reconst_flow.run,
                        data_path,
                        tmp_bval_path,
                        tmp_bvec_path,
                        mask_path,
                        out_dir=out_dir,
                        extract_pam_values=True,
                        **kwargs,
                    )

            # test parallel implementation
            # Avoid SDT for now, as it is quite slow, something to introspect
            if flow.get_short_name() != "sdt" and not sys.platform.startswith("win"):
                reconst_flow = flow()
                reconst_flow._force_overwrite = True
                reconst_flow.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    out_dir=out_dir,
                    parallel=True,
                    num_processes=2,
                    **kwargs,
                )


@needs_cvxpy
@pytest.mark.skipif(sys.platform.startswith('win'), reason="Parallel execution hangs on Windows CI")
def test_reconst_csd_msmt_parallel():
    """Test MSMT-CSD with parallel processing."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Solution may be inaccurate.*",
            category=UserWarning,
        )
        with TemporaryDirectory() as out_dir:
            data_path, bval_path, bvec_path = get_fnames(name="small_64D")
            volume, affine = load_nifti(data_path)
            mask = np.ones_like(volume[:, :, :, 0])
            mask_path = Path(out_dir) / "tmp_mask.nii.gz"
            save_nifti(mask_path, mask.astype(np.uint8), affine)

            reconst_flow = ReconstCSDFlow()
            reconst_flow._force_overwrite = True

            reconst_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                use_msmt=True,
                iso=3,
                sh_order_max=4,
                out_dir=out_dir,
                extract_pam_values=True,
                parallel=True,
                num_processes=2,
            )

            gfa_path = reconst_flow.last_generated_outputs["out_gfa"]
            npt.assert_equal(load_nifti_data(gfa_path).shape, volume.shape[:-1])


def test_reconst_csd_msmt_invalid_iso():
    """Test that iso < 3 raises an error."""
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames(name="small_64D")
        volume, affine = load_nifti(data_path)
        mask = np.ones_like(volume[:, :, :, 0])
        mask_path = Path(out_dir) / "tmp_mask.nii.gz"
        save_nifti(mask_path, mask.astype(np.uint8), affine)

        reconst_flow = ReconstCSDFlow()
        with npt.assert_raises(SystemExit):
            reconst_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                use_msmt=True,
                iso=2,
                sh_order_max=4,
                out_dir=out_dir,
            )


@needs_cvxpy
def test_reconst_csd_msmt_with_tissue_masks():
    """Test MSMT-CSD with pre-provided tissue masks (no HMRF classification).

    Tissue masks are derived from the DWI anisotropic power map via HMRF so
    that the tissue labels are consistent with the DWI signal properties,
    avoiding degenerate response functions.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Solution may be inaccurate.*",
            category=UserWarning,
        )
        with TemporaryDirectory() as out_dir:
            data_path, bval_path, bvec_path = get_fnames(name="small_64D")
            volume, affine = load_nifti(data_path)
            bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
            shape = volume.shape[:3]

            # Derive tissue masks from the DWI-based anisotropic power map so
            # they are consistent with the actual diffusion signal.
            from dipy.core.gradients import gradient_table
            from dipy.core.sphere import HemiSphere
            from dipy.reconst.shm import (
                anisotropic_power,
                normalize_data,
                smooth_pinv,
                sph_harm_lookup,
            )
            from dipy.segment.tissue import TissueClassifierHMRF

            gtab = gradient_table(bvals, bvecs=bvecs)
            dwi_mask_b = ~gtab.b0s_mask
            normed = normalize_data(volume, gtab.b0s_mask)[..., dwi_mask_b]
            signal_pts = HemiSphere(xyz=gtab.bvecs[dwi_mask_b])
            Ba, m, n = sph_harm_lookup.get("descoteaux07")(
                4, signal_pts.theta, signal_pts.phi
            )
            invB = smooth_pinv(Ba, np.sqrt(0.0) * (-n * (n + 1)))
            amap = anisotropic_power(
                np.dot(normed, invB.T), norm_factor=0.00001, power=2, non_negative=True
            )
            hmrf = TissueClassifierHMRF()
            _, seg, _ = hmrf.classify(amap, 3, 0.1)
            wm_mask = np.where(seg == 3, 1, 0).astype(np.uint8)
            gm_mask = np.where(seg == 2, 1, 0).astype(np.uint8)
            csf_mask = np.where(seg == 1, 1, 0).astype(np.uint8)

            mask_path = Path(out_dir) / "tmp_mask.nii.gz"
            wm_path = Path(out_dir) / "wm_in.nii.gz"
            gm_path = Path(out_dir) / "gm_in.nii.gz"
            csf_path = Path(out_dir) / "csf_in.nii.gz"
            save_nifti(mask_path, np.ones(shape, dtype=np.uint8), affine)
            save_nifti(wm_path, wm_mask, affine)
            save_nifti(gm_path, gm_mask, affine)
            save_nifti(csf_path, csf_mask, affine)

            reconst_flow = ReconstCSDFlow()
            reconst_flow._force_overwrite = True

            reconst_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                use_msmt=True,
                wm_file=str(wm_path),
                gm_file=str(gm_path),
                csf_file=str(csf_path),
                iso=3,
                sh_order_max=4,
                out_dir=out_dir,
                extract_pam_values=True,
            )

            gfa_path = reconst_flow.last_generated_outputs["out_gfa"]
            npt.assert_equal(load_nifti_data(gfa_path).shape, volume.shape[:-1])
