import os
from os.path import join as pjoin
from tempfile import TemporaryDirectory
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.io.peaks import load_pam
from dipy.reconst.shm import descoteaux07_legacy_msg
from dipy.sims.voxel import multi_tensor
from dipy.testing import assert_greater
from dipy.workflows.reconst import ReconstDtiFlow, ReconstFwdtiFlow


def test_reconst_dti_wls():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstDtiFlow)


def test_reconst_dti_nlls():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstDtiFlow, extra_args=[], extra_kwargs={})


def test_reconst_dti_alt_tensor():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(
            ReconstDtiFlow, extra_args=[], extra_kwargs={"nifti_tensor": False}
        )


def reconst_flow_core(flow, *, dataname=None, extra_args=None, extra_kwargs=None):
    extra_args = extra_args or []
    extra_kwargs = extra_kwargs or {}
    dataname = dataname or "small_25"

    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames(name=dataname)
        volume, affine = load_nifti(data_path)
        mask = np.ones_like(volume[:, :, :, 0], dtype=np.uint8)
        mask_path = pjoin(out_dir, "tmp_mask.nii.gz")
        save_nifti(mask_path, mask, affine)

        dti_flow = flow()

        args = [data_path, bval_path, bvec_path, mask_path]
        args.extend(extra_args)
        kwargs = {"out_dir": out_dir, "extract_pam_values": True}
        kwargs.update(extra_kwargs)

        dti_flow.run(*args, **kwargs)

        fa_path = dti_flow.last_generated_outputs["out_fa"]
        fa_data = load_nifti_data(fa_path)
        assert_equal(fa_data.shape, volume.shape[:-1])

        tensor_path = dti_flow.last_generated_outputs["out_tensor"]
        tensor_data = load_nifti_data(tensor_path)
        # Per default, tensor data is 5D, with six tensor elements on the last
        # dimension, except if nifti_tensor is set to False:
        if extra_kwargs.get("nifti_tensor", True):
            assert_equal(tensor_data.shape[-1], 6)
            assert_equal(tensor_data.shape[:-2], volume.shape[:-1])
        else:
            assert_equal(tensor_data.shape[-1], 6)
            assert_equal(tensor_data.shape[:-1], volume.shape[:-1])

        for out_name in ["out_ga", "out_md", "out_ad", "out_rd", "out_mode", "out_s0"]:
            out_path = dti_flow.last_generated_outputs[out_name]
            out_data = load_nifti_data(out_path)
            assert_equal(out_data.shape, volume.shape[:-1])

        rgb_path = dti_flow.last_generated_outputs["out_rgb"]
        rgb_data = load_nifti_data(rgb_path)
        assert_equal(rgb_data.shape[-1], 3)
        assert_equal(rgb_data.shape[:-1], volume.shape[:-1])

        evecs_path = dti_flow.last_generated_outputs["out_evec"]
        evecs_data = load_nifti_data(evecs_path)
        assert_equal(evecs_data.shape[-2:], (3, 3))
        assert_equal(evecs_data.shape[:-2], volume.shape[:-1])

        evals_path = dti_flow.last_generated_outputs["out_eval"]
        evals_data = load_nifti_data(evals_path)
        assert_equal(evals_data.shape[-1], 3)
        assert_equal(evals_data.shape[:-1], volume.shape[:-1])

        peaks_dir_path = dti_flow.last_generated_outputs["out_peaks_dir"]
        peaks_dir_data = load_nifti_data(peaks_dir_path)
        assert_equal(peaks_dir_data.shape[-1], 3)
        assert_equal(peaks_dir_data.shape[:-1], volume.shape[:-1])

        peaks_idx_path = dti_flow.last_generated_outputs["out_peaks_indices"]
        peaks_idx_data = load_nifti_data(peaks_idx_path)
        assert_equal(peaks_idx_data.shape[-1], 1)
        assert_equal(peaks_idx_data.shape[:-1], volume.shape[:-1])

        peaks_vals_path = dti_flow.last_generated_outputs["out_peaks_values"]
        peaks_vals_data = load_nifti_data(peaks_vals_path)
        assert_equal(peaks_vals_data.shape[-1], 1)
        assert_equal(peaks_vals_data.shape[:-1], volume.shape[:-1])

        pam = load_pam(dti_flow.last_generated_outputs["out_pam"])
        assert_allclose(pam.peak_dirs.reshape(peaks_dir_data.shape), peaks_dir_data)
        assert_allclose(pam.peak_values, peaks_vals_data)
        assert_allclose(pam.peak_indices, peaks_idx_data)


def simulate_multitensor_data(gtab):
    dwi = np.zeros((2, 2, 2, len(gtab.bvals)))
    # Diffusion of tissue and water compartments are constant for all voxel
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    # volume fractions
    GTF = np.array([[[0.06, 0.71], [0.33, 0.91]], [[0.0, 0.0], [0.0, 0.0]]])
    for i in range(2):
        for j in range(2):
            gtf = GTF[0, i, j]
            S, p = multi_tensor(
                gtab,
                mevals,
                S0=100,
                angles=[(90, 0), (90, 0)],
                fractions=[(1 - gtf) * 100, gtf * 100],
                snr=None,
            )
            dwi[0, i, j] = S
    return dwi


def test_fwdti_error():
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames(name="small_25")
        volume, affine = load_nifti(data_path)
        mask = np.ones_like(volume[:, :, :, 0], dtype=np.uint8)
        mask_path = pjoin(out_dir, "tmp_mask.nii.gz")
        save_nifti(mask_path, mask, affine)

        flow = ReconstFwdtiFlow()

        assert_raises(ValueError, flow.run, data_path, bval_path, bvec_path, mask_path)
        assert_raises(
            ValueError,
            flow.run,
            data_path,
            bval_path,
            bvec_path,
            mask_path,
            fit_method="OLS",
        )
        assert_raises(TypeError, flow.run)


def test_fwdti():
    with TemporaryDirectory() as out_dir:
        _, fbvals, fbvecs = get_fnames(name="small_64D")
        bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)

        # FW model requires multishell data
        bvals_2s = np.concatenate((bvals, bvals * 1.5), axis=0)
        bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
        gtab_2s = gradient_table(bvals_2s, bvecs=bvecs_2s)
        fbvals = pjoin(out_dir, os.path.basename(fbvals))
        fbvecs = pjoin(out_dir, os.path.basename(fbvecs))
        np.savetxt(fbvals, bvals_2s)
        np.savetxt(fbvecs, bvecs_2s)

        dwi = simulate_multitensor_data(gtab_2s)
        fdwi = pjoin(out_dir, "dwi.nii.gz")
        mask = np.ones_like(dwi[..., 0], dtype=bool)
        mask_path = pjoin(out_dir, "mask.nii.gz")
        # import ipdb; ipdb.set_trace()
        save_nifti(mask_path, mask.astype(np.int32), np.eye(4))
        save_nifti(fdwi, dwi, np.eye(4))

        fwdti_flow = ReconstFwdtiFlow()

        args = [fdwi, fbvals, fbvecs, mask_path]
        kwargs = {"out_dir": out_dir, "extract_pam_values": True}

        fwdti_flow.run(*args, **kwargs)

        for out_name in [
            "out_fa",
            "out_tensor",
            "out_ga",
            "out_md",
            "out_ad",
            "out_rd",
            "out_mode",
        ]:
            out_path = fwdti_flow.last_generated_outputs[out_name]
            out_data = load_nifti_data(out_path)
            assert_greater(out_data.mean(), 0)
