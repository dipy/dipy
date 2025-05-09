import logging
import os
from os.path import join as pjoin
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.io.peaks import load_pam
from dipy.reconst.shm import descoteaux07_legacy_msg, sph_harm_ind_list
from dipy.sims.voxel import multi_tensor
from dipy.utils.optpkg import optional_package
from dipy.workflows.reconst import (
    ReconstForecastFlow,
    ReconstGQIFlow,
    ReconstRUMBAFlow,
    ReconstSFMFlow,
)

_, has_sklearn, _ = optional_package("sklearn")
logging.getLogger().setLevel(logging.INFO)


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


def test_reconst_rumba():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstRUMBAFlow)


@pytest.mark.skipif(not has_sklearn, reason="Requires sklearn")
def test_reconst_sfm():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstSFMFlow)


def test_reconst_gqi():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstGQIFlow)


def test_reconst_forecast():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstForecastFlow, use_multishell_data=True)


def reconst_flow_core(flow, *, use_multishell_data=None, **kwargs):
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames(name="small_64D")
        if use_multishell_data:
            bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)

            # FW model requires multishell data
            bvals_2s = np.concatenate((bvals, bvals * 1.5), axis=0)
            bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
            gtab_2s = gradient_table(bvals_2s, bvecs=bvecs_2s)
            bval_path = pjoin(out_dir, os.path.basename(bval_path))
            bvec_path = pjoin(out_dir, os.path.basename(bvec_path))
            np.savetxt(bval_path, bvals_2s)
            np.savetxt(bvec_path, bvecs_2s)

            volume = simulate_multitensor_data(gtab_2s)
            data_path = pjoin(out_dir, "dwi.nii.gz")
            mask = np.ones_like(volume[..., 0], dtype=bool)
            mask_path = pjoin(out_dir, "mask.nii.gz")
            save_nifti(mask_path, mask.astype(np.int32), np.eye(4))
            save_nifti(data_path, volume, np.eye(4))
        else:
            volume, affine = load_nifti(data_path)
            mask = np.ones_like(volume[:, :, :, 0])
            mask_path = pjoin(out_dir, "tmp_mask.nii.gz")
            save_nifti(mask_path, mask.astype(np.uint8), affine)

        reconst_flow = flow()
        for sh_order_max in [
            8,
        ]:
            reconst_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                sh_order_max=sh_order_max,
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
                shm_data.shape[-1], sph_harm_ind_list(sh_order_max)[0].shape[0]
            )
            npt.assert_equal(shm_data.shape[:-1], volume.shape[:-1])

            pam = load_pam(reconst_flow.last_generated_outputs["out_pam"])
            npt.assert_allclose(
                pam.peak_dirs.reshape(peaks_dir_data.shape), peaks_dir_data
            )

            npt.assert_allclose(pam.peak_values, peaks_vals_data)
            npt.assert_allclose(pam.peak_indices, peaks_idx_data)
            npt.assert_allclose(pam.shm_coeff, shm_data, atol=1e-7)
            npt.assert_allclose(pam.gfa, gfa_data)
