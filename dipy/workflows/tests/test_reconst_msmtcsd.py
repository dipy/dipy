from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_3shell_gtab
from dipy.io.image import load_nifti_data, save_nifti
from dipy.reconst.shm import sph_harm_ind_list
from dipy.sims.voxel import single_tensor
from dipy.utils.optpkg import optional_package
from dipy.workflows.reconst import ReconstMSMTCSDFlow

cvx, have_cvxpy, _ = optional_package("cvxpy", min_version="1.4.1")
needs_cvxpy = pytest.mark.skipif(not have_cvxpy, reason="Requires CVXPY")


@needs_cvxpy
def test_reconst_msmtcsd_smoke():
    gtab = get_3shell_gtab()

    wm_evals = np.array([1.7e-3, 0.4e-3, 0.4e-3])
    wm_signal = single_tensor(gtab, 25.0, evals=wm_evals)

    data = np.zeros((2, 2, 1, len(wm_signal)), dtype=np.float32)
    data[...] = wm_signal
    affine = np.eye(4)

    mask = np.ones(data.shape[:3], dtype=np.uint8)

    # Response array shape: (3, n_shells, 4) where n_shells excludes b0.
    # This is the same constant response used in dipy.reconst.tests.test_mcsd.
    response_wm = np.array(
        [
            [1.7e-3, 0.4e-3, 0.4e-3, 25.0],
            [1.7e-3, 0.4e-3, 0.4e-3, 25.0],
            [1.7e-3, 0.4e-3, 0.4e-3, 25.0],
        ]
    )
    response_gm = np.array(
        [
            [4.0e-4, 4.0e-4, 4.0e-4, 40.0],
            [4.0e-4, 4.0e-4, 4.0e-4, 40.0],
            [4.0e-4, 4.0e-4, 4.0e-4, 40.0],
        ]
    )
    response_csf = np.array(
        [
            [3.0e-3, 3.0e-3, 3.0e-3, 100.0],
            [3.0e-3, 3.0e-3, 3.0e-3, 100.0],
            [3.0e-3, 3.0e-3, 3.0e-3, 100.0],
        ]
    )
    response = np.array([response_wm, response_gm, response_csf])

    with TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)

        dwi_path = out_dir / "dwi.nii.gz"
        mask_path = out_dir / "mask.nii.gz"
        bval_path = out_dir / "dwi.bval"
        bvec_path = out_dir / "dwi.bvec"

        save_nifti(dwi_path, data, affine)
        save_nifti(mask_path, mask, affine)
        np.savetxt(bval_path, gtab.bvals)
        np.savetxt(bvec_path, gtab.bvecs.T)

        flow = ReconstMSMTCSDFlow()
        flow._force_overwrite = True
        flow.run(
            str(dwi_path),
            str(bval_path),
            str(bvec_path),
            str(mask_path),
            response=response,
            iso=2,
            sh_order_max=8,
            out_dir=str(out_dir),
            extract_pam_values=True,
            engine="serial",
        )

        gfa_path = flow.last_generated_outputs["out_gfa"]
        gfa_data = load_nifti_data(gfa_path)
        npt.assert_equal(gfa_data.shape, data.shape[:-1])

        peaks_dir_path = flow.last_generated_outputs["out_peaks_dir"]
        peaks_dir_data = load_nifti_data(peaks_dir_path)
        npt.assert_equal(peaks_dir_data.shape[:-1], data.shape[:-1])
        npt.assert_equal(peaks_dir_data.shape[-1], 15)

        shm_path = flow.last_generated_outputs["out_shm"]
        shm_data = load_nifti_data(shm_path)
        npt.assert_equal(shm_data.shape[:-1], data.shape[:-1])
        npt.assert_equal(shm_data.shape[-1], sph_harm_ind_list(8)[0].shape[0])
