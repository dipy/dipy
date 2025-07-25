from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import numpy as np
from numpy.testing import assert_equal

from dipy.core.gradients import generate_bvecs, gradient_table
from dipy.io.image import load_nifti_data, save_nifti
from dipy.sims.voxel import multi_tensor
from dipy.workflows.reconst import ReconstIvimFlow


def test_reconst_ivim():
    with TemporaryDirectory() as out_dir:
        bvals = np.array(
            [
                0.0,
                10.0,
                20.0,
                30.0,
                40.0,
                60.0,
                80.0,
                100.0,
                120.0,
                140.0,
                160.0,
                180.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
                900.0,
                1000.0,
            ]
        )
        N = len(bvals)
        bvecs = generate_bvecs(N)
        temp_bval_path = Path(out_dir) / "temp.bval"
        np.savetxt(temp_bval_path, bvals)
        temp_bvec_path = Path(out_dir) / "temp.bvec"
        np.savetxt(temp_bvec_path, bvecs)

        gtab = gradient_table(bvals, bvecs=bvecs)

        S0, f, D_star, D = 1000.0, 0.132, 0.00885, 0.000921

        mevals = np.array(([D_star, D_star, D_star], [D, D, D]))
        # This gives an isotropic signal.
        data = multi_tensor(
            gtab, mevals, snr=None, S0=S0, fractions=[f * 100, 100 * (1 - f)]
        )
        # Single voxel data
        data_single = data[0]
        temp_affine = np.eye(4)

        data_multi = np.zeros((2, 2, 1, len(gtab.bvals)), dtype=int)
        data_multi[0, 0, 0] = data_multi[0, 1, 0] = data_multi[1, 0, 0] = data_multi[
            1, 1, 0
        ] = data_single
        data_path = Path(out_dir) / "tmp_data.nii.gz"
        save_nifti(data_path, data_multi, temp_affine, dtype=data_multi.dtype)

        mask = np.ones_like(data_multi[..., 0], dtype=np.uint8)
        mask_path = Path(out_dir) / "tmp_mask.nii.gz"
        save_nifti(mask_path, mask, temp_affine)

        ivim_flow = ReconstIvimFlow()

        args = [
            data_path,
            temp_bval_path,
            temp_bvec_path,
            mask_path,
        ]

        msg = "Bounds for this fit have been set from experiments and * "
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=msg, category=UserWarning)
            ivim_flow.run(*args, out_dir=out_dir)

        S0_path = ivim_flow.last_generated_outputs["out_S0_predicted"]
        S0_data = load_nifti_data(S0_path)
        assert_equal(S0_data.shape, data_multi.shape[:-1])

        f_path = ivim_flow.last_generated_outputs["out_perfusion_fraction"]
        f_data = load_nifti_data(f_path)
        assert_equal(f_data.shape, data_multi.shape[:-1])

        D_star_path = ivim_flow.last_generated_outputs["out_D_star"]
        D_star_data = load_nifti_data(D_star_path)
        assert_equal(D_star_data.shape, data_multi.shape[:-1])

        D_path = ivim_flow.last_generated_outputs["out_D"]
        D_data = load_nifti_data(D_path)
        assert_equal(D_data.shape, data_multi.shape[:-1])
