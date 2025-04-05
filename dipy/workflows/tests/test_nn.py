from os.path import join as pjoin
from tempfile import TemporaryDirectory

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.io.image import load_nifti_data, save_nifti
from dipy.nn.evac import EVACPlus
from dipy.utils.optpkg import optional_package
from dipy.workflows.nn import BiasFieldCorrectionFlow, EVACPlusFlow

tf, have_tf, _ = optional_package("tensorflow", min_version="2.18.0")
torch, have_torch, _ = optional_package("torch", min_version="2.2.0")
have_nn = have_tf or have_torch


@pytest.mark.skipif(not have_nn, reason="Requires TensorFlow or Torch")
def test_evac_plus_flow():
    with TemporaryDirectory() as out_dir:
        file_path = get_fnames(name="evac_test_data")

        volume = np.load(file_path)["input"][0]
        temp_affine = np.eye(4)
        temp_path = pjoin(out_dir, "temp.nii.gz")
        save_nifti(temp_path, volume, temp_affine)
        save_masked = True

        evac_flow = EVACPlusFlow()
        evac_flow.run(temp_path, out_dir=out_dir, save_masked=save_masked)

        mask_name = evac_flow.last_generated_outputs["out_mask"]
        masked_name = evac_flow.last_generated_outputs["out_masked"]

        evac = EVACPlus()
        mask = evac.predict(volume, temp_affine)
        masked = volume * mask

        result_mask_data = load_nifti_data(pjoin(out_dir, mask_name))
        npt.assert_array_equal(result_mask_data.astype(np.uint8), mask)

        result_masked_data = load_nifti_data(pjoin(out_dir, masked_name))

        npt.assert_array_equal(result_masked_data, masked)


@pytest.mark.skipif(not have_nn, reason="Requires TensorFlow or Torch")
def test_correct_biasfield_flow():
    # Test with T1 data
    if have_nn:
        with TemporaryDirectory() as out_dir:
            file_path = get_fnames(name="deepn4_test_data")

            volume = np.load(file_path[0])["img"]
            temp_affine = np.load(file_path[0])["affine"]
            temp_path = pjoin(out_dir, "temp.nii.gz")
            save_nifti(temp_path, volume, temp_affine)

            bias_flow = BiasFieldCorrectionFlow()
            bias_flow.run(temp_path, out_dir=out_dir)

            corrected_name = bias_flow.last_generated_outputs["out_corrected"]

            corrected_data = load_nifti_data(pjoin(out_dir, corrected_name))
            npt.assert_almost_equal(
                corrected_data.mean(), 119.03902876428222, decimal=4
            )

    # Test with DWI data
    with TemporaryDirectory() as out_dir:
        fdata, fbval, fbvec = get_fnames(name="small_25")
        args = {
            "input_files": fdata,
            "bval": fbval,
            "bvec": fbvec,
            "method": "b0",
            "out_dir": out_dir,
        }
        bias_flow = BiasFieldCorrectionFlow()
        bias_flow.run(**args)

        corrected_name = bias_flow.last_generated_outputs["out_corrected"]

        corrected_data = load_nifti_data(pjoin(out_dir, corrected_name))
        npt.assert_almost_equal(corrected_data.mean(), 0.0384615384615, decimal=5)

    args = {
        "input_files": fdata,
        "bval": fbval,
        "bvec": fbvec,
        "method": "random",
        "out_dir": out_dir,
    }
    npt.assert_raises(SystemExit, bias_flow.run, **args)
