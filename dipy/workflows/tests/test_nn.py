import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.nn.evac import EVACPlus
from dipy.utils.optpkg import optional_package
from dipy.workflows.nn import BiasFieldCorrectionFlow, EVACPlusFlow

tf, have_tf, _ = optional_package("tensorflow", min_version="2.18.0")
torch, have_torch, _ = optional_package("torch", min_version="2.2.0")
have_nn = have_tf or have_torch


@pytest.mark.skipif(not have_nn, reason="Requires TensorFlow or Torch")
def test_evac_plus_flow(tmp_path):
    file_path = get_fnames(name="evac_test_data")

    volume = np.load(file_path)["input"][0]
    temp_affine = np.eye(4)
    temp_path = tmp_path / "temp.nii.gz"
    save_nifti(temp_path, volume, temp_affine)
    save_masked = True

    evac_flow = EVACPlusFlow()
    evac_flow.run(temp_path, out_dir=tmp_path, save_masked=save_masked)

    mask_name = evac_flow.last_generated_outputs["out_mask"]
    masked_name = evac_flow.last_generated_outputs["out_masked"]

    evac = EVACPlus()
    mask = evac.predict(volume, temp_affine)
    masked = volume * mask

    result_mask_data = load_nifti_data(tmp_path / mask_name)
    npt.assert_array_equal(result_mask_data.astype(np.uint8), mask)

    result_masked_data = load_nifti_data(tmp_path / masked_name)

    npt.assert_array_equal(result_masked_data, masked)


def test_correct_biasfield_flow(tmp_path):
    # Test with T1 data
    if have_nn:
        t1_path = get_fnames(name="stanford_t1")

        volume, affine = load_nifti(t1_path)

        bias_flow = BiasFieldCorrectionFlow()
        bias_flow.run(t1_path, out_dir=tmp_path, method="n4")

        corrected_name = bias_flow.last_generated_outputs["out_corrected"]

        corrected_data = load_nifti_data(tmp_path / corrected_name)
        assert corrected_data.shape == volume.shape

    # Test with DWI data
    fdata, fbval, fbvec = get_fnames(name="small_25")
    args = {"input_files": fdata, "bval": fbval, "bvec": fbvec, "out_dir": tmp_path}
    bias_flow = BiasFieldCorrectionFlow()
    bias_flow.run(**args)

    corrected_name = bias_flow.last_generated_outputs["out_corrected"]

    corrected_data = load_nifti_data(tmp_path / corrected_name)
    npt.assert_almost_equal(corrected_data.mean(), 44.00769230769231, decimal=0)

    args = {
        "input_files": fdata,
        "bval": fbval,
        "bvec": fbvec,
        "method": "random",
        "out_dir": tmp_path,
    }
    npt.assert_raises(SystemExit, bias_flow.run, **args)


def test_correct_biasfield_flow_with_mask(tmp_path):
    """Test that a mask file can be passed to BiasFieldCorrectionFlow."""
    fdata, fbval, fbvec = get_fnames(name="small_25")
    volume, affine = load_nifti(fdata)
    mask_data = np.ones(volume.shape[:3], dtype=np.uint8)
    mask_path = tmp_path / "mask.nii.gz"
    save_nifti(mask_path, mask_data, affine)

    args = {
        "input_files": fdata,
        "bval": fbval,
        "bvec": fbvec,
        "mask": str(mask_path),
        "out_dir": tmp_path,
    }
    bias_flow = BiasFieldCorrectionFlow()
    bias_flow.run(**args)

    corrected_name = bias_flow.last_generated_outputs["out_corrected"]
    corrected_data = load_nifti_data(tmp_path / corrected_name)
    assert corrected_data.shape == volume.shape


def test_correct_biasfield_flow_with_bias_output(tmp_path):
    """Test that out_bias_field is saved when using poly/bspline methods."""
    fdata, fbval, fbvec = get_fnames(name="small_25")
    bias_field_name = "my_bias_field.nii.gz"
    args = {
        "input_files": fdata,
        "bval": fbval,
        "bvec": fbvec,
        "method": "poly",
        "out_dir": tmp_path,
        "out_bias_field": bias_field_name,
    }
    bias_flow = BiasFieldCorrectionFlow()
    bias_flow.run(**args)

    corrected_name = bias_flow.last_generated_outputs["out_corrected"]
    assert (tmp_path / corrected_name).exists()
    assert (tmp_path / bias_field_name).exists()
