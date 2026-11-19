import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.nn.evac import EVACPlus
from dipy.utils.optpkg import optional_package
from dipy.workflows.nn import BiasFieldCorrectionFlow, EVACPlusFlow, Synb0SynFlow

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
    t1_path = get_fnames(name="stanford_t1")

    volume, affine = load_nifti(t1_path)

    bias_flow = BiasFieldCorrectionFlow()
    bias_flow.run(t1_path, out_dir=tmp_path, method="n4", device="cpu")

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
    npt.assert_almost_equal(corrected_data.mean(), 77.25384615384615, decimal=0)

    args = {
        "input_files": fdata,
        "bval": fbval,
        "bvec": fbvec,
        "method": "random",
        "out_dir": tmp_path,
    }
    npt.assert_raises(SystemExit, bias_flow.run, **args)


def test_correct_biasfield_flow_use_cuda_deprecated(tmp_path):
    t1_path = get_fnames(name="stanford_t1")
    bias_flow = BiasFieldCorrectionFlow()
    with pytest.warns(DeprecationWarning, match="use_cuda"):
        npt.assert_raises(
            ValueError,
            bias_flow.run,
            t1_path,
            out_dir=tmp_path,
            method="n4",
            use_cuda=True,
            device="cpu",
        )


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


def test_synb0_syn_flow(tmp_path):
    fraw, _, _ = get_fnames(name="stanford_hardi")
    dwi, dwi_affine = load_nifti(fraw)
    dwi_path = tmp_path / "dwi.nii.gz"
    save_nifti(dwi_path, dwi[..., :2].astype(np.float32), dwi_affine)

    flow = Synb0SynFlow()
    flow.run(dwi_path, get_fnames(name="stanford_t1"), out_dir=tmp_path)

    corrected, affine = load_nifti(flow.last_generated_outputs["out_corrected"])
    field = load_nifti_data(flow.last_generated_outputs["out_field"])
    npt.assert_equal(corrected.shape, (*dwi.shape[:3], 2))
    npt.assert_array_equal(affine, dwi_affine)
    npt.assert_equal(field.shape, dwi.shape[:3])
    npt.assert_(np.all(np.isfinite(field)))
    npt.assert_(np.any(field != 0))
