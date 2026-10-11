import logging
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from scipy.ndimage import gaussian_filter

from dipy.data import get_fnames
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.testing import assert_false, assert_greater, assert_less, assert_true
from dipy.testing.decorators import set_random_number_generator
from dipy.utils.optpkg import optional_package
from dipy.workflows.denoise import (
    DenoiseFlow,
    GibbsRingingFlow,
    LPCAFlow,
    MPPCAFlow,
    NLMeansFlow,
    Patch2SelfFlow,
    _patch2self_extra_args,
    estimate_dwi_snr,
    select_denoising_method,
)

sklearn, has_sklearn, _ = optional_package("sklearn")
needs_sklearn = pytest.mark.skipif(
    not has_sklearn, reason=sklearn._msg if not has_sklearn else ""
)


def test_nlmeans_flow(tmp_path):
    data_path, _, _ = get_fnames()
    volume, affine = load_nifti(data_path)

    nlmeans_flow = NLMeansFlow()

    nlmeans_flow.run(data_path, out_dir=tmp_path)
    assert_true(Path(nlmeans_flow.last_generated_outputs["out_denoised"]).is_file())

    nlmeans_flow._force_overwrite = True
    nlmeans_flow.run(data_path, sigma=4, out_dir=tmp_path)
    denoised_path = nlmeans_flow.last_generated_outputs["out_denoised"]
    assert_true(Path(denoised_path).is_file())
    denoised_data, denoised_affine = load_nifti(denoised_path)
    npt.assert_equal(denoised_data.shape, volume.shape)
    npt.assert_array_almost_equal(denoised_affine, affine)


@needs_sklearn
def test_patch2self_flow(tmp_path):
    data_path, fbvals, _ = get_fnames()

    patch2self_flow = Patch2SelfFlow()
    patch2self_flow.run(
        data_path, fbvals, patch_radius=(0, 0, 0), out_dir=tmp_path, ver=1
    )
    assert_true(Path(patch2self_flow.last_generated_outputs["out_denoised"]).is_file())
    patch2self_flow = Patch2SelfFlow()
    patch2self_flow.run(
        data_path, fbvals, patch_radius=(0, 0, 0), out_dir=tmp_path, ver=3
    )
    assert_true(Path(patch2self_flow.last_generated_outputs["out_denoised"]).is_file())


def test_patch2self_extra_args_ignored_radius_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="dipy"):
        extra_args = _patch2self_extra_args(2, ver=3)
    npt.assert_equal(extra_args, {})
    assert_true(
        any(
            "patch_radius" in record.getMessage()
            for record in caplog.records
            if record.levelname == "WARNING"
        )
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="dipy"):
        extra_args = _patch2self_extra_args(2, ver=1)
    npt.assert_equal(extra_args, {"patch_radius": 2})
    assert_false(
        any(
            "patch_radius" in record.getMessage()
            for record in caplog.records
            if record.levelname == "WARNING"
        )
    )


def test_lpca_flow(tmp_path):
    data_path, fbvals, fbvecs = get_fnames()

    lpca_flow = LPCAFlow()
    lpca_flow.run(data_path, fbvals, fbvecs, out_dir=tmp_path)
    assert_true(Path(lpca_flow.last_generated_outputs["out_denoised"]).is_file())


@set_random_number_generator()
def test_mppca_flow(tmp_path, rng=None):
    S0 = 100 + 2 * rng.standard_normal((22, 23, 30, 20))
    data_path = tmp_path / "random_noise.nii.gz"
    save_nifti(data_path, S0, np.eye(4))

    mppca_flow = MPPCAFlow()
    mppca_flow.run(data_path, out_dir=tmp_path)
    assert_true(Path(mppca_flow.last_generated_outputs["out_denoised"]).is_file())
    assert_false(Path(mppca_flow.last_generated_outputs["out_sigma"]).is_file())

    mppca_flow._force_overwrite = True
    mppca_flow.run(data_path, return_sigma=True, pca_method="svd", out_dir=tmp_path)
    assert_true(Path(mppca_flow.last_generated_outputs["out_denoised"]).is_file())
    assert_true(Path(mppca_flow.last_generated_outputs["out_sigma"]).is_file())

    denoised_path = mppca_flow.last_generated_outputs["out_denoised"]
    denoised_data = load_nifti_data(denoised_path)
    assert_greater(denoised_data.min(), S0.min())
    assert_less(denoised_data.max(), S0.max())
    npt.assert_equal(np.round(denoised_data.mean()), 100)


def test_gibbs_flow(tmp_path):
    def generate_slice():
        Nori = 32
        image = np.zeros((6 * Nori, 6 * Nori))
        image[Nori : 2 * Nori, Nori : 2 * Nori] = 1
        image[Nori : 2 * Nori, 4 * Nori : 5 * Nori] = 1
        image[2 * Nori : 3 * Nori, Nori : 3 * Nori] = 1
        image[3 * Nori : 4 * Nori, 2 * Nori : 3 * Nori] = 2
        image[3 * Nori : 4 * Nori, 4 * Nori : 5 * Nori] = 1
        image[4 * Nori : 5 * Nori, 3 * Nori : 5 * Nori] = 3

        # Corrupt image with gibbs ringing
        c = np.fft.fft2(image)
        c = np.fft.fftshift(c)
        c_crop = c[48:144, 48:144]
        image_gibbs = abs(np.fft.ifft2(c_crop) / 4)
        return image_gibbs

    image4d = np.zeros((96, 96, 2, 2))
    image4d[:, :, 0, 0] = generate_slice()
    image4d[:, :, 1, 0] = generate_slice()
    image4d[:, :, 0, 1] = generate_slice()
    image4d[:, :, 1, 1] = generate_slice()
    data_path = tmp_path / "random_noise.nii.gz"
    save_nifti(data_path, image4d, np.eye(4))

    gibbs_flow = GibbsRingingFlow()
    gibbs_flow.run(data_path, out_dir=tmp_path)
    assert_true(Path(gibbs_flow.last_generated_outputs["out_unring"]).is_file())


def test_select_denoising_method():
    method, _ = select_denoising_method(n_volumes=1, n_dwi=0, snr=np.nan)
    npt.assert_equal(method, "nlmeans")
    method, _ = select_denoising_method(n_volumes=5, n_dwi=4, snr=50.0)
    npt.assert_equal(method, "nlmeans")
    method, _ = select_denoising_method(n_volumes=21, n_dwi=20, snr=50.0)
    npt.assert_equal(method, "mppca")
    method, _ = select_denoising_method(n_volumes=65, n_dwi=64, snr=3.0)
    npt.assert_equal(method, "mppca")
    method, _ = select_denoising_method(n_volumes=65, n_dwi=64, snr=np.nan)
    npt.assert_equal(method, "patch2self")
    method, _ = select_denoising_method(n_volumes=65, n_dwi=64, snr=20.0)
    npt.assert_equal(method, "patch2self")
    method, _ = select_denoising_method(
        n_volumes=21, n_dwi=20, snr=20.0, min_directions=10
    )
    npt.assert_equal(method, "patch2self")
    method, _ = select_denoising_method(n_volumes=65, n_dwi=64, snr=3.0, min_snr=2.0)
    npt.assert_equal(method, "patch2self")


@set_random_number_generator()
def test_estimate_dwi_snr(rng=None):
    data = np.zeros((20, 20, 10, 6), dtype=np.float32)
    data[5:15, 5:15, 2:8, :] = 100
    data = gaussian_filter(data, sigma=(2, 2, 2, 0))
    data += 2 * rng.standard_normal(data.shape).astype(np.float32)
    b0s_mask = np.array([True, False, False, False, False, False])
    snr = estimate_dwi_snr(data=data, b0s_mask=b0s_mask)
    assert_greater(snr, 10)
    assert_less(snr, 50)
    assert_true(np.isnan(estimate_dwi_snr(data=data, b0s_mask=np.ones(6, bool))))


@set_random_number_generator()
def test_denoise_flow(tmp_path, rng=None):
    data = 100 + 2 * rng.standard_normal((22, 23, 10, 20))
    data_path = tmp_path / "few_dirs.nii.gz"
    save_nifti(data_path, data, np.eye(4))
    bvals = np.r_[0, np.full(19, 1000)]
    bval_path = tmp_path / "few_dirs.bval"
    np.savetxt(bval_path, bvals)

    flow = DenoiseFlow()
    flow.run(data_path, bvalues_files=bval_path, out_dir=tmp_path)
    denoised_path = Path(flow.last_generated_outputs["out_denoised"])
    assert_true(denoised_path.is_file())
    npt.assert_equal(load_nifti_data(denoised_path).shape, data.shape)

    flow = DenoiseFlow()
    flow.run(data_path, out_dir=tmp_path / "no_bvals")
    assert_true(Path(flow.last_generated_outputs["out_denoised"]).is_file())

    flow = DenoiseFlow()
    flow.run(data_path, method="nlmeans", out_dir=tmp_path / "nlmeans")
    assert_true(Path(flow.last_generated_outputs["out_denoised"]).is_file())

    data3d_path = tmp_path / "vol.nii.gz"
    save_nifti(data3d_path, data[..., 0], np.eye(4))
    flow = DenoiseFlow()
    flow.run(data3d_path, out_dir=tmp_path / "vol")
    assert_true(Path(flow.last_generated_outputs["out_denoised"]).is_file())

    flow = DenoiseFlow()
    with pytest.raises(SystemExit):
        flow.run(data_path, method="patch2self", out_dir=tmp_path / "p2s")
    flow = DenoiseFlow()
    with pytest.raises(SystemExit):
        flow.run(data_path, method="unknown", out_dir=tmp_path / "unknown")


@set_random_number_generator()
def test_denoise_flow_multiple_inputs(tmp_path, rng=None):
    bvals = np.r_[0, np.full(19, 1000)]
    for subject in ("sub-01", "sub-02"):
        sub_dir = tmp_path / subject
        sub_dir.mkdir()
        data = 100 + 2 * rng.standard_normal((12, 12, 8, 20))
        save_nifti(sub_dir / "dwi.nii.gz", data, np.eye(4))
        np.savetxt(sub_dir / "dwi.bval", bvals)

    flow = DenoiseFlow(output_strategy="append")
    flow.run(
        str(tmp_path / "sub-*" / "dwi.nii.gz"),
        bvalues_files=str(tmp_path / "sub-*" / "dwi.bval"),
        out_dir="denoised",
    )
    for subject in ("sub-01", "sub-02"):
        out_path = tmp_path / subject / "denoised" / "dwi_denoised.nii.gz"
        assert_true(out_path.is_file())

    flow = DenoiseFlow()
    with pytest.raises(SystemExit):
        flow.run(
            str(tmp_path / "sub-*" / "dwi.nii.gz"),
            bvalues_files=tmp_path / "sub-01" / "dwi.bval",
            out_dir=tmp_path / "mismatch",
        )

    flow = DenoiseFlow()
    with pytest.raises(SystemExit):
        flow.run(
            tmp_path / "sub-01" / "dwi.nii.gz",
            bvalues_files=tmp_path / "missing.bval",
            out_dir=tmp_path / "missing",
        )


@set_random_number_generator()
def test_denoise_flow_auto_without_bvals(tmp_path, rng=None):
    data = 100 + 2 * rng.standard_normal((12, 12, 8, 20))
    data_path = tmp_path / "dwi.nii.gz"
    save_nifti(data_path, data, np.eye(4))

    flow = DenoiseFlow()
    flow.run(data_path, min_directions=10, min_snr=1.0, out_dir=tmp_path)
    assert_true(Path(flow.last_generated_outputs["out_denoised"]).is_file())


@needs_sklearn
def test_denoise_flow_patch2self(tmp_path):
    data_path, fbvals, _ = get_fnames()
    flow = DenoiseFlow()
    flow.run(
        data_path,
        bvalues_files=fbvals,
        method="patch2self",
        min_directions=10,
        out_dir=tmp_path,
    )
    assert_true(Path(flow.last_generated_outputs["out_denoised"]).is_file())
