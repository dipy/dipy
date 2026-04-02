import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.io.peaks import load_pam
from dipy.reconst.shm import descoteaux07_legacy_msg, sph_harm_ind_list
from dipy.sims.voxel import multi_tensor
from dipy.utils.optpkg import optional_package
from dipy.workflows.reconst import (
    ReconstForceFlow,
    ReconstForecastFlow,
    ReconstGQIFlow,
    ReconstPowermapFlow,
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
            bval_path = Path(out_dir) / os.path.basename(bval_path)
            bvec_path = Path(out_dir) / os.path.basename(bvec_path)
            np.savetxt(bval_path, bvals_2s)
            np.savetxt(bvec_path, bvecs_2s)

            volume = simulate_multitensor_data(gtab_2s)
            data_path = Path(out_dir) / "dwi.nii.gz"
            mask = np.ones_like(volume[..., 0], dtype=bool)
            mask_path = Path(out_dir) / "mask.nii.gz"
            save_nifti(mask_path, mask.astype(np.int32), np.eye(4))
            save_nifti(data_path, volume, np.eye(4))
        else:
            volume, affine = load_nifti(data_path)
            mask = np.ones_like(volume[:, :, :, 0])
            mask_path = Path(out_dir) / "tmp_mask.nii.gz"
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


def test_reconst_powermap_basic():
    """Test ReconstPowermapFlow basic functionality and parameter variations."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*descoteaux07 SH basis.*",
            category=PendingDeprecationWarning,
        )

        with TemporaryDirectory() as out_dir:
            data_path, bval_path, bvec_path = get_fnames(name="small_64D")
            volume, affine = load_nifti(data_path)
            mask = np.ones_like(volume[:, :, :, 0], dtype=bool)
            mask_path = Path(out_dir) / "tmp_mask.nii.gz"
            save_nifti(mask_path, mask.astype(np.uint8), affine)

            reconst_flow = ReconstPowermapFlow()

            # Test basic functionality with default parameters
            reconst_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                out_dir=out_dir,
            )

            powermap_path = reconst_flow.last_generated_outputs["out_powermap"]
            assert os.path.exists(powermap_path), "Powermap file was not created"
            powermap_data = load_nifti_data(powermap_path)
            npt.assert_equal(powermap_data.shape, volume.shape[:-1])
            assert np.all(powermap_data >= 0), "Powermap should be non-negative"
            assert not np.all(powermap_data == 0), "Powermap should not be all zeros"
            assert np.isfinite(
                powermap_data
            ).all(), "Powermap should contain finite values"

            # Test different SH orders
            for sh_order in [4, 6]:
                reconst_flow.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    sh_order_max=sh_order,
                    out_dir=out_dir,
                    out_powermap=f"powermap_sh{sh_order}.nii.gz",
                )
                powermap_path = reconst_flow.last_generated_outputs["out_powermap"]
                assert os.path.exists(powermap_path)

            # Test different power values and norm factors
            for power in [1, 3]:
                reconst_flow.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    power=power,
                    out_dir=out_dir,
                    out_powermap=f"powermap_power{power}.nii.gz",
                )
                powermap_path = reconst_flow.last_generated_outputs["out_powermap"]
                assert os.path.exists(powermap_path)

            # Test with smoothing
            reconst_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                smooth=0.006,
                out_dir=out_dir,
                out_powermap="powermap_smooth.nii.gz",
            )
            powermap_path = reconst_flow.last_generated_outputs["out_powermap"]
            assert os.path.exists(powermap_path)

            # Test non_negative=False
            reconst_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                non_negative=False,
                out_dir=out_dir,
                out_powermap="powermap_negative_allowed.nii.gz",
            )
            powermap_path = reconst_flow.last_generated_outputs["out_powermap"]
            assert os.path.exists(powermap_path)
            assert reconst_flow.get_short_name() == "powermap"


def test_reconst_powermap_with_shm_files():
    """Test ReconstPowermapFlow with precomputed SH coefficients."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*descoteaux07 SH basis.*",
            category=PendingDeprecationWarning,
        )

        with TemporaryDirectory() as out_dir:
            data_path, bval_path, bvec_path = get_fnames(name="small_64D")
            volume, affine = load_nifti(data_path)
            mask = np.ones_like(volume[:, :, :, 0], dtype=bool)
            mask_path = Path(out_dir) / "tmp_mask.nii.gz"
            save_nifti(mask_path, mask.astype(np.uint8), affine)

            # Generate SH coefficients using GQI
            reconst_flow = ReconstGQIFlow()
            reconst_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                sh_order_max=6,
                extract_pam_values=True,
                out_dir=out_dir,
            )

            shm_path = reconst_flow.last_generated_outputs["out_shm"]
            pam_path = reconst_flow.last_generated_outputs["out_pam"]

            # Test with NIfTI SH file
            powermap_flow = ReconstPowermapFlow()
            powermap_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                shm_files=shm_path,
                out_dir=out_dir,
                out_powermap="powermap_from_shm.nii.gz",
            )
            powermap_path = powermap_flow.last_generated_outputs["out_powermap"]
            assert os.path.exists(powermap_path)
            powermap_data = load_nifti_data(powermap_path)
            npt.assert_equal(powermap_data.shape, volume.shape[:-1])

            # Test with PAM file
            powermap_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                shm_files=pam_path,
                out_dir=out_dir,
                out_powermap="powermap_from_pam.nii.gz",
            )
            powermap_path = powermap_flow.last_generated_outputs["out_powermap"]
            assert os.path.exists(powermap_path)


def test_reconst_powermap_edge_cases():
    """Test ReconstPowermapFlow error handling and edge cases."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*descoteaux07 SH basis.*",
            category=PendingDeprecationWarning,
        )

        with TemporaryDirectory() as out_dir:
            data_path, bval_path, bvec_path = get_fnames(name="small_64D")
            volume, affine = load_nifti(data_path)
            mask = np.ones_like(volume[:, :, :, 0], dtype=bool)
            mask_path = Path(out_dir) / "tmp_mask.nii.gz"
            save_nifti(mask_path, mask.astype(np.uint8), affine)

            reconst_flow = ReconstPowermapFlow()

            # Test with invalid SH file extension
            invalid_shm_path = Path(out_dir) / "invalid.txt"
            invalid_shm_path.write_text("dummy content")
            with pytest.raises(ValueError, match="SH coefficients file must be"):
                reconst_flow.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    shm_files=str(invalid_shm_path),
                    out_dir=out_dir,
                )

            # Test with invalid sh_basis (should fallback to default)
            reconst_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                sh_basis="invalid_basis",
                out_dir=out_dir,
                out_powermap="powermap_invalid_basis.nii.gz",
            )
            powermap_path = reconst_flow.last_generated_outputs["out_powermap"]
            assert os.path.exists(powermap_path)

            # Test different norm factors produce different results
            norm_factors = [0.00001, 0.001]
            powermaps = []
            for i, norm_factor in enumerate(norm_factors):
                reconst_flow.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    norm_factor=norm_factor,
                    out_dir=out_dir,
                    out_powermap=f"powermap_norm{i}.nii.gz",
                )
                powermap_path = reconst_flow.last_generated_outputs["out_powermap"]
                powermap_data = load_nifti_data(powermap_path)
                assert np.all(powermap_data >= 0), "Powermap should be non-negative"
                assert np.isfinite(powermap_data).all(), "All values should be finite"
                powermaps.append(powermap_data)

            # Different norm factors should produce different results
            assert not np.allclose(
                powermaps[0], powermaps[1], rtol=0.1
            ), "Different norm factors should produce different results"


# ---------------------------------------------------------------------------
# FORCE workflow tests
# ---------------------------------------------------------------------------


def _make_fake_simulations(n_gradients, *, n_sims=100):
    """Build a minimal simulations dict for FORCE tests.

    Parameters
    ----------
    n_gradients : int
        Number of DWI gradients (determines signal length).
    n_sims : int
        Number of library entries.

    Returns
    -------
    sims : dict
        Minimal simulations dict accepted by FORCEModel._prepare_library.
    """
    rng = np.random.default_rng(42)
    n_dirs = len(default_sphere.vertices)
    sims = {
        "signals": rng.random((n_sims, n_gradients)).astype(np.float32),
        "fa": rng.random(n_sims).astype(np.float32),
        "md": rng.random(n_sims).astype(np.float32) * 3e-3,
        "rd": rng.random(n_sims).astype(np.float32) * 1e-3,
        "wm_fraction": rng.random(n_sims).astype(np.float32),
        "gm_fraction": rng.random(n_sims).astype(np.float32),
        "csf_fraction": rng.random(n_sims).astype(np.float32),
        "num_fibers": rng.integers(0, 3, n_sims).astype(np.float32),
        "dispersion": rng.random(n_sims).astype(np.float32),
        "nd": rng.random(n_sims).astype(np.float32),
        "labels": rng.integers(0, 2, (n_sims, n_dirs)).astype(np.int8),
        "fraction_array": rng.random((n_sims, 3)).astype(np.float32),
    }
    return sims


def _make_fake_simulations_dki(n_gradients, *, n_sims=100):
    """Like _make_fake_simulations but with DKI fields."""
    sims = _make_fake_simulations(n_gradients, n_sims=n_sims)
    rng = np.random.default_rng(99)
    sims.update(
        {
            "ak": rng.random(n_sims).astype(np.float32),
            "rk": rng.random(n_sims).astype(np.float32),
            "mk": rng.random(n_sims).astype(np.float32),
            "kfa": rng.random(n_sims).astype(np.float32),
        }
    )
    return sims


def _patch_generate(monkeypatch, fake_sims):
    """Patch FORCEModel.generate to inject fake_sims without running simulations."""

    def _fake_generate(self, **kwargs):
        self.simulations = fake_sims
        self._prepare_library()
        return self

    monkeypatch.setattr("dipy.reconst.force.FORCEModel.generate", _fake_generate)


def test_reconst_force_basic(monkeypatch, tmp_path):
    """ReconstForceFlow runs and produces expected output files."""
    data_path, bval_path, bvec_path = get_fnames(name="small_64D")
    volume, affine = load_nifti(data_path)
    mask = np.ones(volume.shape[:3], dtype=np.uint8)
    mask_path = tmp_path / "mask.nii.gz"
    save_nifti(mask_path, mask, affine)

    bvals, _ = read_bvals_bvecs(bval_path, bvec_path)
    n_gradients = len(bvals)

    _patch_generate(monkeypatch, _make_fake_simulations(n_gradients))

    flow = ReconstForceFlow()
    flow.run(
        str(data_path),
        str(bval_path),
        str(bvec_path),
        str(mask_path),
        n_neighbors=5,
        engine="serial",
        out_dir=str(tmp_path),
    )

    for metric in [
        "fa",
        "md",
        "rd",
        "wm_fraction",
        "gm_fraction",
        "csf_fraction",
        "num_fibers",
        "dispersion",
        "nd",
        "uncertainty",
        "ambiguity",
    ]:
        out_path = flow.last_generated_outputs[f"out_{metric}"]
        assert os.path.exists(out_path), f"Missing output: {metric}"
        data_arr = load_nifti_data(out_path)
        npt.assert_equal(data_arr.shape, volume.shape[:3])
        assert np.isfinite(data_arr).all(), f"Non-finite values in {metric}"


def test_reconst_force_select_metrics(monkeypatch, tmp_path):
    """save_metrics limits which output files are written."""
    data_path, bval_path, bvec_path = get_fnames(name="small_64D")
    volume, affine = load_nifti(data_path)
    mask = np.ones(volume.shape[:3], dtype=np.uint8)
    mask_path = tmp_path / "mask.nii.gz"
    save_nifti(mask_path, mask, affine)

    bvals, _ = read_bvals_bvecs(bval_path, bvec_path)
    _patch_generate(monkeypatch, _make_fake_simulations(len(bvals)))

    flow = ReconstForceFlow()
    flow.run(
        str(data_path),
        str(bval_path),
        str(bvec_path),
        str(mask_path),
        n_neighbors=5,
        engine="serial",
        save_metrics=["fa", "md"],
        out_dir=str(tmp_path),
        out_fa="only_fa.nii.gz",
        out_md="only_md.nii.gz",
    )

    assert os.path.exists(flow.last_generated_outputs["out_fa"])
    assert os.path.exists(flow.last_generated_outputs["out_md"])


def test_reconst_force_dki_metrics(monkeypatch, tmp_path):
    """DKI metrics are saved when compute_kurtosis=True."""
    data_path, bval_path, bvec_path = get_fnames(name="small_64D")
    volume, affine = load_nifti(data_path)
    mask = np.ones(volume.shape[:3], dtype=np.uint8)
    mask_path = tmp_path / "mask.nii.gz"
    save_nifti(mask_path, mask, affine)

    bvals, _ = read_bvals_bvecs(bval_path, bvec_path)
    _patch_generate(monkeypatch, _make_fake_simulations_dki(len(bvals)))

    flow = ReconstForceFlow()
    flow.run(
        str(data_path),
        str(bval_path),
        str(bvec_path),
        str(mask_path),
        n_neighbors=5,
        engine="serial",
        compute_kurtosis=True,
        out_dir=str(tmp_path),
    )

    for metric in ["mk", "ak", "rk", "kfa"]:
        out_path = flow.last_generated_outputs[f"out_{metric}"]
        assert os.path.exists(out_path), f"Missing DKI output: {metric}"
        data_arr = load_nifti_data(out_path)
        npt.assert_equal(data_arr.shape, volume.shape[:3])


def test_reconst_force_ray_fallback(monkeypatch, tmp_path):
    """engine='ray' falls back to serial when ray is not available."""
    data_path, bval_path, bvec_path = get_fnames(name="small_64D")
    volume, affine = load_nifti(data_path)
    mask = np.ones(volume.shape[:3], dtype=np.uint8)
    mask_path = tmp_path / "mask.nii.gz"
    save_nifti(mask_path, mask, affine)

    bvals, _ = read_bvals_bvecs(bval_path, bvec_path)
    _patch_generate(monkeypatch, _make_fake_simulations(len(bvals)))

    # Simulate ray being unavailable
    monkeypatch.setattr(
        "dipy.utils.optpkg.optional_package",
        lambda name, *a, **kw: (
            (None, False, None) if name == "ray" else optional_package(name, *a, **kw)
        ),
    )

    flow = ReconstForceFlow()
    flow.run(
        str(data_path),
        str(bval_path),
        str(bvec_path),
        str(mask_path),
        n_neighbors=5,
        engine="ray",
        save_metrics=["fa"],
        out_dir=str(tmp_path),
    )

    out_path = flow.last_generated_outputs["out_fa"]
    assert os.path.exists(out_path)
