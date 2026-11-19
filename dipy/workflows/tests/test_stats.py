#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.stats.analysis import SpectraGrid
from dipy.testing import assert_true
from dipy.testing.decorators import set_random_number_generator
from dipy.tracking.streamline import Streamlines, set_number_of_points
from dipy.utils.optpkg import optional_package
from dipy.workflows.stats import (
    BundleAnalysisTractometryFlow,
    BundleShapeAnalysis,
    LinearMixedModelsFlow,
    SNRinCCFlow,
    SpectraFDRFlow,
    SpectraHFDRExternalFlow,
    SpectraLMMFlow,
    SpectraProfileFlow,
    buan_bundle_profiles,
)

pd, have_pandas, _ = optional_package("pandas")
_, have_statsmodels, _ = optional_package("statsmodels", min_version="0.14.0")
_, have_tables, _ = optional_package("tables")
_, have_matplotlib, _ = optional_package("matplotlib")


def test_stats(tmp_path):
    data_path, bval_path, bvec_path = get_fnames(name="small_101D")
    volume, affine = load_nifti(data_path)
    mask = np.ones_like(volume[:, :, :, 0], dtype=np.uint8)
    mask_path = tmp_path / "tmp_mask.nii.gz"
    save_nifti(mask_path, mask, affine)

    snr_flow = SNRinCCFlow(force=True)
    args = [data_path, bval_path, bvec_path, mask_path]
    snr_flow.run(*args, out_dir=tmp_path)

    assert_true((tmp_path / "product.json").exists())
    assert_true(os.stat(tmp_path / "product.json").st_size != 0)
    assert_true((tmp_path / "cc.nii.gz").exists())
    assert_true(os.stat(tmp_path / "cc.nii.gz").st_size != 0)
    assert_true((tmp_path / "mask_noise.nii.gz").exists())
    assert_true(os.stat(tmp_path / "mask_noise.nii.gz").st_size != 0)

    snr_flow._force_overwrite = True
    snr_flow.run(*args, out_dir=tmp_path)
    assert_true((tmp_path / "product.json").exists())
    assert_true(os.stat(tmp_path / "product.json").st_size != 0)
    assert_true((tmp_path / "cc.nii.gz").exists())
    assert_true(os.stat(tmp_path / "cc.nii.gz").st_size != 0)
    assert_true((tmp_path / "mask_noise.nii.gz").exists())
    assert_true(os.stat(tmp_path / "mask_noise.nii.gz").st_size != 0)

    snr_flow._force_overwrite = True
    snr_flow.run(*args, bbox_threshold=(0.5, 1, 0, 0.15, 0, 0.2), out_dir=tmp_path)
    assert_true((tmp_path / "product.json").exists())
    assert_true(os.stat(tmp_path / "product.json").st_size != 0)
    assert_true((tmp_path / "cc.nii.gz").exists())
    assert_true(os.stat(tmp_path / "cc.nii.gz").st_size != 0)
    assert_true((tmp_path / "mask_noise.nii.gz").exists())
    assert_true(os.stat(tmp_path / "mask_noise.nii.gz").st_size != 0)


@pytest.mark.skipif(
    not have_pandas or not have_statsmodels or not have_tables or not have_matplotlib,
    reason="Requires Pandas, StatsModels and PyTables",
)
@set_random_number_generator()
def test_buan_bundle_profiles(tmp_path, rng=None):
    data_path = get_fnames(name="fornix")
    fornix = load_tractogram(data_path, "same", bbox_valid_check=False).streamlines

    f = Streamlines(fornix)

    mb = tmp_path / "model_bundles"

    os.mkdir(mb)

    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(sft, Path(mb) / "temp.trk", bbox_valid_check=False)

    rb = tmp_path / "rec_bundles"
    os.mkdir(rb)

    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(sft, Path(rb) / "temp.trk", bbox_valid_check=False)

    ob = tmp_path / "org_bundles"
    os.mkdir(ob)

    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(sft, Path(ob) / "temp.trk", bbox_valid_check=False)

    dt = tmp_path / "anatomical_measures"
    os.mkdir(dt)

    fa = rng.random((255, 255, 255))

    save_nifti(Path(dt) / "fa.nii.gz", fa, affine=np.eye(4))

    out_dir = tmp_path / "output"
    os.mkdir(out_dir)

    buan_bundle_profiles(
        mb,
        rb,
        ob,
        dt,
        group_id=1,
        subject="10001",
        no_disks=100,
        out_dir=out_dir,
    )

    assert_true((out_dir / "temp_fa.h5").exists())


@pytest.mark.skipif(
    not have_pandas or not have_statsmodels or not have_tables or not have_matplotlib,
    reason="Requires Pandas, StatsModels, PyTables, and matplotlib",
)
@set_random_number_generator()
def test_bundle_analysis_tractometry_flow(tmp_path, rng=None):
    data_path = get_fnames(name="fornix")
    fornix = load_tractogram(data_path, "same", bbox_valid_check=False).streamlines

    f = Streamlines(fornix)

    mb = tmp_path / "model_bundles"
    sub = tmp_path / "subjects"

    os.mkdir(mb)
    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(sft, Path(mb) / "temp.trk", bbox_valid_check=False)

    os.mkdir(sub)

    os.mkdir(Path(sub) / "patient")

    os.mkdir(Path(sub) / "control")

    p = sub / "patient" / "10001"
    os.mkdir(p)

    c = sub / "control" / "20002"
    os.mkdir(c)

    for pre in [p, c]:
        os.mkdir(Path(pre) / "rec_bundles")

        sft = StatefulTractogram(f, data_path, Space.RASMM)
        save_tractogram(
            sft,
            Path(pre) / "rec_bundles" / "temp.trk",
            bbox_valid_check=False,
        )
        os.mkdir(Path(pre) / "org_bundles")

        sft = StatefulTractogram(f, data_path, Space.RASMM)
        save_tractogram(
            sft,
            Path(pre) / "org_bundles" / "temp.trk",
            bbox_valid_check=False,
        )
        os.mkdir(Path(pre) / "anatomical_measures")

        fa = rng.random((255, 255, 255))

        save_nifti(
            Path(pre) / "anatomical_measures" / "fa.nii.gz",
            fa,
            affine=np.eye(4),
        )

    out_dir = tmp_path / "output"
    os.mkdir(out_dir)

    ba_flow = BundleAnalysisTractometryFlow()

    ba_flow.run(mb, sub, out_dir=out_dir)

    assert_true(Path(Path(out_dir) / "temp_fa.h5").exists())

    dft = pd.read_hdf(Path(out_dir) / "temp_fa.h5")

    # assert_true(dft.bundle.unique() == "temp")

    assert_true(set(dft.subject.unique()) == {"10001", "20002"})


@pytest.mark.skipif(
    not have_pandas or not have_statsmodels or not have_tables or not have_matplotlib,
    reason="Requires Pandas, StatsModels, PyTables, and matplotlib",
)
def test_linear_mixed_models_flow(tmp_path):
    out_dir = tmp_path / "output"
    os.mkdir(out_dir)

    d = {
        "disk": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5] * 10,
        "fa": [0.21, 0.234, 0.44, 0.44, 0.5, 0.23, 0.55, 0.34, 0.76, 0.34] * 10,
        "subject": [
            "10001",
            "10001",
            "10001",
            "10001",
            "10001",
            "20002",
            "20002",
            "20002",
            "20002",
            "20002",
        ]
        * 10,
        "group": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10,
    }

    df = pd.DataFrame(data=d)
    store = pd.HDFStore(Path(out_dir) / "temp_fa.h5")
    store.append("fa", df, data_columns=True)
    store.close()

    lmm_flow = LinearMixedModelsFlow()

    out_dir2 = tmp_path / "output2"
    os.mkdir(out_dir2)

    input_path = Path(out_dir) / "*"

    lmm_flow.run(input_path, no_disks=5, out_dir=out_dir2)

    assert_true(Path(Path(out_dir2) / "temp_fa_pvalues.npy").exists())
    assert_true(Path(Path(out_dir2) / "temp_fa.png").exists())

    # test error
    d2 = {
        "disk": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5] * 1,
        "fa": [0.21, 0.234, 0.44, 0.44, 0.5, 0.23, 0.55, 0.34, 0.76, 0.34] * 1,
        "subject": [
            "10001",
            "10001",
            "10001",
            "10001",
            "10001",
            "20002",
            "20002",
            "20002",
            "20002",
            "20002",
        ]
        * 1,
        "group": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 1,
    }

    df = pd.DataFrame(data=d2)

    out_dir3 = tmp_path / "output3"
    os.mkdir(out_dir3)

    store = pd.HDFStore(Path(out_dir3) / "temp_fa.h5")
    store.append("fa", df, data_columns=True)
    store.close()

    out_dir4 = tmp_path / "output4"
    os.mkdir(out_dir4)

    input_path = Path(out_dir3) / "f*"
    # OS error raised if path is wrong or file does not exist
    npt.assert_raises(OSError, lmm_flow.run, input_path, no_disks=5, out_dir=out_dir4)

    input_path = Path(out_dir3) / "*"
    # value error raised if length of data frame is less than 100
    npt.assert_raises(
        ValueError, lmm_flow.run, input_path, no_disks=5, out_dir=out_dir4
    )


@set_random_number_generator()
def test_bundle_analysis_tractometry_flow_single_subject(tmp_path, rng=None):
    data_path = get_fnames(name="fornix")
    fornix = load_tractogram(data_path, "same", bbox_valid_check=False).streamlines

    f = Streamlines(fornix)

    mb = tmp_path / "model_bundles"
    os.mkdir(mb)
    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(sft, Path(mb) / "temp.trk", bbox_valid_check=False)

    # Single-subject directory directly contains rec_bundles/, org_bundles/,
    # and anatomical_measures/ (no patient/control subdirs)
    single_sub = tmp_path / "10001"
    os.mkdir(single_sub)

    os.mkdir(single_sub / "rec_bundles")
    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(
        sft, single_sub / "rec_bundles" / "temp.trk", bbox_valid_check=False
    )

    os.mkdir(single_sub / "org_bundles")
    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(
        sft, single_sub / "org_bundles" / "temp.trk", bbox_valid_check=False
    )

    os.mkdir(single_sub / "anatomical_measures")
    fa = rng.random((40, 40, 40))
    save_nifti(
        single_sub / "anatomical_measures" / "fa.nii.gz",
        fa,
        affine=np.eye(4),
    )

    out_dir = tmp_path / "output"
    os.mkdir(out_dir)

    ba_flow = BundleAnalysisTractometryFlow()
    ba_flow.run(mb, single_sub, out_dir=out_dir)

    assert_true(Path(out_dir / "temp_fa_profile.npy").exists())


@pytest.mark.skipif(
    not have_pandas or not have_statsmodels or not have_tables or not have_matplotlib,
    reason="Requires Pandas, StatsModels, PyTables, and matplotlib",
)
@set_random_number_generator()
def test_bundle_shape_analysis_flow(tmp_path, rng=None):
    data_path = get_fnames(name="fornix")
    fornix = load_tractogram(data_path, "same", bbox_valid_check=False).streamlines

    f = Streamlines(fornix)

    mb = tmp_path / "model_bundles"
    sub = tmp_path / "subjects"

    os.mkdir(mb)
    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(sft, Path(mb) / "temp.trk", bbox_valid_check=False)

    os.mkdir(sub)

    os.mkdir(sub / "patient")

    os.mkdir(sub / "control")

    p = sub / "patient" / "10001"
    os.mkdir(p)

    c = sub / "control" / "20002"
    os.mkdir(c)

    for pre in [p, c]:
        os.mkdir(Path(pre) / "rec_bundles")

        sft = StatefulTractogram(f, data_path, Space.RASMM)
        save_tractogram(
            sft,
            Path(pre) / "rec_bundles" / "temp.trk",
            bbox_valid_check=False,
        )
        os.mkdir(Path(pre) / "org_bundles")

        sft = StatefulTractogram(f, data_path, Space.RASMM)
        save_tractogram(
            sft,
            Path(pre) / "org_bundles" / "temp.trk",
            bbox_valid_check=False,
        )
        os.mkdir(Path(pre) / "anatomical_measures")

        fa = rng.random((255, 255, 255))

        save_nifti(
            Path(pre) / "anatomical_measures" / "fa.nii.gz",
            fa,
            affine=np.eye(4),
        )

    out_dir = tmp_path / "output"
    os.mkdir(out_dir)

    sm_flow = BundleShapeAnalysis()

    sm_flow.run(sub, out_dir=out_dir)

    assert_true(Path(Path(out_dir) / "temp.npy").exists())


def _save_spectra_bundle(path, reference, *, seed=0, n_lines=60):
    """Save a straight synthetic bundle along x, as used by the SPECTRA tests."""
    rng = np.random.default_rng(seed)
    lines = [
        np.c_[np.linspace(0, 50, 60), rng.normal(0, 3, 60), rng.normal(0, 3, 60)] + 25
        for _ in range(n_lines)
    ]
    sft = StatefulTractogram(Streamlines(lines), reference, Space.RASMM)
    save_tractogram(sft, path, bbox_valid_check=False)
    return load_tractogram(path, reference="same", bbox_valid_check=False).streamlines


def test_spectra_profile_flow(tmp_path):
    rng = np.random.default_rng(1)
    fa = rng.random((90, 60, 60))
    md = rng.random((90, 60, 60))
    for name, metric in (("fa", fa), ("md", md)):
        save_nifti(tmp_path / f"{name}.nii.gz", metric, np.eye(4))
    reference = tmp_path / "fa.nii.gz"

    for folder in ("model", "rec", "org"):
        (tmp_path / folder).mkdir()
    model = _save_spectra_bundle(tmp_path / "model" / "AF_L.trk", reference)
    rec = _save_spectra_bundle(tmp_path / "rec" / "AF_L.trk", reference, seed=1)
    _save_spectra_bundle(tmp_path / "org" / "AF_L.trk", reference, seed=1)

    out_dir = tmp_path / "profiles" / "sub01"
    SpectraProfileFlow().run(
        tmp_path / "model",
        tmp_path / "rec",
        tmp_path / "org",
        tmp_path / "*.nii.gz",
        out_dir=out_dir,
    )

    # Same profiles as SpectraGrid, on the evenly resampled bundle
    grid = SpectraGrid(model)
    rec = Streamlines(set_number_of_points(s, nb_points=len(s)) for s in rec)
    for name, metric in (("fa", fa), ("md", md)):
        saved = np.load(out_dir / "AF_L" / f"{name}.npz")
        npt.assert_allclose(
            saved["profile_2d"],
            grid.profile(rec, rec, metric, np.eye(4)),
            equal_nan=True,
        )
        npt.assert_allclose(
            saved["profile_1d"],
            grid.profile_1d_adaptive_buan(rec, rec, metric, np.eye(4)),
            equal_nan=True,
        )
        assert saved["n_segments"] == grid.n_segments
        assert saved["n_radial"] == grid.n_radial
        npt.assert_allclose(saved["segment_length"], grid.segment_length_actual)

    # Bundles with too few streamlines are skipped
    out_dir = tmp_path / "skipped"
    SpectraProfileFlow().run(
        tmp_path / "model",
        tmp_path / "rec",
        tmp_path / "org",
        reference,
        min_streamlines=100,
        out_dir=out_dir,
    )
    assert not (out_dir / "AF_L").exists()

    with pytest.raises(ValueError):
        SpectraProfileFlow().run(
            tmp_path / "model", tmp_path / "rec", tmp_path / "missing", reference
        )
    with pytest.raises(ValueError):
        SpectraProfileFlow().run(
            tmp_path / "model",
            tmp_path / "rec",
            tmp_path / "org",
            reference,
            robust_method="cubic",
        )


def _spectra_study(tmp_path, *, n_subjects=90, seed=2):
    """Model bundle, profiles and metadata of a simulated SPECTRA study.

    The cases (AD) have a lower FA in the first 5 segments of the bundle.
    """
    from scipy.ndimage import gaussian_filter

    reference = tmp_path / "ref.nii.gz"
    save_nifti(reference, np.zeros((90, 60, 60)), np.eye(4))
    (tmp_path / "model").mkdir()
    model = _save_spectra_bundle(tmp_path / "model" / "AF_L.trk", reference)
    grid = SpectraGrid(model)

    rng = np.random.default_rng(seed)
    metadata = pd.DataFrame(
        {
            "subject": [f"sub{i:02d}" for i in range(n_subjects)],
            "dx": np.repeat(["AD", "CN"], n_subjects // 2),
            "age": rng.normal(70, 8, n_subjects),
            "sex": np.tile(["F", "M"], n_subjects // 2),
            "site": np.tile(["a", "b", "c"], n_subjects // 3),
        }
    )
    shape = (n_subjects, grid.n_segments, grid.n_radial)
    noise = gaussian_filter(rng.standard_normal(shape), sigma=(0, 1, 0.5))
    profiles = 0.5 + 0.02 * noise / noise.std()
    profiles[metadata["dx"] == "AD", :5] -= 0.03
    for subject, profile in zip(metadata["subject"], profiles):
        folder = tmp_path / "profiles" / subject / "AF_L"
        folder.mkdir(parents=True)
        np.savez(folder / "fa.npz", profile_2d=profile, profile_1d=profile.mean(1))
    # A subject without profile is left out
    metadata.loc[n_subjects] = ["sub99", "AD", 70.0, "F", "a"]
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    return grid


@pytest.mark.skipif(
    not have_pandas or not have_statsmodels,
    reason="Requires pandas and statsmodels",
)
def test_spectra_lmm_and_fdr_flows(tmp_path):
    grid = _spectra_study(tmp_path)
    args = (tmp_path / "metadata.csv", tmp_path / "profiles", tmp_path / "model")
    kwargs = {"target_case": "AD", "target_ref": "CN", "mode": "2d"}

    # Stage 1: fit the models of the bundle
    SpectraLMMFlow().run(*args, **kwargs, out_dir=tmp_path / "fits")
    fit = np.load(tmp_path / "fits" / "AF_L.npz")
    assert fit["t"].shape == (grid.n_segments, grid.n_radial)
    assert fit["n_subjects_total"] == 90
    assert fit["l_s"] > 0
    assert "residuals" not in fit

    # Stage 2: hierarchical FDR, with a sweep of alpha_cell
    out_dir = tmp_path / "hfdr"
    SpectraFDRFlow().run(
        tmp_path / "fits", tmp_path / "model", alpha_cell=[0.05, 0.25], out_dir=out_dir
    )
    cells = pd.read_csv(out_dir / "spectra_stats.csv")
    blocks = pd.read_csv(out_dir / "spectra_blocks.csv")
    summary = pd.read_csv(out_dir / "spectra_summary.csv")
    for column in ("t", "p", "d_cohen", "p_cond", "p_hfdr", "block"):
        assert column in cells.columns
    assert blocks["significant"].any()
    assert summary.loc[0, "n_subjects"] == 90
    assert summary.loc[0, "n_significant_ac05"] <= summary.loc[0, "n_significant_ac25"]
    significant = cells.groupby(cells["segment"] < 5)["significant_ac05"].mean()
    assert significant[True] > 0.8
    assert significant[False] < 0.1

    # Benjamini-Hochberg over all the cells
    out_dir = tmp_path / "fdr"
    SpectraFDRFlow().run(
        tmp_path / "fits",
        tmp_path / "model",
        stats_mode="fdr",
        adaptive=True,
        out_dir=out_dir,
    )
    cells = pd.read_csv(out_dir / "spectra_stats.csv")
    assert "p_fdr" in cells.columns
    assert not (out_dir / "spectra_blocks.csv").exists()
    assert cells.loc[cells["segment"] < 5, "significant"].mean() > 0.8

    # 1D profiles with one block per bundle; edge cells are left out
    SpectraLMMFlow().run(
        *args,
        target_case="AD",
        target_ref="CN",
        stats_mode="hfdr_bundle",
        edge_mask=[1],
        out_dir=tmp_path / "fits_1d",
    )
    fit = np.load(tmp_path / "fits_1d" / "AF_L.npz")
    assert fit["t"].shape == (grid.n_segments, 1)
    assert not fit["mask"][[0, -1]].any()
    SpectraFDRFlow().run(
        tmp_path / "fits_1d",
        tmp_path / "model",
        stats_mode="hfdr_bundle",
        out_dir=tmp_path / "hfdr_1d",
    )
    assert len(pd.read_csv(tmp_path / "hfdr_1d" / "spectra_blocks.csv")) == 1

    # Fits without length scales cannot be used for the hierarchical FDR
    SpectraLMMFlow().run(
        *args, **kwargs, stats_mode="fdr", out_dir=tmp_path / "fits_fdr"
    )
    with pytest.raises(ValueError):
        SpectraFDRFlow().run(tmp_path / "fits_fdr", tmp_path / "model")

    # The grid options must be those of the profiles
    with pytest.raises(ValueError):
        SpectraLMMFlow().run(*args, **kwargs, segment_length=2.0, out_dir=tmp_path)
    with pytest.raises(ValueError):
        SpectraLMMFlow().run(*args, **kwargs, stats_mode="anova", out_dir=tmp_path)


@pytest.mark.skipif(
    not have_pandas or not have_statsmodels,
    reason="Requires pandas and statsmodels",
)
def test_spectra_hfdr_external_flow(tmp_path):
    from scipy import stats

    from dipy.stats.analysis import (
        estimate_length_scales,
        fit_grid_lmm,
        hierarchical_fdr,
    )

    grid = _spectra_study(tmp_path)
    metadata = pd.read_csv(tmp_path / "metadata.csv").iloc[:90]
    profiles = np.array(
        [
            np.load(tmp_path / "profiles" / s / "AF_L" / "fa.npz")["profile_2d"]
            for s in metadata["subject"]
        ]
    )
    fit = fit_grid_lmm(
        profiles,
        metadata,
        "dx",
        groups="site",
        case_group="AD",
        ref_group="CN",
        fit_reduced=True,
    )

    # Statistics and residuals in the CSV formats of SPECTRA
    s_idx, r_idx = np.nonzero(fit["mask"])
    pd.DataFrame(
        {
            "grid_name": "AF_L",
            "s_idx": s_idx,
            "r_idx": r_idx,
            "t_stat": fit["t"][s_idx, r_idx],
            "p_value": fit["p"][s_idx, r_idx],
            "beta": fit["beta"][s_idx, r_idx],
        }
    ).to_csv(tmp_path / "stats.csv", index=False)
    (tmp_path / "residuals").mkdir()
    s_all, r_all, subject = np.nonzero(~np.isnan(fit["residuals_reduced"]))
    pd.DataFrame(
        {
            "s_idx": s_all,
            "r_idx": r_all,
            "subject_id": metadata["subject"].to_numpy()[subject],
            "residual": fit["residuals_reduced"][s_all, r_all, subject],
        }
    ).to_csv(tmp_path / "residuals" / "AF_L_residuals.csv", index=False)

    out_dir = tmp_path / "external"
    SpectraHFDRExternalFlow().run(
        tmp_path / "stats.csv",
        tmp_path / "model",
        residuals_folder=tmp_path / "residuals",
        out_dir=out_dir,
    )
    cells = pd.read_csv(out_dir / "spectra_stats.csv")
    summary = pd.read_csv(out_dir / "spectra_summary.csv")

    # Same result as the library functions
    l_s, l_r = estimate_length_scales(fit["residuals_reduced"], grid)
    npt.assert_allclose(summary.loc[0, ["l_s", "l_r"]].to_numpy(float), [l_s, l_r])
    # As in SPECTRA, the z-scores use p-values bounded to [1e-15, 1 - 1e-15]
    p_bounded = np.clip(fit["p"], 1e-15, 1 - 1e-15)
    z = np.sign(fit["t"]) * stats.norm.ppf(1 - p_bounded / 2)
    bundle = {
        "grid": grid,
        "t": fit["t"],
        "p": fit["p"],
        "z": z,
        "l_s": l_s,
        "l_r": l_r,
    }
    p_hfdr, _ = hierarchical_fdr({"AF_L": bundle})
    npt.assert_allclose(
        cells["p_hfdr"].to_numpy(float),
        p_hfdr["AF_L"][cells["segment"], cells["radial"]],
        equal_nan=True,
    )
    assert cells.loc[cells["segment"] < 5, "significant_ac05"].mean() > 0.8
    # The uncorrected p-values are saved as given
    npt.assert_allclose(cells["p_value"], fit["p"][cells["segment"], cells["radial"]])

    # Bundle mode needs no residuals
    SpectraHFDRExternalFlow().run(
        tmp_path / "stats.csv",
        tmp_path / "model",
        no_resel_blocks=True,
        out_dir=tmp_path / "external_bundle",
    )
    blocks = pd.read_csv(tmp_path / "external_bundle" / "spectra_blocks.csv")
    assert len(blocks) == 1

    pd.DataFrame({"grid_name": ["AF_L"], "s_idx": [0]}).to_csv(
        tmp_path / "bad.csv", index=False
    )
    with pytest.raises(ValueError):
        SpectraHFDRExternalFlow().run(tmp_path / "bad.csv", tmp_path / "model")
