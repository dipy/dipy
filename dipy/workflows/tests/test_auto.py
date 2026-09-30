import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy.testing as npt
import pytest

from dipy.io.tests.test_bids import _make_bids_tree, _write_gradients, _write_nifti
from dipy.workflows.combined_workflow import AutoFlow
from dipy.workflows.tests.test_qc import _phantom


def _summary(deriv_root):
    """Load the JSON batch summary written in a derivative dataset."""
    names = [f for f in os.listdir(deriv_root) if f.endswith("_batch_summary.json")]
    npt.assert_equal(len(names), 1)
    with open(os.path.join(deriv_root, names[0])) as f:
        return json.load(f)


def _configs(deriv_root):
    """List the pipeline configurations written under a derivative dataset."""
    return sorted(
        os.path.relpath(os.path.join(directory, name), deriv_root)
        for directory, _, names in os.walk(deriv_root)
        for name in names
        if name.endswith("_pipeline.toml")
    )


def _phantom_bids_tree(root):
    """Build a one subject BIDS dataset holding a DWI phantom."""
    root = Path(root)
    (root / "sub-01" / "dwi").mkdir(parents=True)
    with open(root / "dataset_description.json", "w") as f:
        json.dump({"Name": "phantom", "BIDSVersion": "1.9.0"}, f)

    paths = _phantom(str(root / "sub-01" / "dwi"))
    for source, target in (
        ("dwi", "sub-01_dwi.nii.gz"),
        ("bvals", "sub-01_dwi.bval"),
        ("bvecs", "sub-01_dwi.bvec"),
    ):
        os.rename(paths[source], root / "sub-01" / "dwi" / target)

    return str(root)


def test_auto_single_subject_dry_run():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        out_dir = os.path.join(tmp, "out")

        AutoFlow().run(
            [paths["dwi"], paths["bvals"], paths["bvecs"]],
            pipeline_type="denoise",
            dry_run=True,
            out_dir=out_dir,
        )

        configs = [f for f in os.listdir(out_dir) if f.endswith("_pipeline.toml")]
        npt.assert_equal(len(configs), 1)
        npt.assert_equal(
            any(f.endswith("_dipy_auto.log") for f in os.listdir(out_dir)), True
        )


def test_auto_bids_dry_run():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        deriv = os.path.join(tmp, "deriv")

        # sub-03 has no gradient files and is covered by its own test
        AutoFlow().run(
            [root, deriv, "participant"],
            bids=True,
            participant_label=["01", "02"],
            pipeline_type="denoise",
            dry_run=True,
        )

        with open(os.path.join(deriv, "dataset_description.json")) as f:
            description = json.load(f)
        npt.assert_equal(description["DatasetType"], "derivative")
        npt.assert_equal(description["GeneratedBy"][0]["Name"], "dipy")

        summary = _summary(deriv)
        npt.assert_equal(len(summary["runs"]), 4)
        npt.assert_equal(summary["pipeline"], "denoise")
        npt.assert_equal(summary["bids_dir"], root)

        # one configuration per acquisition, each pointing at its own DWI
        configs = _configs(deriv)
        npt.assert_equal(len(configs), 4)

        run = next(r for r in summary["runs"] if r["label"] == "sub-01")
        toml_file = next(
            f for f in os.listdir(run["out_dir"]) if f.endswith("_pipeline.toml")
        )
        text = Path(run["out_dir"], toml_file).read_text()
        npt.assert_equal(
            os.path.join(root, "sub-01", "dwi", "sub-01_dwi.nii.gz") in text, True
        )
        npt.assert_equal(
            os.path.join(root, "sub-01", "anat", "sub-01_T1w.nii.gz") in text, True
        )


def test_auto_bids_default_derivatives_root():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")

        AutoFlow().run(
            [root],
            bids=True,
            participant_label=["01", "02"],
            pipeline_type="denoise",
            dry_run=True,
        )

        deriv = os.path.join(root, "derivatives", "dipy")
        npt.assert_equal(
            os.path.exists(os.path.join(deriv, "dataset_description.json")), True
        )
        npt.assert_equal(len(_summary(deriv)["runs"]), 4)


def test_auto_bids_filters():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        deriv = os.path.join(tmp, "deriv")

        AutoFlow().run(
            [root, deriv],
            bids=True,
            participant_label=["02"],
            session_label=["1"],
            pipeline_type="denoise",
            dry_run=True,
        )

        summary = _summary(deriv)
        npt.assert_equal(len(summary["runs"]), 2)
        npt.assert_equal({r["subject"] for r in summary["runs"]}, {"02"})
        npt.assert_equal({r["session"] for r in summary["runs"]}, {"1"})


def test_auto_bids_no_match_exits():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [root, os.path.join(tmp, "deriv")],
            bids=True,
            participant_label=["99"],
            pipeline_type="denoise",
            dry_run=True,
        )


def test_auto_bids_derivatives_layout():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        deriv = os.path.join(tmp, "deriv")

        AutoFlow().run(
            [root, deriv],
            bids=True,
            participant_label=["01", "02"],
            pipeline_type="denoise",
            dry_run=True,
        )

        out_dirs = {r["label"]: r["out_dir"] for r in _summary(deriv)["runs"]}
        npt.assert_equal(out_dirs["sub-01"], os.path.join(deriv, "sub-01", "dwi"))
        npt.assert_equal(
            out_dirs["sub-02_ses-2"], os.path.join(deriv, "sub-02", "ses-2", "dwi")
        )
        # two acquisitions share one session: each gets its own subdirectory
        npt.assert_equal(
            out_dirs["sub-02_ses-1_acq-b1000_dir-AP_run-1"],
            os.path.join(deriv, "sub-02", "ses-1", "dwi", "acq-b1000_dir-AP_run-1"),
        )
        npt.assert_equal(
            out_dirs["sub-02_ses-1_acq-b1000_dir-AP_run-2"],
            os.path.join(deriv, "sub-02", "ses-1", "dwi", "acq-b1000_dir-AP_run-2"),
        )


def test_auto_bids_missing_gradients_is_isolated():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        deriv = os.path.join(tmp, "deriv")

        # sub-03 has no bval/bvec: it fails while the rest of the batch runs
        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [root, deriv],
            bids=True,
            pipeline_type="denoise",
        )

        summary = _summary(deriv)
        statuses = {r["label"]: r["status"] for r in summary["runs"]}
        npt.assert_equal(statuses["sub-03"], "failed")
        npt.assert_equal(len([s for s in statuses.values() if s == "failed"]), 1)
        failure = next(r for r in summary["runs"] if r["label"] == "sub-03")
        npt.assert_equal("b-value" in failure["error"], True)


def test_auto_bids_stop_on_error():
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "ds"
        _make_bids_tree(root)
        # a subject sorting before the others and missing its gradients
        _write_nifti(root / "sub-00/dwi/sub-00_dwi.nii.gz")
        deriv = os.path.join(tmp, "deriv")

        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [str(root), deriv],
            bids=True,
            pipeline_type="denoise",
            stop_on_error=True,
            dry_run=True,
        )

        summary = _summary(deriv)
        npt.assert_equal(len(summary["runs"]), 1)
        npt.assert_equal(summary["runs"][0]["status"], "failed")


def test_auto_bids_conflicting_inputs():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        paths = _phantom(tmp)

        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [root, paths["dwi"]],
            bids=True,
            pipeline_type="denoise",
            dry_run=True,
        )

        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [root],
            bids=True,
            t1_file=paths["dwi"],
            pipeline_type="denoise",
            dry_run=True,
        )


def test_auto_bids_rejects_group_level():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [root, os.path.join(tmp, "deriv"), "group"],
            bids=True,
            pipeline_type="denoise",
            dry_run=True,
        )


def test_auto_detects_bids_without_flag():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [root],
            pipeline_type="denoise",
            dry_run=True,
        )


def test_auto_bids_requires_a_dataset():
    with TemporaryDirectory() as tmp:
        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [os.path.join(tmp, "not_a_dataset.toml")],
            bids=True,
            dry_run=True,
        )


def test_auto_bids_missing_dataset_description():
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "ds"
        _write_nifti(root / "sub-01/dwi/sub-01_dwi.nii.gz")
        _write_gradients(str(root / "sub-01/dwi/sub-01_dwi"))

        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [str(root), os.path.join(tmp, "deriv")],
            bids=True,
            pipeline_type="denoise",
            dry_run=True,
        )


def test_auto_bids_smoke():
    with TemporaryDirectory() as tmp:
        root = _phantom_bids_tree(Path(tmp) / "ds")
        deriv = os.path.join(tmp, "deriv")

        AutoFlow().run([root, deriv, "participant"], bids=True, pipeline_type="denoise")

        summary = _summary(deriv)
        npt.assert_equal([r["status"] for r in summary["runs"]], ["completed"])

        run_dir = Path(summary["runs"][0]["out_dir"])
        npt.assert_equal((run_dir / "reports" / "index.html").exists(), True)
        npt.assert_equal((run_dir / "dwi_nlmeans.nii.gz").exists(), True)

        # headline outputs are exposed under BIDS-style names
        link = run_dir / "sub-01_desc-denoised_dwi.nii.gz"
        npt.assert_equal(link.exists(), True)
        with open(run_dir / "sub-01_desc-denoised_dwi.json") as f:
            sidecar = json.load(f)
        npt.assert_equal(sidecar["GeneratedByStage"]["name"], "denoise")
        npt.assert_equal(sidecar["RawSources"][0].startswith("bids::"), True)

        names = [f for f in os.listdir(deriv) if f.endswith("_batch_summary.html")]
        npt.assert_equal(len(names), 1)


if __name__ == "__main__":
    pytest.main([__file__])
