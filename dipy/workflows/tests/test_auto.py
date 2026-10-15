import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib

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
            pipeline_type="dti_only",
            dry_run=True,
            out_dir=out_dir,
        )

        configs = [f for f in os.listdir(out_dir) if f.endswith("_pipeline.toml")]
        npt.assert_equal(len(configs), 1)
        npt.assert_equal(
            any(f.endswith("_dipy_auto.log") for f in os.listdir(out_dir)), True
        )


def test_auto_releases_log_files():
    handlers = list(logging.getLogger("dipy").handlers)

    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        inputs = [paths["dwi"], paths["bvals"], paths["bvecs"]]
        out_dir = os.path.join(tmp, "out")

        AutoFlow().run(inputs, pipeline_type="dti_only", dry_run=True, out_dir=out_dir)
        npt.assert_equal(logging.getLogger("dipy").handlers, handlers)

        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            inputs[:1],
            pipeline_type="dti_only",
            dry_run=True,
            out_dir=out_dir,
        )
        npt.assert_equal(logging.getLogger("dipy").handlers, handlers)

        root = _make_bids_tree(Path(tmp) / "ds")
        AutoFlow().run(
            [root, os.path.join(tmp, "deriv")],
            bids=True,
            participant_label=["01"],
            pipeline_type="dti_only",
            dry_run=True,
        )
        npt.assert_equal(logging.getLogger("dipy").handlers, handlers)


def test_auto_list_pipelines():
    with TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "out")
        npt.assert_equal(AutoFlow().run([], list_pipelines=True, out_dir=out_dir), None)
        npt.assert_equal(os.path.exists(out_dir), False)


def test_auto_unknown_pipeline_type():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [paths["dwi"], paths["bvals"], paths["bvecs"]],
            pipeline_type="does_not_exist",
            dry_run=True,
            out_dir=os.path.join(tmp, "out"),
        )


def test_auto_config_file():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        out_dir = os.path.join(tmp, "out")
        config_file = os.path.join(tmp, "custom.toml")
        Path(config_file).write_text(
            "[General]\n"
            'name = "custom_denoise"\n\n'
            "[[pipeline]]\n"
            'name = "denoise"\n'
            'cli = "dipy_denoise_nlmeans"\n'
            'input_files = "${io.dwi}"\n'
        )

        AutoFlow().run(
            [config_file, paths["dwi"], paths["bvals"], paths["bvecs"]],
            dry_run=True,
            out_dir=out_dir,
        )

        # the user's configuration is used in place, not copied
        npt.assert_equal(
            [f for f in os.listdir(out_dir) if f.endswith("_pipeline.toml")], []
        )
        npt.assert_equal(
            any(f.endswith("_dipy_auto.log") for f in os.listdir(out_dir)), True
        )


def test_auto_config_file_left_untouched_by_conflicts():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        out_dir = os.path.join(tmp, "out")
        config_file = os.path.join(tmp, "custom.toml")
        text = (
            "[[pipeline]]\n"
            'name = "d1"\ncli = "dipy_denoise_nlmeans"\ninput_files = "${io.dwi}"\n'
            "\n[[pipeline]]\n"
            'name = "d2"\ncli = "dipy_denoise_nlmeans"\ninput_files = "${io.dwi}"\n'
        )
        Path(config_file).write_text(text)

        AutoFlow().run(
            [config_file, paths["dwi"], paths["bvals"], paths["bvecs"]],
            out_dir=out_dir,
        )

        npt.assert_equal(Path(config_file).read_text(), text)
        resolved = os.path.join(out_dir, "custom_resolved.toml")
        with open(resolved, "rb") as f:
            stages = tomllib.load(f)["pipeline"]
        npt.assert_equal(
            [s["out_denoised"] for s in stages],
            ["denoised_d1.nii.gz", "denoised_d2.nii.gz"],
        )
        reports = [
            f for f in os.listdir(out_dir) if f.endswith("_pipeline_report.html")
        ]
        npt.assert_equal(len(reports), 1)
        npt.assert_equal(resolved in Path(out_dir, reports[0]).read_text(), True)


def test_auto_uses_the_configured_output_directory():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        out_dir = os.path.join(tmp, "from_config")
        config_file = os.path.join(tmp, "custom.toml")
        Path(config_file).write_text(
            f'[io]\nout_dir = "{out_dir}"\n\n'
            '[[pipeline]]\nname = "denoise"\ncli = "dipy_denoise_nlmeans"\n'
            'input_files = "${io.dwi}"\n'
        )

        AutoFlow().run(
            [config_file, paths["dwi"], paths["bvals"], paths["bvecs"]], dry_run=True
        )

        npt.assert_equal(
            any(f.endswith("_dipy_auto.log") for f in os.listdir(out_dir)), True
        )


def test_auto_invalid_config_file():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        inputs = [paths["dwi"], paths["bvals"], paths["bvecs"]]
        out_dir = os.path.join(tmp, "out")

        broken = os.path.join(tmp, "broken.toml")
        Path(broken).write_text("[[pipeline]\nname = ")
        no_stage = os.path.join(tmp, "no_stage.toml")
        Path(no_stage).write_text('[General]\nname = "empty"\n')

        for config_file in (os.path.join(tmp, "missing.toml"), broken, no_stage):
            npt.assert_raises(
                SystemExit,
                AutoFlow().run,
                [config_file, *inputs],
                dry_run=True,
                out_dir=out_dir,
            )


def test_auto_missing_inputs():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        out_dir = os.path.join(tmp, "out")

        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [paths["dwi"]],
            pipeline_type="dti_only",
            dry_run=True,
            out_dir=out_dir,
        )
        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [os.path.join(tmp, "nope.nii.gz"), paths["bvals"], paths["bvecs"]],
            pipeline_type="dti_only",
            dry_run=True,
            out_dir=out_dir,
        )


def test_auto_rejects_invalid_mask():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        mask_file = os.path.join(tmp, "mask.nii.gz")
        _write_nifti(mask_file)

        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [paths["dwi"], paths["bvals"], paths["bvecs"]],
            mask_file=mask_file,
            pipeline_type="dti_only",
            dry_run=True,
            out_dir=os.path.join(tmp, "out"),
        )


def test_auto_interactive_requires_bvals():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [paths["dwi"]],
            interactive_mode=True,
            dry_run=True,
            out_dir=os.path.join(tmp, "out"),
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
            pipeline_type="dti_only",
            dry_run=True,
        )

        with open(os.path.join(deriv, "dataset_description.json")) as f:
            description = json.load(f)
        npt.assert_equal(description["DatasetType"], "derivative")
        npt.assert_equal(description["GeneratedBy"][0]["Name"], "dipy")

        summary = _summary(deriv)
        npt.assert_equal(len(summary["runs"]), 4)
        npt.assert_equal(summary["pipeline"], "dti_only")
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


def test_auto_bids_runs_do_not_inherit_inputs():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        deriv = os.path.join(tmp, "deriv")
        config_file = os.path.join(tmp, "custom.toml")
        Path(config_file).write_text(
            "[io]\n"
            f'session = "9"\nt1w = "{os.path.join(tmp, "other_T1w.nii.gz")}"\n'
            f'dwi_opp_phase = "{os.path.join(tmp, "other_epi.nii.gz")}"\n\n'
            '[[pipeline]]\nname = "denoise"\ncli = "dipy_denoise_nlmeans"\n'
            'input_files = "${io.dwi}"\n'
        )

        AutoFlow().run(
            [config_file, root, deriv],
            bids=True,
            participant_label=["01", "02"],
            dry_run=True,
        )

        io_sections = {}
        for run in _summary(deriv)["runs"]:
            name = next(
                f for f in os.listdir(run["out_dir"]) if f.endswith("_pipeline.toml")
            )
            with open(os.path.join(run["out_dir"], name), "rb") as f:
                io_sections[run["label"]] = tomllib.load(f)["io"]

        # sub-01 has no session and no fieldmap: the configured ones are cleared
        npt.assert_equal(io_sections["sub-01"]["session"], "")
        npt.assert_equal(io_sections["sub-01"]["dwi_opp_phase"], "")
        npt.assert_equal(
            io_sections["sub-01"]["t1w"],
            os.path.join(root, "sub-01", "anat", "sub-01_T1w.nii.gz"),
        )
        npt.assert_equal(io_sections["sub-02_ses-2"]["session"], "2")


def test_auto_bids_unknown_start_stage():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        deriv = os.path.join(tmp, "deriv")

        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [root, deriv],
            bids=True,
            participant_label=["01"],
            pipeline_type="dti_only",
            start="dti_fitt",
            dry_run=True,
        )
        # rejected once, before any acquisition is processed
        npt.assert_equal(
            [f for f in os.listdir(deriv) if f.endswith("_batch_summary.json")]
            if os.path.isdir(deriv)
            else [],
            [],
        )


def test_auto_bids_default_derivatives_root():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")

        AutoFlow().run(
            [root],
            bids=True,
            participant_label=["01", "02"],
            pipeline_type="dti_only",
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
            pipeline_type="dti_only",
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
            pipeline_type="dti_only",
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
            pipeline_type="dti_only",
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
            pipeline_type="dti_only",
            dry_run=True,
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
            pipeline_type="dti_only",
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
            pipeline_type="dti_only",
            dry_run=True,
        )

        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [root],
            bids=True,
            t1_file=paths["dwi"],
            pipeline_type="dti_only",
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
            pipeline_type="dti_only",
            dry_run=True,
        )


def test_auto_detects_bids_without_flag():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        npt.assert_raises(
            SystemExit,
            AutoFlow().run,
            [root],
            pipeline_type="dti_only",
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
            pipeline_type="dti_only",
            dry_run=True,
        )


def test_auto_bids_smoke():
    with TemporaryDirectory() as tmp:
        root = _phantom_bids_tree(Path(tmp) / "ds")
        deriv = os.path.join(tmp, "deriv")

        AutoFlow().run(
            [root, deriv, "participant"], bids=True, pipeline_type="dti_only"
        )

        summary = _summary(deriv)
        npt.assert_equal([r["status"] for r in summary["runs"]], ["completed"])

        run_dir = Path(summary["runs"][0]["out_dir"])
        npt.assert_equal((run_dir / "reports" / "index.html").exists(), True)
        npt.assert_equal((run_dir / "dwi_denoised.nii.gz").exists(), True)

        # headline outputs are exposed under BIDS-style names
        link = run_dir / "sub-01_desc-denoised_dwi.nii.gz"
        npt.assert_equal(link.exists(), True)
        with open(run_dir / "sub-01_desc-denoised_dwi.json") as f:
            sidecar = json.load(f)
        npt.assert_equal(sidecar["GeneratedByStage"]["name"], "denoise")
        npt.assert_equal(
            sidecar["RawSources"], ["bids:raw:sub-01/dwi/sub-01_dwi.nii.gz"]
        )

        names = [f for f in os.listdir(deriv) if f.endswith("_batch_summary.html")]
        npt.assert_equal(len(names), 1)


if __name__ == "__main__":
    pytest.main([__file__])
