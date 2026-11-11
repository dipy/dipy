import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import numpy.testing as npt
import pytest

from dipy.io.bids import (
    DWIRun,
    build_entity_name,
    derivative_path,
    find_dwi_runs,
    find_opposite_pe_fmap,
    find_t1w,
    inherited_files,
    list_sessions,
    list_subjects,
    parse_entities,
    phase_encoding_info,
    read_sidecar,
    validate_dataset,
    write_dataset_description,
)
from dipy.io.image import save_nifti


def _write_nifti(path):
    """Write a tiny 4D volume, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_nifti(str(path), np.zeros((4, 4, 4, 6), dtype=np.float32), np.eye(4))


def _write_gradients(stem):
    """Write ``<stem>.bval`` and ``<stem>.bvec`` beside a DWI file."""
    bvals = np.array([0, 1000, 1000, 1000, 1000, 1000], dtype=float)
    bvecs = np.eye(3)[[0, 0, 1, 2, 0, 1]]
    np.savetxt(f"{stem}.bval", bvals[None], fmt="%d")
    np.savetxt(f"{stem}.bvec", bvecs.T, fmt="%.6f")


def _write_json(path, content):
    """Write a JSON sidecar, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f)


def _make_bids_tree(root):
    """Build a small BIDS dataset covering the cases the reader must handle.

    Contains an unsessioned subject, a two-session subject with two runs, an
    opposite phase-encoding fieldmap declaring ``IntendedFor``, root and
    subject level inherited sidecars, and a subject without gradient files.
    """
    root = Path(root)
    _write_json(
        root / "dataset_description.json", {"Name": "test", "BIDSVersion": "1.9.0"}
    )
    _write_json(root / "dwi.json", {"Manufacturer": "GE", "EchoTime": 0.1})

    _write_nifti(root / "sub-01/dwi/sub-01_dwi.nii.gz")
    _write_gradients(str(root / "sub-01/dwi/sub-01_dwi"))
    _write_json(
        root / "sub-01/dwi/sub-01_dwi.json",
        {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05},
    )
    _write_nifti(root / "sub-01/anat/sub-01_T1w.nii.gz")

    _write_json(
        root / "sub-02/sub-02_dwi.json",
        {"Manufacturer": "Siemens", "PhaseEncodingDirection": "j"},
    )
    _write_nifti(root / "sub-02/ses-1/anat/sub-02_ses-1_T1w.nii.gz")
    for run in (1, 2):
        stem = root / f"sub-02/ses-1/dwi/sub-02_ses-1_acq-b1000_dir-AP_run-{run}_dwi"
        _write_nifti(f"{stem}.nii.gz")
        _write_gradients(str(stem))
    _write_json(
        root / "sub-02/ses-1/dwi/sub-02_ses-1_acq-b1000_dir-AP_run-1_dwi.json",
        {"TotalReadoutTime": 0.07},
    )
    _write_nifti(root / "sub-02/ses-1/fmap/sub-02_ses-1_acq-b1000_dir-PA_epi.nii.gz")
    _write_json(
        root / "sub-02/ses-1/fmap/sub-02_ses-1_acq-b1000_dir-PA_epi.json",
        {
            "PhaseEncodingDirection": "j-",
            "IntendedFor": ["ses-1/dwi/sub-02_ses-1_acq-b1000_dir-AP_run-1_dwi.nii.gz"],
        },
    )
    _write_nifti(root / "sub-02/ses-2/dwi/sub-02_ses-2_dwi.nii.gz")
    _write_gradients(str(root / "sub-02/ses-2/dwi/sub-02_ses-2_dwi"))

    _write_nifti(root / "sub-03/dwi/sub-03_dwi.nii.gz")

    return str(root)


def test_parse_entities():
    entities = parse_entities(filename="sub-01_ses-2_acq-b1000_dir-AP_run-3_dwi.nii.gz")
    npt.assert_equal(entities["sub"], "01")
    npt.assert_equal(entities["ses"], "2")
    npt.assert_equal(entities["acq"], "b1000")
    npt.assert_equal(entities["dir"], "AP")
    npt.assert_equal(entities["run"], "3")
    npt.assert_equal(entities["suffix"], "dwi")
    npt.assert_equal(entities["extension"], ".nii.gz")

    plain = parse_entities(filename="/data/sub-01/dwi/sub-01_dwi.nii")
    npt.assert_equal(plain["extension"], ".nii")
    npt.assert_equal(plain["suffix"], "dwi")
    npt.assert_equal("ses" in plain, False)

    unknown = parse_entities(filename="sub-01_weird-value_dwi.json")
    npt.assert_equal(unknown["weird"], "value")
    npt.assert_equal(unknown["extension"], ".json")


def test_build_entity_name():
    npt.assert_equal(
        build_entity_name(
            entities={"desc": "FA", "sub": "01", "ses": "1"},
            suffix="dwimap",
            extension=".nii.gz",
        ),
        "sub-01_ses-1_desc-FA_dwimap.nii.gz",
    )

    # entities of a parsed name round-trip, suffix and extension excluded
    name = "sub-02_ses-1_acq-b1000_dir-AP_run-1_dwi.nii.gz"
    entities = parse_entities(filename=name)
    npt.assert_equal(
        build_entity_name(entities=entities, suffix="dwi", extension=".nii.gz"), name
    )

    npt.assert_equal(
        build_entity_name(
            entities={"sub": "01", "zzz": "1"}, suffix="mask", extension=".nii.gz"
        ),
        "sub-01_zzz-1_mask.nii.gz",
    )


def test_validate_dataset():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        description = validate_dataset(root=root)
        npt.assert_equal(description["Name"], "test")

        npt.assert_raises(ValueError, validate_dataset, root=Path(tmp) / "missing")

        no_description = Path(tmp) / "nodesc"
        (no_description / "sub-01").mkdir(parents=True)
        npt.assert_raises(ValueError, validate_dataset, root=no_description)
        npt.assert_equal(
            validate_dataset(root=no_description, require_description=False), {}
        )

        no_subject = Path(tmp) / "nosub"
        no_subject.mkdir()
        _write_json(no_subject / "dataset_description.json", {"Name": "x"})
        npt.assert_raises(ValueError, validate_dataset, root=no_subject)

        malformed = Path(tmp) / "bad"
        (malformed / "sub-01").mkdir(parents=True)
        (malformed / "dataset_description.json").write_text(
            "{not json", encoding="utf-8"
        )
        npt.assert_raises(ValueError, validate_dataset, root=malformed)

        as_file = Path(tmp) / "file.txt"
        as_file.write_text("x", encoding="utf-8")
        npt.assert_raises(ValueError, validate_dataset, root=as_file)


def test_list_subjects_and_sessions():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        npt.assert_equal(list_subjects(root=root), ["01", "02", "03"])
        npt.assert_equal(list_sessions(root=root, subject="01"), [])
        npt.assert_equal(list_sessions(root=root, subject="02"), ["1", "2"])
        npt.assert_equal(list_sessions(root=root, subject="sub-02"), ["1", "2"])


def test_inherited_files():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        dwi = os.path.join(root, "sub-01/dwi/sub-01_dwi.nii.gz")
        chain = inherited_files(root=root, path=dwi, suffix="dwi", extension=".json")
        npt.assert_equal(len(chain), 2)
        npt.assert_equal(Path(chain[0]).name, "sub-01_dwi.json")
        npt.assert_equal(Path(chain[1]).name, "dwi.json")

        # a session-specific sidecar does not apply to another session
        other = os.path.join(root, "sub-02/ses-2/dwi/sub-02_ses-2_dwi.nii.gz")
        names = [
            Path(path).name
            for path in inherited_files(
                root=root, path=other, suffix="dwi", extension=".json"
            )
        ]
        npt.assert_equal(names, ["sub-02_dwi.json", "dwi.json"])


def test_read_sidecar_inheritance():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")

        sidecar, metadata = read_sidecar(
            root=root, path=os.path.join(root, "sub-01/dwi/sub-01_dwi.nii.gz")
        )
        npt.assert_equal(Path(sidecar).name, "sub-01_dwi.json")
        # inherited from the dataset root
        npt.assert_equal(metadata["Manufacturer"], "GE")
        # nearest file wins
        npt.assert_equal(metadata["PhaseEncodingDirection"], "j-")
        npt.assert_equal(metadata["TotalReadoutTime"], 0.05)

        _, run1 = read_sidecar(
            root=root,
            path=os.path.join(
                root, "sub-02/ses-1/dwi/sub-02_ses-1_acq-b1000_dir-AP_run-1_dwi.nii.gz"
            ),
        )
        npt.assert_equal(run1["Manufacturer"], "Siemens")
        npt.assert_equal(run1["PhaseEncodingDirection"], "j")
        npt.assert_equal(run1["TotalReadoutTime"], 0.07)
        npt.assert_equal(run1["EchoTime"], 0.1)

        _, run2 = read_sidecar(
            root=root,
            path=os.path.join(
                root, "sub-02/ses-1/dwi/sub-02_ses-1_acq-b1000_dir-AP_run-2_dwi.nii.gz"
            ),
        )
        npt.assert_equal("TotalReadoutTime" in run2, False)

        _, none = read_sidecar(
            root=root, path=os.path.join(root, "sub-03/dwi/sub-03_dwi.nii.gz")
        )
        npt.assert_equal(none["Manufacturer"], "GE")


def test_find_dwi_runs_and_filters():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")

        runs = find_dwi_runs(root=root)
        npt.assert_equal(len(runs), 5)
        npt.assert_equal([run.subject for run in runs], ["01", "02", "02", "02", "03"])
        npt.assert_equal([run.session for run in runs], [None, "1", "1", "2", None])

        npt.assert_equal(len(find_dwi_runs(root=root, participant_label=["02"])), 3)
        npt.assert_equal(len(find_dwi_runs(root=root, participant_label=["sub-02"])), 3)
        npt.assert_equal(
            len(
                find_dwi_runs(root=root, participant_label=["02"], session_label=["1"])
            ),
            2,
        )
        npt.assert_equal(len(find_dwi_runs(root=root, acquisition_label=["b1000"])), 2)
        npt.assert_equal(len(find_dwi_runs(root=root, run_label=["2"])), 1)
        npt.assert_equal(find_dwi_runs(root=root, participant_label=["99"]), [])


def test_find_dwi_runs_gradients():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")

        incomplete = [run for run in find_dwi_runs(root=root) if run.subject == "03"]
        npt.assert_equal(len(incomplete), 1)
        npt.assert_equal(incomplete[0].bval, None)
        npt.assert_equal(incomplete[0].bvec, None)
        npt.assert_equal(incomplete[0].is_complete, False)

        complete = find_dwi_runs(root=root, require_gradients=True)
        npt.assert_equal(len(complete), 4)
        npt.assert_equal(all(run.is_complete for run in complete), True)
        npt.assert_equal(Path(complete[0].bval).name, "sub-01_dwi.bval")


def test_run_labels_and_paths():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        runs = {run.label: run for run in find_dwi_runs(root=root)}

        npt.assert_equal("sub-01" in runs, True)
        npt.assert_equal("sub-02_ses-1_acq-b1000_dir-AP_run-1" in runs, True)
        npt.assert_equal(runs["sub-01"].relative_dir, os.path.join("sub-01", "dwi"))
        npt.assert_equal(
            runs["sub-02_ses-2"].relative_dir,
            os.path.join("sub-02", "ses-2", "dwi"),
        )


def test_find_t1w():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")

        unsessioned = find_t1w(root=root, entities={"sub": "01"})
        npt.assert_equal(Path(unsessioned).name, "sub-01_T1w.nii.gz")

        same_session = find_t1w(root=root, entities={"sub": "02", "ses": "1"})
        npt.assert_equal(Path(same_session).name, "sub-02_ses-1_T1w.nii.gz")

        # ses-2 has no anat of its own: the other session is used as fallback
        fallback = find_t1w(root=root, entities={"sub": "02", "ses": "2"})
        npt.assert_equal(Path(fallback).name, "sub-02_ses-1_T1w.nii.gz")

        npt.assert_equal(find_t1w(root=root, entities={"sub": "03"}), None)
        npt.assert_equal(find_t1w(root=root, entities={}), None)


def test_find_t1w_prefers_plain_name():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        _write_nifti(Path(root) / "sub-01/anat/sub-01_desc-preproc_T1w.nii.gz")
        chosen = find_t1w(root=root, entities={"sub": "01"})
        npt.assert_equal(Path(chosen).name, "sub-01_T1w.nii.gz")


def test_find_opposite_pe_fmap():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        dwi = os.path.join(
            root, "sub-02/ses-1/dwi/sub-02_ses-1_acq-b1000_dir-AP_run-1_dwi.nii.gz"
        )
        entities = parse_entities(filename=dwi)
        _, metadata = read_sidecar(root=root, path=dwi)

        fmap, fmap_metadata = find_opposite_pe_fmap(
            root=root, dwi_path=dwi, dwi_metadata=metadata, entities=entities
        )
        npt.assert_equal(Path(fmap).name, "sub-02_ses-1_acq-b1000_dir-PA_epi.nii.gz")
        npt.assert_equal(fmap_metadata["PhaseEncodingDirection"], "j-")

        # run-2 is not in IntendedFor: the opposite dir- entity rule applies
        other = os.path.join(
            root, "sub-02/ses-1/dwi/sub-02_ses-1_acq-b1000_dir-AP_run-2_dwi.nii.gz"
        )
        fmap, _ = find_opposite_pe_fmap(
            root=root,
            dwi_path=other,
            dwi_metadata=read_sidecar(root=root, path=other)[1],
            entities=parse_entities(filename=other),
        )
        npt.assert_equal(Path(fmap).name, "sub-02_ses-1_acq-b1000_dir-PA_epi.nii.gz")

        # no fmap directory at all
        plain = os.path.join(root, "sub-01/dwi/sub-01_dwi.nii.gz")
        fmap, fmap_metadata = find_opposite_pe_fmap(
            root=root,
            dwi_path=plain,
            dwi_metadata=read_sidecar(root=root, path=plain)[1],
            entities=parse_entities(filename=plain),
        )
        npt.assert_equal(fmap, None)
        npt.assert_equal(fmap_metadata, {})


def test_find_opposite_pe_fmap_by_phase_encoding():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        # an fmap with neither IntendedFor nor a dir- entity
        _write_nifti(Path(root) / "sub-01/fmap/sub-01_epi.nii.gz")
        _write_json(
            Path(root) / "sub-01/fmap/sub-01_epi.json",
            {"PhaseEncodingDirection": "j"},
        )
        dwi = os.path.join(root, "sub-01/dwi/sub-01_dwi.nii.gz")
        fmap, metadata = find_opposite_pe_fmap(
            root=root,
            dwi_path=dwi,
            dwi_metadata=read_sidecar(root=root, path=dwi)[1],
            entities=parse_entities(filename=dwi),
        )
        npt.assert_equal(Path(fmap).name, "sub-01_epi.nii.gz")
        npt.assert_equal(metadata["PhaseEncodingDirection"], "j")


def test_phase_encoding_info():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        run = next(r for r in find_dwi_runs(root=root) if r.subject == "01")
        info = phase_encoding_info(run=run)
        npt.assert_equal(info["PhaseEncodingDirection"], "j-")
        npt.assert_equal(info["TotalReadoutTime"], 0.05)
        npt.assert_equal(info["EchoTime"], 0.1)
        npt.assert_equal("EffectiveEchoSpacing" in info, False)

        empty = phase_encoding_info(
            run=DWIRun(root=root, dwi="x.nii.gz", entities={"sub": "01"})
        )
        npt.assert_equal(empty, {})


def test_derivative_path():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        deriv = os.path.join(tmp, "deriv")
        runs = {run.label: run for run in find_dwi_runs(root=root)}

        npt.assert_equal(
            derivative_path(deriv_root=deriv, run=runs["sub-01"]),
            os.path.join(deriv, "sub-01", "dwi"),
        )
        npt.assert_equal(
            derivative_path(deriv_root=deriv, run=runs["sub-02_ses-2"]),
            os.path.join(deriv, "sub-02", "ses-2", "dwi"),
        )
        npt.assert_equal(
            derivative_path(
                deriv_root=deriv, run=runs["sub-02_ses-2"], datatype="anat"
            ),
            os.path.join(deriv, "sub-02", "ses-2", "anat"),
        )


def test_write_dataset_description():
    with TemporaryDirectory() as tmp:
        deriv = os.path.join(tmp, "deriv")
        target = write_dataset_description(
            path=deriv, source_datasets=[tmp], pipeline_name="denoise"
        )
        with open(target, encoding="utf-8") as f:
            content = json.load(f)

        npt.assert_equal(content["Name"], "dipy")
        npt.assert_equal(content["DatasetType"], "derivative")
        npt.assert_equal(content["GeneratedBy"][0]["Name"], "dipy")
        npt.assert_equal("denoise" in content["GeneratedBy"][0]["Description"], True)
        npt.assert_equal(len(content["SourceDatasets"]), 1)
        npt.assert_equal(
            content["SourceDatasets"][0]["URL"].startswith("file://"), True
        )

        # the written description is itself a readable BIDS description
        Path(deriv, "sub-01").mkdir()
        npt.assert_equal(validate_dataset(root=deriv)["DatasetType"], "derivative")


def test_validate_dataset_ignores_sub_files():
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "ds"
        root.mkdir()
        _write_json(root / "dataset_description.json", {"Name": "x"})
        (root / "sub-01.tsv").write_text("x", encoding="utf-8")
        npt.assert_raises(ValueError, validate_dataset, root=root)


def test_find_dwi_runs_glob_characters_in_root():
    with TemporaryDirectory() as tmp:
        # Windows forbids * and ? in file names; [ ] still form a glob pattern
        name = "ds[1]" if os.name == "nt" else "ds[1]*?"
        root = _make_bids_tree(Path(tmp) / name)
        runs = find_dwi_runs(root=root)
        npt.assert_equal(len(runs), 5)
        npt.assert_equal(Path(runs[0].bval).name, "sub-01_dwi.bval")


def test_find_dwi_runs_nii_and_nii_gz_duplicates(caplog):
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        _write_nifti(Path(root) / "sub-01/dwi/sub-01_dwi.nii")
        runs = [run for run in find_dwi_runs(root=root) if run.subject == "01"]
        npt.assert_equal(len(runs), 1)
        npt.assert_equal(Path(runs[0].dwi).name, "sub-01_dwi.nii.gz")
        npt.assert_equal("ignoring" in caplog.text, True)

        # a lone .nii file is still found
        os.remove(os.path.join(root, "sub-01/dwi/sub-01_dwi.nii.gz"))
        runs = [run for run in find_dwi_runs(root=root) if run.subject == "01"]
        npt.assert_equal(len(runs), 1)
        npt.assert_equal(Path(runs[0].dwi).name, "sub-01_dwi.nii")


def test_find_dwi_runs_label_filters():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        npt.assert_equal(len(find_dwi_runs(root=root, run_label=["01"])), 1)
        npt.assert_equal(len(find_dwi_runs(root=root, run_label=["run-001"])), 1)
        npt.assert_equal(len(find_dwi_runs(root=root, run_label=["3"])), 0)
        # subject labels are not numbers: 2 does not select sub-02
        npt.assert_equal(len(find_dwi_runs(root=root, participant_label=["2"])), 0)
        # a bare string is one label, not a sequence of characters
        npt.assert_equal(len(find_dwi_runs(root=root, participant_label="02")), 3)


def test_find_dwi_runs_numeric_run_order():
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "ds"
        _write_json(root / "dataset_description.json", {"Name": "x"})
        for run in (10, 2):
            _write_nifti(root / f"sub-01/dwi/sub-01_run-{run}_dwi.nii.gz")
        runs = find_dwi_runs(root=root)
        npt.assert_equal([run.entities["run"] for run in runs], ["2", "10"])


def test_inherited_files_most_specific_wins():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        dwi_dir = Path(root) / "sub-01/dwi"
        stem = dwi_dir / "sub-01_run-1_dwi"
        _write_nifti(f"{stem}.nii.gz")
        np.savetxt(f"{stem}.bval", np.array([[0, 500, 500, 500, 500, 500]]), fmt="%d")
        _write_json(f"{stem}.json", {"TotalReadoutTime": 0.09})

        chain = inherited_files(
            root=root, path=f"{stem}.nii.gz", suffix="dwi", extension=".json"
        )
        npt.assert_equal(
            [Path(path).name for path in chain],
            ["sub-01_run-1_dwi.json", "sub-01_dwi.json", "dwi.json"],
        )

        sidecar, metadata = read_sidecar(root=root, path=f"{stem}.nii.gz")
        npt.assert_equal(Path(sidecar).name, "sub-01_run-1_dwi.json")
        npt.assert_equal(metadata["TotalReadoutTime"], 0.09)
        npt.assert_equal(metadata["PhaseEncodingDirection"], "j-")

        run = next(r for r in find_dwi_runs(root=root) if r.entities.get("run") == "1")
        npt.assert_equal(Path(run.bval).name, "sub-01_run-1_dwi.bval")
        # the generic bvec still applies through inheritance
        npt.assert_equal(Path(run.bvec).name, "sub-01_dwi.bvec")


def test_find_t1w_other_session_warns(caplog):
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        find_t1w(root=root, entities={"sub": "02", "ses": "1"})
        npt.assert_equal("No T1w in" in caplog.text, False)
        fallback = find_t1w(root=root, entities={"sub": "02", "ses": "2"})
        npt.assert_equal(Path(fallback).name, "sub-02_ses-1_T1w.nii.gz")
        npt.assert_equal("No T1w in sub-02/ses-2" in caplog.text, True)


def _make_fmap_case(*, root, intended_for):
    """Add sub-04 with a dir-AP DWI and a dir-PA, opposite PE fieldmap."""
    root = Path(root)
    dwi = root / "sub-04/dwi/sub-04_dir-AP_dwi.nii.gz"
    _write_nifti(dwi)
    _write_json(
        root / "sub-04/dwi/sub-04_dir-AP_dwi.json", {"PhaseEncodingDirection": "j-"}
    )
    _write_nifti(root / "sub-04/fmap/sub-04_dir-PA_epi.nii.gz")
    content = {"PhaseEncodingDirection": "j"}
    if intended_for is not None:
        content["IntendedFor"] = intended_for
    _write_json(root / "sub-04/fmap/sub-04_dir-PA_epi.json", content)
    return str(dwi)


def _fmap_of(*, root, dwi):
    """Return the fieldmap chosen for ``dwi``, resolving its own metadata."""
    return find_opposite_pe_fmap(
        root=root,
        dwi_path=dwi,
        dwi_metadata=read_sidecar(root=root, path=dwi)[1],
        entities=parse_entities(filename=dwi),
    )[0]


def test_find_opposite_pe_fmap_skips_other_modality():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        dwi = _make_fmap_case(
            root=root, intended_for=["bids::sub-04/func/sub-04_task-rest_bold.nii.gz"]
        )
        npt.assert_equal(_fmap_of(root=root, dwi=dwi), None)

    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        dwi = _make_fmap_case(root=root, intended_for=None)
        fmap = _fmap_of(root=root, dwi=dwi)
        npt.assert_equal(Path(fmap).name, "sub-04_dir-PA_epi.nii.gz")

    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        dwi = _make_fmap_case(
            root=root, intended_for="bids:raw:sub-04/dwi/sub-04_dir-AP_dwi.nii.gz"
        )
        fmap = _fmap_of(root=root, dwi=dwi)
        npt.assert_equal(Path(fmap).name, "sub-04_dir-PA_epi.nii.gz")


def test_fmap_of_other_run_is_a_warned_last_resort(caplog):
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        run_2 = os.path.join(
            root, "sub-02/ses-1/dwi/sub-02_ses-1_acq-b1000_dir-AP_run-2_dwi.nii.gz"
        )
        npt.assert_equal(
            Path(_fmap_of(root=root, dwi=run_2)).name,
            "sub-02_ses-1_acq-b1000_dir-PA_epi.nii.gz",
        )
        npt.assert_equal("IntendedFor other diffusion runs" in caplog.text, True)

        caplog.clear()
        fmap_dir = Path(root) / "sub-02/ses-1/fmap"
        _write_nifti(fmap_dir / "sub-02_ses-1_acq-b1000_dir-PA_run-2_epi.nii.gz")
        # an empty IntendedFor, as the run-1 one would otherwise be inherited
        _write_json(
            fmap_dir / "sub-02_ses-1_acq-b1000_dir-PA_run-2_epi.json",
            {"PhaseEncodingDirection": "j-", "IntendedFor": []},
        )
        npt.assert_equal(
            Path(_fmap_of(root=root, dwi=run_2)).name,
            "sub-02_ses-1_acq-b1000_dir-PA_run-2_epi.nii.gz",
        )
        npt.assert_equal("IntendedFor other diffusion runs" in caplog.text, False)

        run_1 = run_2.replace("run-2", "run-1")
        npt.assert_equal(
            Path(_fmap_of(root=root, dwi=run_1)).name,
            "sub-02_ses-1_acq-b1000_dir-PA_epi.nii.gz",
        )


def test_non_ascii_digit_run_labels():
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "ds"
        _write_json(root / "dataset_description.json", {"Name": "x"})
        for run in ("2", "\u00b2", "10"):
            _write_nifti(root / f"sub-01/dwi/sub-01_run-{run}_dwi.nii.gz")
        runs = find_dwi_runs(root=root)
        npt.assert_equal([run.entities["run"] for run in runs], ["2", "10", "\u00b2"])
        selected = find_dwi_runs(root=root, run_label=["\u00b2", "02"])
        npt.assert_equal(
            sorted(run.entities["run"] for run in selected), ["2", "\u00b2"]
        )


def test_dwi_run_is_hashable():
    with TemporaryDirectory() as tmp:
        root = _make_bids_tree(Path(tmp) / "ds")
        runs = find_dwi_runs(root=root)
        npt.assert_equal(len(set(runs)), len(runs))
        again = find_dwi_runs(root=root)
        npt.assert_equal(runs[0] == again[0], True)
        npt.assert_equal(hash(runs[0]), hash(again[0]))
        npt.assert_equal({run: run.label for run in runs}[again[0]], runs[0].label)


def test_write_dataset_description_links():
    with TemporaryDirectory() as tmp:
        source = Path(tmp) / "raw data"
        source.mkdir()
        deriv = os.path.join(tmp, "deriv")
        target = write_dataset_description(path=deriv, source_datasets=[str(source)])
        with open(target, encoding="utf-8") as f:
            content = json.load(f)

        uri = source.absolute().as_uri()
        npt.assert_equal(content["SourceDatasets"], [{"URL": uri}])
        npt.assert_equal(content["DatasetLinks"], {"raw": uri})
        npt.assert_equal(" " in uri, False)

        target = write_dataset_description(path=deriv)
        with open(target, encoding="utf-8") as f:
            content = json.load(f)
        npt.assert_equal(content["SourceDatasets"], [])
        npt.assert_equal("DatasetLinks" in content, False)


if __name__ == "__main__":
    pytest.main([__file__])
