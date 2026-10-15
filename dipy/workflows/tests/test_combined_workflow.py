from datetime import UTC, date, datetime, time, timedelta, timezone
import hashlib
import json
import logging
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib
import warnings

import numpy as np
import numpy.testing as npt
import pytest
from scipy.ndimage import shift

from dipy.io.bids import DWIRun
from dipy.io.image import load_nifti, save_nifti
from dipy.utils.deprecator import ArgsDeprecationWarning
from dipy.utils.logging import file_handler
from dipy.workflows import interactive, report, templates
from dipy.workflows.combined_workflow import (
    PipelineError,
    _classify_inputs,
    _fill_io_section,
    _load_stage_state,
    _load_timing_cache,
    _pipeline_fingerprint,
    _planned_stage_outputs,
    _resolve_atlas_paths,
    _save_timing_cache,
    _validate_io_refs,
    _validate_mask_file,
    _write_bids_derivative_links,
    build_dependency_graph,
    check_start_stage,
    detect_output_conflicts,
    discover_stage_outputs,
    execute_semantic_pipeline,
    extract_variable_references,
    format_toml_config,
    optional_stages_to_skip,
    resolve_output_conflicts,
    resolve_stage_parameters,
    topological_sort,
    validate_pipeline_config,
    visualize_pipeline_dag,
)
from dipy.workflows.denoise import NLMeansFlow
from dipy.workflows.stage_isolation import execute_stage_isolated
from dipy.workflows.tests.test_qc import _phantom


def _denoise_mask_config():
    """Two stage pipeline: NLMeans denoising feeding a threshold mask."""
    return {
        "General": {"name": "test"},
        "pipeline": [
            {
                "name": "denoise",
                "cli": "dipy_denoise_nlmeans",
                "input_files": "${io.dwi}",
            },
            {
                "name": "mask",
                "cli": "dipy_mask",
                "input_files": "${denoise.out_denoised}",
                "lb": 15,
            },
        ],
    }


def _write_mask(path, *, fraction, values=(0, 1)):
    """Write a mask whose first ``fraction`` of slices hold ``values[1]``."""
    data = np.full((10, 10, 10), values[0], dtype=np.float32)
    data[: int(10 * fraction)] = values[1]
    save_nifti(path, data, np.eye(4))
    return path


def test_extract_variable_references():
    npt.assert_equal(
        extract_variable_references("${io.dwi} and ${denoise.out_denoised}"),
        {"io.dwi", "denoise.out_denoised"},
    )
    npt.assert_equal(extract_variable_references("no reference"), set())
    npt.assert_equal(extract_variable_references(42), set())


def test_build_dependency_graph():
    stages = [
        {"name": "a", "cli": "x", "input_files": "${io.dwi}"},
        {"name": "b", "cli": "x", "input_files": "${a.out_file}"},
        {"name": "c", "cli": "x", "inputs": ["${a.out_file}", "${b.out_file}"]},
        {"name": "d", "cli": "x", "options": {"ref": "${c.out_file}"}},
    ]

    deps = build_dependency_graph(stages)

    npt.assert_equal("a" in deps, False)
    npt.assert_equal(deps["b"], {"a"})
    npt.assert_equal(deps["c"], {"a", "b"})
    npt.assert_equal(deps["d"], {"c"})


def test_topological_sort():
    stages = [{"name": n} for n in ("c", "b", "a")]
    order = topological_sort(stages, {"c": {"b"}, "b": {"a"}})
    npt.assert_equal(order, ["a", "b", "c"])

    npt.assert_raises(
        ValueError, topological_sort, stages, {"a": {"b"}, "b": {"a"}, "c": {"a"}}
    )


def test_visualize_pipeline_dag():
    stages = [{"name": "a", "cli": "cli_a"}, {"name": "b", "cli": "cli_b"}]
    text = visualize_pipeline_dag(stages, {"b": {"a"}}, ["a", "b"])

    npt.assert_equal("a (cli_a) ← [no dependencies]" in text, True)
    npt.assert_equal("b (cli_b) ← [a]" in text, True)


def test_resolve_stage_parameters():
    stage = {
        "input_files": "${io.dwi}",
        "mask": "${mask.out_mask}",
        "list": ["${io.bvals}", 3],
        "nested": {"ref": "${mask.out_mask}"},
        "lb": 15,
    }
    resolved = resolve_stage_parameters(
        stage,
        {"mask": {"out_mask": "/out/mask.nii.gz"}},
        {"dwi": "/in/dwi.nii.gz", "bvals": "/in/dwi.bval"},
    )

    npt.assert_equal(resolved["input_files"], "/in/dwi.nii.gz")
    npt.assert_equal(resolved["mask"], "/out/mask.nii.gz")
    npt.assert_equal(resolved["list"], ["/in/dwi.bval", 3])
    npt.assert_equal(resolved["nested"], {"ref": "/out/mask.nii.gz"})
    npt.assert_equal(resolved["lb"], 15)

    npt.assert_raises(
        ValueError,
        resolve_stage_parameters,
        {"input_files": "${mask.out_missing}"},
        {"mask": {"out_mask": "/out/mask.nii.gz"}},
        {},
    )


def test_validate_pipeline_config():
    is_valid, errors = validate_pipeline_config(_denoise_mask_config()["pipeline"])
    npt.assert_equal(is_valid, True)
    npt.assert_equal(errors, [])

    stages = [
        {"name": "no_cli"},
        {"name": "bad_cli", "cli": "dipy_does_not_exist"},
        {"name": "denoise", "cli": "dipy_denoise_nlmeans", "input_files": "${dwi}"},
        {
            "name": "mask",
            "cli": "dipy_mask",
            "input_files": "${ghost.out_file}",
            "lb": "${denoise.out_missing}",
            "ub": "${denoise.out_dir}",
        },
        {"name": "typo", "cli": "dipy_mask", "input_files": "x", "lbb": 1},
    ]
    is_valid, errors = validate_pipeline_config(stages)

    npt.assert_equal(is_valid, False)
    npt.assert_equal(len(errors), 7)
    for expected in (
        "'no_cli' missing 'cli'",
        "Unknown CLI: dipy_does_not_exist",
        "missing dot notation",
        "unknown stage 'ghost'",
        "Unknown output 'out_missing'",
        "'typo': Unknown parameters for dipy_mask: ['lbb']",
        "'typo': Missing required parameters for dipy_mask: ['lb']",
    ):
        npt.assert_equal(any(expected in e for e in errors), True)


def test_validate_pipeline_config_parameter_tables():
    stages = [
        {
            "name": "mask",
            "cli": "dipy_mask",
            "inputs": {"input_files": "${io.dwi}"},
            "params": {"lb": 15},
            "optional": True,
        }
    ]
    npt.assert_equal(validate_pipeline_config(stages), (True, []))

    stages[0]["params"] = {"lbb": 15}
    is_valid, errors = validate_pipeline_config(stages)
    npt.assert_equal(is_valid, False)
    npt.assert_equal(len(errors), 2)


def test_optional_stages_to_skip():
    stages = [
        {"name": "b0", "cli": "x", "input_files": "${io.dwi}"},
        {
            "name": "t1_reg",
            "cli": "x",
            "static": "${b0.out_b0}",
            "moving": ["${io.t1w}"],
            "optional": True,
        },
        {
            "name": "t1_seg",
            "cli": "x",
            "input": "${t1_reg.out_moved}",
            "optional": True,
        },
        {"name": "dti", "cli": "x", "input_files": "${io.dwi}"},
    ]
    order = ["b0", "t1_reg", "t1_seg", "dti"]

    npt.assert_equal(
        optional_stages_to_skip(
            pipeline_stages=stages,
            execution_order=order,
            io_config={"dwi": "dwi.nii.gz", "t1w": "t1.nii.gz"},
        ),
        {},
    )

    skipped = optional_stages_to_skip(
        pipeline_stages=stages,
        execution_order=order,
        io_config={"dwi": "dwi.nii.gz", "t1w": ""},
    )
    npt.assert_equal(
        skipped,
        {
            "t1_reg": "no value for io.t1w",
            "t1_seg": "needs the skipped stage(s) t1_reg",
        },
    )

    del stages[2]["optional"]
    with pytest.raises(PipelineError) as error:
        optional_stages_to_skip(
            pipeline_stages=stages, execution_order=order, io_config={"dwi": "a"}
        )
    npt.assert_equal(error.value.stage, "t1_seg")


def test_detect_and_resolve_output_conflicts():
    config = {
        "pipeline": [
            {"name": "d1", "cli": "dipy_denoise_nlmeans"},
            {"name": "d2", "cli": "dipy_denoise_nlmeans"},
            {"name": "d3", "cli": "dipy_denoise_nlmeans", "out_dir": "other"},
            {
                "name": "d4",
                "cli": "dipy_denoise_nlmeans",
                "out_denoised": "explicit.nii.gz",
            },
        ]
    }

    conflicts = detect_output_conflicts(pipeline_stages=config["pipeline"])
    npt.assert_equal(conflicts, {"out_denoised": ["d1", "d2"]})

    warnings = resolve_output_conflicts(config=config, conflicts=conflicts)
    npt.assert_equal(len(warnings), 2)
    npt.assert_equal(config["pipeline"][0]["out_denoised"], "denoised_d1.nii.gz")
    npt.assert_equal(config["pipeline"][1]["out_denoised"], "denoised_d2.nii.gz")
    npt.assert_equal("out_denoised" in config["pipeline"][2], False)
    npt.assert_equal(detect_output_conflicts(pipeline_stages=config["pipeline"]), {})


def test_detect_output_conflicts_reads_parameter_tables():
    stages = [
        {"name": "d1", "cli": "dipy_denoise_nlmeans"},
        {"name": "d2", "cli": "dipy_denoise_nlmeans", "params": {"out_dir": "o"}},
        {
            "name": "d3",
            "cli": "dipy_denoise_nlmeans",
            "params": {"out_denoised": "mine.nii.gz"},
        },
    ]
    npt.assert_equal(detect_output_conflicts(pipeline_stages=stages), {})


def test_format_toml_config_roundtrip():
    config = {
        "General": {"name": "roundtrip", "version": 2},
        "io": {"dwi": "C:\\data\\dwi.nii.gz", "out_dir": "."},
        "pipeline": [
            {
                "name": "denoise",
                "cli": "dipy_denoise_nlmeans",
                "input_files": "${io.dwi}",
                "rician": True,
                "sigma": 1.5,
                "patch_radius": 1,
            }
        ],
    }

    npt.assert_equal(tomllib.loads(format_toml_config(config)), config)


def test_timing_cache():
    with TemporaryDirectory() as tmp:
        npt.assert_equal(_load_timing_cache(out_dir=tmp, fingerprint="abc"), {})

        _save_timing_cache(out_dir=tmp, fingerprint="abc", timing_data={"a": 1.5})
        npt.assert_equal(_load_timing_cache(out_dir=tmp, fingerprint="abc"), {"a": 1.5})
        npt.assert_equal(_load_timing_cache(out_dir=tmp, fingerprint="xyz"), {})


def test_pipeline_fingerprint():
    stages = _denoise_mask_config()["pipeline"]
    first = _pipeline_fingerprint(pipeline_stages=stages, io_config={"dwi": "a"})

    npt.assert_equal(len(first), 16)
    npt.assert_equal(
        _pipeline_fingerprint(pipeline_stages=stages, io_config={"dwi": "a"}), first
    )
    npt.assert_equal(
        _pipeline_fingerprint(pipeline_stages=stages, io_config={"dwi": "b"}) != first,
        True,
    )


def test_classify_inputs():
    with TemporaryDirectory() as tmp:
        found = _classify_inputs(
            input_files=[
                "cfg.toml",
                "dwi.nii.gz",
                "dwi.BVAL",
                "dwi.bvecs",
                "unknown.xyz",
            ],
            bids_mode=False,
        )
        npt.assert_equal(found["config_file"], "cfg.toml")
        npt.assert_equal(found["dwi"], "dwi.nii.gz")
        npt.assert_equal(found["bvals"], "dwi.BVAL")
        npt.assert_equal(found["bvecs"], "dwi.bvecs")
        npt.assert_equal(found["output_dir"], None)

        report = _classify_inputs(
            input_files=["a_report.html", "dwi.nii.gz"], bids_mode=False
        )
        npt.assert_equal(report["report"], "a_report.html")

        out = os.path.join(tmp, "out")
        bids = _classify_inputs(input_files=[out, "participant"], bids_mode=True)
        npt.assert_equal(bids["output_dir"], out)
        npt.assert_equal(bids["analysis_level"], "participant")


def test_fill_io_section():
    with TemporaryDirectory() as tmp:
        config = {"io": {"dwi": "old.nii.gz", "bvals": "keep.bval"}}
        out_dir = os.path.join(tmp, "out")

        io_config = _fill_io_section(
            config=config,
            overrides={"dwi": "new.nii.gz", "bvals": None, "t1w": ""},
            out_dir=out_dir,
            out_report="report.toml",
        )

        npt.assert_equal(io_config["dwi"], "new.nii.gz")
        npt.assert_equal(io_config["bvals"], "keep.bval")
        npt.assert_equal("t1w" in io_config, False)
        npt.assert_equal(io_config["out_report"], os.path.join(out_dir, "report.toml"))
        npt.assert_equal(os.path.isdir(out_dir), True)


def test_validate_io_refs():
    with TemporaryDirectory() as tmp:
        dwi = os.path.join(tmp, "dwi.nii.gz")
        open(dwi, "w").close()
        config = {
            "pipeline": [
                {
                    "name": "s",
                    "cli": "x",
                    "input_files": "${io.dwi}",
                    "bvals": "${io.bvals}",
                    "bvecs": "${io.bvecs}",
                    "pe": "${io.pe_dir}",
                    "report": "${io.out_report}",
                }
            ]
        }
        io_config = {"dwi": dwi, "bvals": "", "bvecs": os.path.join(tmp, "nope")}

        missing, invalid = _validate_io_refs(config=config, io_config=io_config)

        npt.assert_equal(missing, [("io.bvals", "bvals")])
        npt.assert_equal(invalid, [("io.bvecs", os.path.join(tmp, "nope"))])

        # inputs of optional stages may be left empty, but not point nowhere
        config["pipeline"].append(
            {"name": "t1", "cli": "x", "moving": "${io.t1w}", "optional": True}
        )
        io_config.update(bvals=dwi, bvecs=dwi)
        npt.assert_equal(
            _validate_io_refs(config=config, io_config=io_config), ([], [])
        )
        io_config["t1w"] = os.path.join(tmp, "t1.nii.gz")
        npt.assert_equal(
            _validate_io_refs(config=config, io_config=io_config),
            ([], [("io.t1w", os.path.join(tmp, "t1.nii.gz"))]),
        )


def test_validate_mask_file():
    with TemporaryDirectory() as tmp:
        _validate_mask_file(
            mask_file=_write_mask(os.path.join(tmp, "ok.nii.gz"), fraction=0.5)
        )

        # a tight brain mask on a large field of view covers little of it
        log_file = os.path.join(tmp, "run.log")
        with file_handler(log_file, level=logging.INFO):
            _validate_mask_file(
                mask_file=_write_mask(os.path.join(tmp, "small.nii.gz"), fraction=0.1)
            )
        with open(log_file, encoding="utf-8") as f:
            npt.assert_equal("covers only 10.0% of the volume" in f.read(), True)

        for name, kwargs, message in (
            ("empty.nii.gz", {"fraction": 0}, "is empty"),
            ("labels.nii.gz", {"fraction": 0.5, "values": (2, 1)}, "not binary"),
        ):
            with pytest.raises(PipelineError, match=message):
                _validate_mask_file(
                    mask_file=_write_mask(os.path.join(tmp, name), **kwargs)
                )

        with pytest.raises(PipelineError, match="Cannot load mask file"):
            _validate_mask_file(mask_file=os.path.join(tmp, "missing.nii.gz"))


def test_resolve_atlas_paths_without_download():
    config = {"pipeline": [{"name": "denoise", "cli": "dipy_denoise_nlmeans"}]}
    io_config = {}
    _resolve_atlas_paths(config=config, io_config=io_config)
    npt.assert_equal(io_config, {})

    config = {"pipeline": [{"name": "slr", "cli": "dipy_slr"}]}
    io_config = {"atlas_tractogram": "atlas.trk", "bundle_atlas_dir": "bundles"}
    _resolve_atlas_paths(config=config, io_config=io_config)
    npt.assert_equal(
        io_config, {"atlas_tractogram": "atlas.trk", "bundle_atlas_dir": "bundles"}
    )


def test_discover_stage_outputs():
    with TemporaryDirectory() as tmp:
        stage = {"name": "denoise", "cli": "dipy_denoise_nlmeans"}
        io_config = {"out_dir": tmp}

        npt.assert_equal(
            discover_stage_outputs(
                stage_name="denoise", stage_config=stage, io_config=io_config
            ),
            {"out_dir": tmp},
        )

        open(os.path.join(tmp, "dwi_nlmeans.nii.gz"), "w").close()
        open(os.path.join(tmp, "custom.nii.gz"), "w").close()

        found = discover_stage_outputs(
            stage_name="denoise", stage_config=stage, io_config=io_config
        )
        npt.assert_equal(found["out_denoised"], os.path.join(tmp, "dwi_nlmeans.nii.gz"))

        found = discover_stage_outputs(
            stage_name="denoise",
            stage_config={**stage, "out_denoised": "custom.nii.gz"},
            io_config=io_config,
        )
        npt.assert_equal(found["out_denoised"], os.path.join(tmp, "custom.nii.gz"))

        npt.assert_equal(
            discover_stage_outputs(
                stage_name="x", stage_config={"cli": "nope"}, io_config=io_config
            ),
            {},
        )


@pytest.mark.parametrize("name", templates.list_predefined_pipelines())
def test_predefined_pipelines_are_valid(name):
    config = tomllib.loads(templates.get_predefined_pipeline(pipeline_name=name))
    stages = config["pipeline"]

    is_valid, errors = validate_pipeline_config(stages)
    npt.assert_equal(errors, [])
    npt.assert_equal(is_valid, True)

    order = topological_sort(stages, build_dependency_graph(stages))
    npt.assert_equal(sorted(order), sorted(s["name"] for s in stages))
    npt.assert_equal(
        len(templates.get_pipeline_description(pipeline_name=name)) > 0, True
    )


def test_preprocessing_matches_full():
    def stages(name):
        text = templates.get_predefined_pipeline(pipeline_name=name)
        return tomllib.loads(text)["pipeline"]

    preprocessing = stages("preprocessing")
    npt.assert_equal(stages("full")[: len(preprocessing)], preprocessing)


def test_template_lookup_and_listing():
    npt.assert_raises(KeyError, templates.get_predefined_pipeline, pipeline_name="x")
    npt.assert_raises(KeyError, templates.get_pipeline_description, pipeline_name="x")

    every = templates.list_predefined_pipelines()
    npt.assert_equal(every, list(templates.PREDEFINED_PIPELINES))
    npt.assert_equal(
        templates.list_predefined_pipelines(log_level=logging.DEBUG), every
    )
    npt.assert_equal(
        templates.list_predefined_pipelines(log_level=logging.INFO), ["full"]
    )

    text = templates.list_pipelines_with_descriptions()
    for name in every:
        npt.assert_equal(f"  {name}" in text, True)
    short = templates.list_pipelines_with_descriptions(log_level=logging.INFO)
    npt.assert_equal("  full" in short, True)
    npt.assert_equal("  denoise" in short, False)


def test_execute_pipeline_rejects_bad_configs():
    io_config = {"out_dir": "."}

    with pytest.raises(PipelineError, match="No \\[\\[pipeline\\]\\]"):
        execute_semantic_pipeline(config={"pipeline": []}, io_config=io_config)

    with pytest.raises(PipelineError, match="Did you mean: mask"):
        execute_semantic_pipeline(
            config=_denoise_mask_config(), io_config=io_config, start="masq"
        )

    bad = _denoise_mask_config()
    bad["pipeline"][1]["input_files"] = "${denoise.out_missing}"
    npt.assert_raises(
        PipelineError, execute_semantic_pipeline, config=bad, io_config=io_config
    )

    cycle = _denoise_mask_config()
    cycle["pipeline"][0]["input_files"] = "${mask.out_mask}"
    npt.assert_raises(
        PipelineError, execute_semantic_pipeline, config=cycle, io_config=io_config
    )


def test_execute_pipeline_dry_run_resolves_conflicts():
    with TemporaryDirectory() as tmp:
        config = {
            "pipeline": [
                {"name": "d1", "cli": "dipy_denoise_nlmeans", "input_files": "a"},
                {"name": "d2", "cli": "dipy_denoise_nlmeans", "input_files": "b"},
            ]
        }
        config_file = os.path.join(tmp, "pipeline.toml")

        info = execute_semantic_pipeline(
            config=config,
            io_config={"out_dir": tmp},
            config_file_path=config_file,
            dry_run=True,
            overwrite_config=True,
        )

        npt.assert_equal(info["dry_run"], True)
        npt.assert_equal(info["stages"], [])
        npt.assert_equal(info["execution_order"], ["d1", "d2"])
        with open(config_file, "rb") as f:
            saved = tomllib.load(f)
        npt.assert_equal(
            [s["out_denoised"] for s in saved["pipeline"]],
            ["denoised_d1.nii.gz", "denoised_d2.nii.gz"],
        )
        npt.assert_equal(os.listdir(tmp), ["pipeline.toml"])


def test_execute_pipeline_runs_and_resumes():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir)
        io_config = {"dwi": paths["dwi"], "out_dir": out_dir}

        info = execute_semantic_pipeline(
            config=_denoise_mask_config(), io_config=io_config
        )

        npt.assert_equal([s["name"] for s in info["stages"]], ["denoise", "mask"])
        npt.assert_equal(all(s["success"] for s in info["stages"]), True)
        npt.assert_equal(any("skipped" in s for s in info["stages"]), False)
        npt.assert_equal(os.path.exists(info["report"]), True)
        npt.assert_equal(
            os.path.exists(os.path.join(out_dir, "reports", "index.html")), True
        )
        mask_file = info["stages"][1]["outputs"]["out_mask"]
        npt.assert_equal(os.path.exists(mask_file), True)
        npt.assert_equal(
            set(
                _load_timing_cache(
                    out_dir=out_dir,
                    fingerprint=_pipeline_fingerprint(
                        pipeline_stages=_denoise_mask_config()["pipeline"],
                        io_config=io_config,
                    ),
                )
            ),
            {"denoise", "mask"},
        )

        # existing outputs are reused instead of recomputed
        mtime = os.path.getmtime(mask_file)
        info = execute_semantic_pipeline(
            config=_denoise_mask_config(), io_config=io_config
        )
        npt.assert_equal(
            [s.get("skipped") for s in info["stages"]], ["restart", "restart"]
        )
        npt.assert_equal(os.path.getmtime(mask_file), mtime)

        # --start runs its stage even though its outputs exist
        info = execute_semantic_pipeline(
            config=_denoise_mask_config(), io_config=io_config, start="mask"
        )
        npt.assert_equal([s.get("skipped") for s in info["stages"]], ["restart", None])
        npt.assert_equal(os.path.getmtime(mask_file) > mtime, True)

        # --start skips earlier stages and wires their outputs from disk
        info = execute_semantic_pipeline(
            config=_denoise_mask_config(),
            io_config=io_config,
            start="mask",
            force=True,
        )
        npt.assert_equal(info["stages"][0]["skipped"], "restart")
        npt.assert_equal(
            info["stages"][0]["outputs"]["out_denoised"],
            os.path.join(out_dir, "dwi_nlmeans.nii.gz"),
        )
        npt.assert_equal("skipped" in info["stages"][1], False)
        npt.assert_equal(info["execution_order"], ["denoise", "mask"])


def test_execute_pipeline_uses_user_mask():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir)
        user_mask = _write_mask(os.path.join(tmp, "user_mask.nii.gz"), fraction=0.5)

        info = execute_semantic_pipeline(
            config=_denoise_mask_config(),
            io_config={"dwi": paths["dwi"], "mask": user_mask, "out_dir": out_dir},
        )

        mask_stage = info["stages"][1]
        npt.assert_equal(mask_stage["skipped"], "user_mask")
        npt.assert_equal(mask_stage["outputs"]["out_mask"], user_mask)
        npt.assert_equal(os.path.exists(os.path.join(out_dir, "mask.nii.gz")), False)


def test_execute_pipeline_passes_only_set_parameters():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        config = {
            "pipeline": [
                {
                    "name": "mask",
                    "cli": "dipy_median_otsu",
                    "input_files": "${io.dwi}",
                    "bvalues_files": "${io.bvals}",
                }
            ]
        }

        # median_otsu deprecates its autocrop argument, which is never set here
        with warnings.catch_warnings():
            warnings.simplefilter("error", ArgsDeprecationWarning)
            info = execute_semantic_pipeline(
                config=config, io_config={**paths, "out_dir": tmp}
            )
        npt.assert_equal(os.path.exists(info["stages"][0]["outputs"]["out_mask"]), True)


def test_execute_pipeline_stage_failure():
    with TemporaryDirectory() as tmp:
        io_config = {"dwi": os.path.join(tmp, "missing.nii.gz"), "out_dir": tmp}

        with pytest.raises(PipelineError) as error:
            execute_semantic_pipeline(
                config=_denoise_mask_config(), io_config=io_config
            )

        npt.assert_equal(error.value.stage, "denoise")
        npt.assert_equal(
            os.path.exists(os.path.join(tmp, "reports", "index.html")), True
        )


def test_execute_pipeline_failed_rerun_hides_previous_reports():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir)
        reports_dir = os.path.join(out_dir, "reports")

        execute_semantic_pipeline(
            config=_denoise_mask_config(),
            io_config={"dwi": paths["dwi"], "out_dir": out_dir},
        )
        first = report.load_stage_records(
            reports_dir=reports_dir, stage_names=["denoise", "mask"]
        )
        npt.assert_equal(first[0]["run_id"], first[1]["run_id"])

        with pytest.raises(PipelineError):
            execute_semantic_pipeline(
                config=_denoise_mask_config(),
                io_config={
                    "dwi": os.path.join(tmp, "missing.nii.gz"),
                    "out_dir": out_dir,
                },
                force=True,
            )

        denoise, _ = report.load_stage_records(
            reports_dir=reports_dir, stage_names=["denoise", "mask"]
        )
        npt.assert_equal(denoise["qc"]["status"], "FAIL")
        npt.assert_equal(denoise["run_id"] != first[0]["run_id"], True)
        npt.assert_equal(
            report.load_stage_record(
                reports_dir=reports_dir,
                index=2,
                stage_name="mask",
                run_id=denoise["run_id"],
            ),
            None,
        )
        with open(os.path.join(reports_dir, "index.html")) as f:
            npt.assert_equal("PENDING" in f.read(), True)


def test_full_pipeline_skips_t1_registration_without_t1():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        config = tomllib.loads(templates.get_predefined_pipeline(pipeline_name="full"))
        io_config = {**config["io"], **paths, "out_dir": tmp}

        info = execute_semantic_pipeline(
            config=config, io_config=io_config, dry_run=True
        )
        npt.assert_equal(
            info["optional_skips"], {"t1_registration": "no value for io.t1w"}
        )

        io_config["t1w"] = paths["dwi"]
        info = execute_semantic_pipeline(
            config=config, io_config=io_config, dry_run=True
        )
        npt.assert_equal(info["optional_skips"], {})


def test_execute_pipeline_optional_t1_registration():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        data, affine = load_nifti(paths["dwi"])
        t1w = os.path.join(tmp, "t1w.nii.gz")
        save_nifti(t1w, shift(data[..., 0], (1.5, -1, 0.5)).astype(np.float32), affine)
        config = {
            "pipeline": [
                {
                    "name": "b0_extraction",
                    "cli": "dipy_extract_b0",
                    "input_files": "${io.dwi}",
                    "bvalues_files": "${io.bvals}",
                },
                {
                    "name": "t1_registration",
                    "cli": "dipy_align_affine",
                    "static_image_files": "${b0_extraction.out_b0}",
                    "moving_image_files": "${io.t1w}",
                    "transform": "rigid",
                    "optional": True,
                },
            ]
        }

        skipped_dir = os.path.join(tmp, "no_t1")
        os.makedirs(skipped_dir)
        info = execute_semantic_pipeline(
            config=config,
            io_config={**paths, "t1w": "", "out_dir": skipped_dir},
        )
        npt.assert_equal(info["stages"][1]["skipped"], "optional")
        npt.assert_equal(
            os.path.exists(os.path.join(skipped_dir, "moved.nii.gz")), False
        )
        with open(os.path.join(skipped_dir, "reports", "index.html")) as f:
            npt.assert_equal("badge-optional" in f.read(), True)

        out_dir = os.path.join(tmp, "with_t1")
        os.makedirs(out_dir)
        info = execute_semantic_pipeline(
            config=config, io_config={**paths, "t1w": t1w, "out_dir": out_dir}
        )
        npt.assert_equal("skipped" in info["stages"][1], False)
        # the 2.5 mm voxels of the phantom turn the voxel shift into millimetres
        npt.assert_array_almost_equal(
            np.loadtxt(info["stages"][1]["outputs"]["out_affine"])[:3, 3],
            [3.75, -2.5, 1.25],
            decimal=0,
        )


def test_stage_isolation_keeps_openmp_out_of_the_child():
    # the spawned child imports this module first: it must not load any of
    # DIPY's compiled extensions, which link the libomp torch crashes on
    code = (
        "import sys, dipy.workflows.stage_isolation; "
        "print(sorted(name for name, module in list(sys.modules.items()) "
        "if name.startswith('dipy') "
        "and str(getattr(module, '__file__', '') or '').endswith(('.so', '.pyd'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    npt.assert_equal(result.stdout.strip(), "[]")


def test_execute_stage_isolated():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        outputs = execute_stage_isolated(
            module_name="dipy.workflows.mask",
            class_name="MaskFlow",
            init_params={},
            run_params={"input_files": paths["dwi"], "lb": 15, "out_dir": tmp},
        )
        npt.assert_equal(os.path.exists(os.path.join(tmp, "mask.nii.gz")), True)
        npt.assert_equal(outputs is not None, True)


def test_execute_stage_isolated_reports_child_errors():
    with TemporaryDirectory() as tmp:
        log_file = os.path.join(tmp, "run.log")
        missing = os.path.join(tmp, "nope.nii.gz")

        with (
            file_handler(log_file, level=logging.INFO),
            pytest.raises(RuntimeError, match="File not found") as error,
        ):
            execute_stage_isolated(
                module_name="dipy.workflows.mask",
                class_name="MaskFlow",
                init_params={},
                run_params={"input_files": missing, "lb": 1},
            )
        npt.assert_equal("Traceback" in str(error.value), True)

        # the child appends to the log files of the parent
        with open(log_file) as f:
            text = f.read()
        npt.assert_equal(
            "Isolated stage dipy.workflows.mask.MaskFlow failed" in text, True
        )
        npt.assert_equal(missing in text, True)

        # a child killed before reporting does not hang the parent
        with pytest.raises(RuntimeError, match="without reporting a result"):
            execute_stage_isolated(
                module_name="os", class_name="abort", init_params={}, run_params={}
            )


def test_format_toml_config_nested_values():
    config = {
        "General": {"name": 'quote " and \\ backslash', "tags": ["a", "b"]},
        "io": {"dwi": "C:\\data\\dwi.nii.gz", "flag": False, "empty": ""},
        "pipeline": [
            {
                "name": "track",
                "cli": "dipy_track",
                "inputs": {"pam_files": "${csd.out_pam}", "nested": {"x": 1}},
                "params": {"seed_density": 2, "level_iters": [1000, 500]},
                "bvalues_files": ["${io.bvals}"],
                "optional": True,
                "sigma": 1e-3,
                "skipped": None,
            }
        ],
    }

    parsed = tomllib.loads(format_toml_config(config))
    expected = json.loads(json.dumps(config))
    del expected["pipeline"][0]["skipped"]
    npt.assert_equal(parsed, expected)
    npt.assert_equal(
        tomllib.loads(format_toml_config({"pipeline": [{"name": "a", "x": (1, 2)}]})),
        {"pipeline": [{"name": "a", "x": [1, 2]}]},
    )


def test_validate_pipeline_config_nested_references():
    stages = [
        {"name": "denoise", "cli": "dipy_denoise_nlmeans", "input_files": "${io.dwi}"},
        {
            "name": "mask",
            "cli": "dipy_mask",
            "inputs": {"input_files": ["${ghost.out_file}"]},
            "params": {"lb": "${denoise.out_missing}"},
        },
    ]
    is_valid, errors = validate_pipeline_config(stages)

    npt.assert_equal(is_valid, False)
    npt.assert_equal(any("unknown stage 'ghost'" in e for e in errors), True)
    npt.assert_equal(any("Unknown output 'out_missing'" in e for e in errors), True)


def test_resolve_stage_parameters_rejects_stages_without_outputs():
    stage = {"input_files": "${denoise.out_denoised}"}

    with pytest.raises(PipelineError, match="stage 'denoise' has no outputs"):
        resolve_stage_parameters(stage, {}, {}, known_stages=["denoise", "mask"])

    # without the list of stages, unknown references are left for validation
    npt.assert_equal(resolve_stage_parameters(stage, {}, {}), stage)


def test_check_start_stage():
    check_start_stage(start=None, stage_names=["a"])
    check_start_stage(start="a", stage_names=["a"])
    with pytest.raises(PipelineError, match="Did you mean: mask"):
        check_start_stage(start="masq", stage_names=["denoise", "mask"])


def test_discover_stage_outputs_ignores_lookalikes():
    with TemporaryDirectory() as tmp:
        stage = {"name": "denoise", "cli": "dipy_denoise_nlmeans"}
        io_config = {"out_dir": tmp}

        # a dangling BIDS link and a file whose name ends like an output
        os.symlink(
            os.path.join(tmp, "gone.nii.gz"),
            os.path.join(tmp, "sub-01_desc-denoised_dwi.nii.gz"),
        )
        open(os.path.join(tmp, "my_denoised.nii.gz"), "w").close()
        open(os.path.join(tmp, "denoised.nii.gz"), "w").close()
        npt.assert_equal(
            discover_stage_outputs(
                stage_name="denoise", stage_config=stage, io_config=io_config
            ),
            {"out_dir": tmp},
        )

        # the default name is found only when it resolves to a file
        default = os.path.join(tmp, "dwi_nlmeans.nii.gz")
        os.symlink(os.path.join(tmp, "gone.nii.gz"), default)
        npt.assert_equal(
            "out_denoised"
            in discover_stage_outputs(
                stage_name="denoise", stage_config=stage, io_config=io_config
            ),
            False,
        )

        # names set under params and resolved against [io] are honoured
        dti = {
            "name": "dti",
            "cli": "dipy_fit_dti",
            "params": {"out_fa": "${io.prefix}_fa.nii.gz"},
        }
        open(os.path.join(tmp, "kfa.nii.gz"), "w").close()
        found = discover_stage_outputs(
            stage_name="dti", stage_config=dti, io_config={**io_config, "prefix": "s"}
        )
        npt.assert_equal("out_fa" in found, False)
        open(os.path.join(tmp, "s_fa.nii.gz"), "w").close()
        found = discover_stage_outputs(
            stage_name="dti", stage_config=dti, io_config={**io_config, "prefix": "s"}
        )
        npt.assert_equal(found["out_fa"], os.path.join(tmp, "s_fa.nii.gz"))


def test_fill_io_section_resets_run_keys():
    with TemporaryDirectory() as tmp:
        config = {"io": {"t1w": "other_t1.nii.gz", "session": "9", "atlas": "a"}}

        io_config = _fill_io_section(
            config=config,
            overrides={"t1w": None, "session": "", "atlas": None},
            out_dir=tmp,
            out_report="report.toml",
            reset_keys={"t1w", "session"},
        )

        npt.assert_equal(io_config["t1w"], "")
        npt.assert_equal(io_config["session"], "")
        npt.assert_equal(io_config["atlas"], "a")


def test_execute_pipeline_dry_run_keeps_user_config():
    with TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "out")
        config_file = os.path.join(tmp, "mine.toml")
        text = (
            "[[pipeline]]\n"
            'name = "d1"\ncli = "dipy_denoise_nlmeans"\ninput_files = "a"\n\n'
            "[[pipeline]]\n"
            'name = "d2"\ncli = "dipy_denoise_nlmeans"\ninput_files = "b"\n'
        )
        with open(config_file, "w") as f:
            f.write(text)
        with open(config_file, "rb") as f:
            config = tomllib.load(f)

        execute_semantic_pipeline(
            config=config,
            io_config={"out_dir": out_dir},
            config_file_path=config_file,
            dry_run=True,
        )

        with open(config_file) as f:
            npt.assert_equal(f.read(), text)
        with open(os.path.join(out_dir, "mine_resolved.toml"), "rb") as f:
            saved = tomllib.load(f)
        npt.assert_equal(
            [s["out_denoised"] for s in saved["pipeline"]],
            ["denoised_d1.nii.gz", "denoised_d2.nii.gz"],
        )


def test_execute_pipeline_user_mask_wins_over_previous_outputs():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir)
        config = {
            "pipeline": [
                {
                    "name": "mask",
                    "cli": "dipy_median_otsu",
                    "input_files": "${io.dwi}",
                    "bvalues_files": "${io.bvals}",
                },
                {
                    "name": "dti",
                    "cli": "dipy_fit_dti",
                    "input_files": "${io.dwi}",
                    "bvalues_files": "${io.bvals}",
                    "bvectors_files": "${io.bvecs}",
                    "mask_files": "${mask.out_mask}",
                    "save_metrics": ["fa"],
                },
            ]
        }

        info = execute_semantic_pipeline(
            config=config, io_config={**paths, "out_dir": out_dir}
        )
        auto_mask = info["stages"][0]["outputs"]["out_mask"]
        npt.assert_equal(os.path.exists(auto_mask), True)
        fa = info["stages"][1]["outputs"]["out_fa"]
        mtime = os.path.getmtime(fa)

        data, affine = load_nifti(paths["dwi"])
        user_mask = os.path.join(tmp, "user_mask.nii.gz")
        save_nifti(user_mask, np.ones(data.shape[:3], dtype=np.uint8), affine)
        info = execute_semantic_pipeline(
            config=config, io_config={**paths, "mask": user_mask, "out_dir": out_dir}
        )

        npt.assert_equal(info["stages"][0]["skipped"], "user_mask")
        npt.assert_equal(info["stages"][0]["outputs"]["out_mask"], user_mask)
        # the fit reads the user mask, so its earlier outputs are recomputed
        npt.assert_equal("skipped" in info["stages"][1], False)
        npt.assert_equal(os.path.getmtime(fa) > mtime, True)


def test_execute_pipeline_reruns_stages_downstream_of_a_rerun():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir)
        io_config = {"dwi": paths["dwi"], "out_dir": out_dir}

        execute_semantic_pipeline(config=_denoise_mask_config(), io_config=io_config)

        # the denoised volume is gone: denoise runs again, and so must mask
        mask_file = os.path.join(out_dir, "mask.nii.gz")
        mtime = os.path.getmtime(mask_file)
        os.remove(os.path.join(out_dir, "dwi_nlmeans.nii.gz"))
        info = execute_semantic_pipeline(
            config=_denoise_mask_config(), io_config=io_config
        )
        npt.assert_equal([s.get("skipped") for s in info["stages"]], [None, None])
        npt.assert_equal(os.path.getmtime(mask_file) > mtime, True)

        # a stage whose parameters changed runs again, its inputs do not
        changed = _denoise_mask_config()
        changed["pipeline"][1]["lb"] = 10
        info = execute_semantic_pipeline(config=changed, io_config=io_config)
        npt.assert_equal([s.get("skipped") for s in info["stages"]], ["restart", None])

        info = execute_semantic_pipeline(config=changed, io_config=io_config)
        npt.assert_equal(
            [s.get("skipped") for s in info["stages"]], ["restart", "restart"]
        )

        # outputs written to a directory the first run created are reused
        nested = _denoise_mask_config()
        nested["pipeline"][0]["out_dir"] = "${io.out_dir}/denoised"
        for expected in ([None, None], ["restart", "restart"]):
            info = execute_semantic_pipeline(config=nested, io_config=io_config)
            npt.assert_equal([s.get("skipped") for s in info["stages"]], expected)


def _dwi_run(root):
    """A DWIRun of a one subject dataset rooted at ``root``."""
    dwi = os.path.join(root, "sub-01", "dwi", "sub-01_dwi.nii.gz")
    os.makedirs(os.path.dirname(dwi))
    return DWIRun(root=root, dwi=dwi, entities={"sub": "01"})


def test_write_bids_derivative_links():
    with TemporaryDirectory() as tmp:
        run = _dwi_run(os.path.join(tmp, "ds"))
        deriv = os.path.join(tmp, "deriv")
        run_dir = os.path.join(deriv, "sub-01", "dwi")
        outputs = {}
        for stage in ("dti", "force", "tracking"):
            os.makedirs(os.path.join(run_dir, stage))
            outputs[stage] = {}
            for key, name in (
                ("out_fa", "fa.nii.gz"),
                ("out_md", "md.nii.gz"),
                ("out_tractogram", "tractogram.trx"),
                ("out_mask", "mask"),
            ):
                path = os.path.join(run_dir, stage, name)
                open(path, "w").close()
                outputs[stage][key] = path

        execution_info = {
            "stages": [
                {"name": "dti_fit", "cli": "dipy_fit_dti", "outputs": outputs["dti"]},
                {
                    "name": "force_fit",
                    "cli": "dipy_fit_force",
                    "outputs": {"out_fa": outputs["force"]["out_fa"]},
                },
                {
                    "name": "track_csd",
                    "cli": "dipy_track",
                    "outputs": {
                        "out_tractogram": outputs["tracking"]["out_tractogram"]
                    },
                },
            ]
        }
        del outputs["dti"]["out_tractogram"]

        created = _write_bids_derivative_links(
            run=run, deriv_root=deriv, execution_info=execution_info
        )

        npt.assert_equal(
            sorted(os.path.basename(link) for link in created),
            [
                "sub-01_desc-FAdti_dwimap.nii.gz",
                "sub-01_desc-FAforce_dwimap.nii.gz",
                "sub-01_desc-MD_dwimap.nii.gz",
                "sub-01_desc-brain_mask",
                "sub-01_desc-trackcsd_tractography.trx",
            ],
        )
        npt.assert_equal(
            os.path.realpath(
                os.path.join(run_dir, "sub-01_desc-FAforce_dwimap.nii.gz")
            ),
            os.path.realpath(outputs["force"]["out_fa"]),
        )

        with open(os.path.join(run_dir, "sub-01_desc-FAdti_dwimap.json")) as f:
            sidecar = json.load(f)
        npt.assert_equal(
            sidecar["RawSources"], ["bids:raw:sub-01/dwi/sub-01_dwi.nii.gz"]
        )
        npt.assert_equal(sidecar["GeneratedByStage"]["name"], "dti_fit")
        # an output without extension still gets a named sidecar
        npt.assert_equal(
            os.path.exists(os.path.join(run_dir, "sub-01_desc-brain_mask.json")), True
        )


def _all_choices():
    """Answers selecting every preprocessing step and reconstruction."""
    preprocessing = {
        "denoise_patch2self": True,
        "reslice": True,
        "gibbs": True,
        "motion_correction": True,
        "bias_correction": True,
        "mask": True,
    }
    methods = ["dti", "force", "dki", "csd", "csa", "gqi", "mapmri"]
    return preprocessing, methods


def test_interactive_pipeline_config_is_valid():
    preprocessing, methods = _all_choices()
    config = interactive.pipeline_config_from_choices(
        preprocessing=preprocessing,
        recon_methods=methods,
        brain_extraction_method="median_otsu",
        tracking_methods=["dti", "csd"],
        mapmri_timing=(0.0157, 0.0365),
    )
    stages = config["pipeline"]

    npt.assert_equal(validate_pipeline_config(stages), (True, []))
    npt.assert_equal(
        [s["name"] for s in stages[:6]],
        ["denoise_patch2self", "gibbs", "motion", "bias", "reslice", "mask"],
    )
    # the order documented in the full template, after denoising
    full = tomllib.loads(templates.get_predefined_pipeline(pipeline_name="full"))
    npt.assert_equal(
        [s["cli"] for s in stages[1:5]], [s["cli"] for s in full["pipeline"][1:5]]
    )

    # every stage after motion correction reads the rotated b-vectors
    after_motion = stages[3:]
    for stage in after_motion:
        for key in ("bvec", "bvectors_files", "bvecs_files"):
            if key in stage:
                npt.assert_equal(stage[key], "${motion.out_bvecs}")
    npt.assert_equal(
        any("${io.bvecs}" in json.dumps(stage) for stage in after_motion), False
    )

    with pytest.raises(ValueError, match="MAPMRI"):
        interactive.pipeline_config_from_choices(
            preprocessing=preprocessing,
            recon_methods=["dti", "mapmri"],
            brain_extraction_method="median_otsu",
            tracking_methods=[],
        )


def _interactive_call(expression, answers):
    """Evaluate an interactive prompt in a child fed ``answers`` on stdin."""
    code = (
        "import json; from dipy.workflows import interactive; "
        f"print('RESULT=' + json.dumps({expression}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        input=answers,
        capture_output=True,
        text=True,
        check=True,
    )
    line = next(x for x in result.stdout.splitlines() if "RESULT=" in x)
    return json.loads(line.split("RESULT=", 1)[1])


def test_interactive_prompts_read_answers():
    # the recommended defaults include the brain mask the fits rely on
    selected = _interactive_call(
        "interactive.interactive_preprocessing_selection()", "\n"
    )
    npt.assert_equal(selected["mask"], True)
    npt.assert_equal(selected["denoise_patch2self"], True)

    timing = _interactive_call(
        "interactive.interactive_mapmri_timing()", "abc\n0.0157\n-1\n0.0365\n"
    )
    npt.assert_equal(timing, [0.0157, 0.0365])


def test_interactive_synthseg_check_does_not_load_torch():
    code = (
        "import sys; from dipy.workflows import interactive; "
        "interactive.check_synthseg_available(); print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    npt.assert_equal(result.stdout.strip(), "False")


def test_interactive_menu_offers_the_recommended_pipeline():
    data_chars = interactive.DataCharacteristics(
        num_bvals=2,
        unique_bvals=[0, 1000],
        is_single_shell=True,
        is_multi_shell=False,
        supports_dki=False,
        recommended_pipeline="tractography",
        recommended_methods=["dti", "csd"],
    )

    npt.assert_equal(
        interactive.pipeline_menu(data_chars=data_chars, log_level=logging.INFO),
        ["tractography", "full"],
    )
    npt.assert_equal(
        interactive.pipeline_menu(data_chars=data_chars, log_level=logging.DEBUG),
        templates.list_predefined_pipelines(),
    )


def test_format_toml_config_dates_and_times():
    config = {
        "General": {
            "day": date(1979, 5, 27),
            "clock": time(7, 32, 0, 500000),
            "local": datetime(1979, 5, 27, 7, 32),  # noqa: DTZ001
            "utc": datetime(1979, 5, 27, 7, 32, tzinfo=UTC),
            "offset": datetime(
                1979, 5, 27, 0, 32, 0, 999999, tzinfo=timezone(timedelta(hours=-7))
            ),
            "list": [date(2024, 5, 1), time(0, 0)],
        }
    }
    npt.assert_equal(tomllib.loads(format_toml_config(config)), config)

    parsed = tomllib.loads("[General]\nwhen = 2024-05-01T10:00:00+02:00\n")
    npt.assert_equal(tomllib.loads(format_toml_config(parsed)), parsed)

    with pytest.raises(TypeError, match="time of day with an offset"):
        format_toml_config({"t": time(7, 32, tzinfo=UTC)})


def test_config_files_are_utf8_whatever_the_locale():
    code = (
        "import locale, sys\n"
        "from dipy.workflows import combined_workflow as cw\n"
        "assert locale.getpreferredencoding(False).lower() in "
        "('ascii', 'us-ascii', 'ansi_x3.4-1968')\n"
        "out = sys.argv[1]\n"
        "config = {'General': {'name': 'a \\u2192 b'}}\n"
        "path = cw._write_pipeline_config(config=config, out_dir=out, "
        "timestamp='t', pipeline_name='p')\n"
        "path = cw._write_resolved_config(config=config, config_file_path=path, "
        "out_dir=out, overwrite=False)\n"
        "cw._save_stage_state(out_dir=out, stage_state={'s \\u2192': {}})\n"
        "assert cw._load_stage_state(out_dir=out) == {'s \\u2192': {}}\n"
        "cw._save_timing_cache(out_dir=out, fingerprint='f', "
        "timing_data={'s \\u2192': 1.0})\n"
        "assert cw._load_timing_cache(out_dir=out, fingerprint='f') == "
        "{'s \\u2192': 1.0}\n"
        "print(path)\n"
    )
    # under the C locale, Linux decodes a non-ASCII -c argument to surrogates
    assert code.isascii()
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONCOERCECLOCALE": "0"}
    env.pop("PYTHONUTF8", None)
    with TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "-X", "utf8=0", "-c", code, tmp],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            check=False,
        )
        npt.assert_equal(result.returncode, 0, err_msg=result.stderr)
        with open(result.stdout.splitlines()[-1], "rb") as f:
            npt.assert_equal(
                tomllib.loads(f.read().decode("utf-8"))["General"]["name"],
                "a \u2192 b",
            )


def _three_stage_config():
    """Bias field correction, named-mixing denoising and a no-op reslicing.

    Each stage writes an output its default name does not predict: the bias
    field correction renames its output, the denoising prefixes it with the
    input names and the reslicing returns its input, the volume already
    having the target voxel size.
    """
    return {
        "General": {"name": "test"},
        "pipeline": [
            {
                "name": "bias",
                "cli": "dipy_correct_biasfield",
                "input_files": "${io.dwi}",
                "bval": "${io.bvals}",
                "bvec": "${io.bvecs}",
                "method": "bspline",
            },
            {
                "name": "denoise",
                "cli": "dipy_denoise_nlmeans",
                "input_files": "${bias.out_corrected}",
                "mix_names": True,
            },
            {
                "name": "reslice",
                "cli": "dipy_reslice",
                "input_files": "${denoise.out_denoised}",
                "new_vox_size": [2.5],
            },
            {
                "name": "b0",
                "cli": "dipy_extract_b0",
                "input_files": "${reslice.out_resliced}",
                "bvalues_files": "${io.bvals}",
            },
        ],
    }


def test_execute_pipeline_resumes_stages_with_unpredictable_outputs():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        out_dir = os.path.join(tmp, "out")
        io_config = {**paths, "out_dir": out_dir}

        info = execute_semantic_pipeline(
            config=_three_stage_config(), io_config=io_config
        )
        npt.assert_equal([s.get("skipped") for s in info["stages"]], [None] * 4)
        outputs = {s["name"]: s["outputs"] for s in info["stages"]}
        denoised = os.path.join(out_dir, "dwi_biasfield_corrected__dwi_nlmeans.nii.gz")
        npt.assert_equal(
            str(outputs["bias"]["out_corrected"]),
            os.path.join(out_dir, "dwi_biasfield_corrected.nii.gz"),
        )
        npt.assert_equal(str(outputs["denoise"]["out_denoised"]), denoised)
        npt.assert_equal(str(outputs["reslice"]["out_resliced"]), denoised)

        state = _load_stage_state(out_dir=out_dir)
        npt.assert_equal(
            state["reslice"]["outputs"],
            {"out_resliced": os.path.basename(denoised)},
        )

        b0 = os.path.join(out_dir, "b0.nii.gz")
        mtime = os.path.getmtime(b0)
        info = execute_semantic_pipeline(
            config=_three_stage_config(), io_config=io_config
        )
        npt.assert_equal([s.get("skipped") for s in info["stages"]], ["restart"] * 4)
        npt.assert_equal(info["stages"][2]["outputs"]["out_resliced"], denoised)
        npt.assert_equal(os.path.getmtime(b0), mtime)

        info = execute_semantic_pipeline(
            config=_three_stage_config(), io_config=io_config, start="b0"
        )
        npt.assert_equal(
            [s.get("skipped") for s in info["stages"]], ["restart"] * 3 + [None]
        )
        npt.assert_equal(info["stages"][1]["outputs"]["out_denoised"], denoised)

        # a recorded output deleted since is computed again
        os.remove(denoised)
        info = execute_semantic_pipeline(
            config=_three_stage_config(), io_config=io_config
        )
        npt.assert_equal(
            [s.get("skipped") for s in info["stages"]], ["restart", None, None, None]
        )


def _denoise_dti_config(*, patch_radius, mask):
    return {
        "General": {"name": "test"},
        "pipeline": [
            {
                "name": "denoise",
                "cli": "dipy_denoise_nlmeans",
                "input_files": "${io.dwi}",
                "patch_radius": patch_radius,
            },
            {
                "name": "dti",
                "cli": "dipy_fit_dti",
                "input_files": "${denoise.out_denoised}",
                "bvalues_files": "${io.bvals}",
                "bvectors_files": "${io.bvecs}",
                "mask_files": mask,
                "save_metrics": ["fa"],
            },
        ],
    }


def test_execute_pipeline_reruns_stages_left_behind_by_a_failed_run():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        mask = os.path.join(tmp, "mask.nii.gz")
        save_nifti(mask, np.ones((24, 24, 16), np.uint8), np.diag([2.5] * 3 + [1]))
        out_dir = os.path.join(tmp, "out")
        io_config = {**paths, "out_dir": out_dir}
        fa = os.path.join(out_dir, "fa.nii.gz")
        denoised = os.path.join(out_dir, "dwi_nlmeans.nii.gz")

        execute_semantic_pipeline(
            config=_denoise_dti_config(patch_radius=1, mask=mask), io_config=io_config
        )

        # denoise runs again with new parameters, then dti fails
        with open(paths["bvecs"]) as f:
            bvecs = f.read()
        with open(paths["bvecs"], "w") as f:
            f.write("garbage\n")
        with pytest.raises(PipelineError) as error:
            execute_semantic_pipeline(
                config=_denoise_dti_config(patch_radius=2, mask=mask),
                io_config=io_config,
            )
        npt.assert_equal(error.value.stage, "dti")
        npt.assert_equal(os.path.getmtime(fa) < os.path.getmtime(denoised), True)
        npt.assert_equal(_load_stage_state(out_dir=out_dir)["dti"]["invalid"], True)

        # dti, computed from the previous denoised volume, cannot be reused
        with open(paths["bvecs"], "w") as f:
            f.write(bvecs)
        info = execute_semantic_pipeline(
            config=_denoise_dti_config(patch_radius=2, mask=mask), io_config=io_config
        )
        npt.assert_equal([s.get("skipped") for s in info["stages"]], ["restart", None])
        npt.assert_equal(os.path.getmtime(fa) > os.path.getmtime(denoised), True)
        npt.assert_equal("invalid" in _load_stage_state(out_dir=out_dir)["dti"], False)

        info = execute_semantic_pipeline(
            config=_denoise_dti_config(patch_radius=2, mask=mask), io_config=io_config
        )
        npt.assert_equal(
            [s.get("skipped") for s in info["stages"]], ["restart", "restart"]
        )


def test_execute_pipeline_refuses_to_overwrite_its_inputs():
    def digest(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    config = {
        "General": {"name": "test"},
        "pipeline": [
            {
                "name": "denoise",
                "cli": "dipy_denoise_nlmeans",
                "input_files": "${io.dwi}",
            }
        ],
    }
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        execute_semantic_pipeline(config=config, io_config={**paths, "out_dir": tmp})
        denoised = os.path.join(tmp, "dwi_nlmeans.nii.gz")
        before = digest(denoised)

        # the denoised volume fed back as the input of the same pipeline
        with pytest.raises(PipelineError, match="would overwrite its input file"):
            execute_semantic_pipeline(
                config=config, io_config={**paths, "dwi": denoised, "out_dir": tmp}
            )
        npt.assert_equal(digest(denoised), before)

        # an [io] file the stage does not read is protected as well
        with pytest.raises(PipelineError, match="io.bvals"):
            execute_semantic_pipeline(
                config=config,
                io_config={**paths, "bvals": denoised, "out_dir": tmp},
            )
        npt.assert_equal(digest(denoised), before)


def test_planned_stage_outputs_follow_the_naming_options():
    workflow_class = NLMeansFlow
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        run_params = {"input_files": paths["dwi"], "out_dir": tmp}
        npt.assert_equal(
            _planned_stage_outputs(
                workflow_class=workflow_class, init_params={}, run_params=run_params
            ),
            [os.path.join(os.path.abspath(tmp), "dwi_nlmeans.nii.gz")],
        )
        npt.assert_equal(
            _planned_stage_outputs(
                workflow_class=workflow_class,
                init_params={"mix_names": True},
                run_params={**run_params, "out_denoised": "d.nii.gz"},
            ),
            [os.path.join(os.path.abspath(tmp), "dwi__d.nii.gz")],
        )
        npt.assert_equal(
            _planned_stage_outputs(
                workflow_class=workflow_class,
                init_params={},
                run_params={"input_files": os.path.join(tmp, "missing.nii.gz")},
            ),
            [],
        )


def test_execute_pipeline_trusts_no_output_after_an_unreadable_state():
    with TemporaryDirectory() as tmp:
        paths = _phantom(tmp)
        out_dir = os.path.join(tmp, "out")
        io_config = {"dwi": paths["dwi"], "out_dir": out_dir}
        execute_semantic_pipeline(config=_denoise_mask_config(), io_config=io_config)

        state = os.path.join(out_dir, "pipeline_state.json")
        with open(state, "w", encoding="utf-8") as f:
            f.write('{"stages": {"denoise": ')
        info = execute_semantic_pipeline(
            config=_denoise_mask_config(), io_config=io_config
        )
        npt.assert_equal([s.get("skipped") for s in info["stages"]], [None, None])
        npt.assert_equal(
            sorted(_load_stage_state(out_dir=out_dir)), ["denoise", "mask"]
        )
        npt.assert_equal([f for f in os.listdir(out_dir) if f.endswith(".tmp")], [])

        info = execute_semantic_pipeline(
            config=_denoise_mask_config(), io_config=io_config
        )
        npt.assert_equal(
            [s.get("skipped") for s in info["stages"]], ["restart", "restart"]
        )
