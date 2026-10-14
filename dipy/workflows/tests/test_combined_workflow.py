import logging
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib

import numpy as np
import numpy.testing as npt
import pytest
from scipy.ndimage import shift

from dipy.io.image import load_nifti, save_nifti
from dipy.workflows import templates
from dipy.workflows.combined_workflow import (
    PipelineError,
    _classify_inputs,
    _fill_io_section,
    _load_timing_cache,
    _pipeline_fingerprint,
    _resolve_atlas_paths,
    _save_timing_cache,
    _validate_io_refs,
    _validate_mask_file,
    build_dependency_graph,
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
        _validate_mask_file(_write_mask(os.path.join(tmp, "ok.nii.gz"), fraction=0.5))

        for name, kwargs in (
            ("small.nii.gz", {"fraction": 0.1}),
            ("labels.nii.gz", {"fraction": 0.5, "values": (2, 1)}),
        ):
            npt.assert_raises(
                SystemExit,
                _validate_mask_file,
                _write_mask(os.path.join(tmp, name), **kwargs),
            )

        npt.assert_raises(
            SystemExit, _validate_mask_file, os.path.join(tmp, "missing.nii.gz")
        )


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

    npt.assert_raises(
        ValueError,
        execute_semantic_pipeline,
        config={"pipeline": []},
        io_config=io_config,
    )

    with pytest.raises(ValueError, match="Did you mean: mask"):
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
    # the spawned child imports this module first: it must not load the
    # libomp that DIPY's extensions link against, or torch crashes later
    code = (
        "import sys, dipy.workflows.stage_isolation; "
        "print(any(name.startswith('dipy.') and name.endswith('omp') "
        "for name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    npt.assert_equal(result.stdout.strip(), "False")


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

        with pytest.raises(RuntimeError, match="exited with code"):
            execute_stage_isolated(
                module_name="dipy.workflows.mask",
                class_name="MaskFlow",
                init_params={},
                run_params={"input_files": os.path.join(tmp, "nope.nii.gz"), "lb": 1},
            )
