"""Semantic Pipeline Execution System for DIPY Workflows.

This module implements a flexible pipeline system based on semantic naming
and automatic DAG-based wiring. The pipeline:
- Uses semantic stage names
- Automatically infers data flow from output→input name matching
- Executes stages in topological order based on dependencies
- Supports interactive mode for dynamic pipeline construction
- Supports flexible TOML configuration via [[pipeline]] sections
"""

from collections import Counter, defaultdict, deque
import copy
from datetime import datetime
import difflib
from functools import partial
import glob
import hashlib
import http.server
import importlib
import inspect
import json
import os

# PyTorch wheels on macOS bundle their own libomp.dylib, which conflicts with
# the conda/Homebrew libomp.dylib loaded by scipy/numpy.  Both are LLVM libomp
# (the same implementation), so duplicate-loading is safe: the second
# initialisation becomes a no-op and there is no thread-pool corruption risk.
# This must be set before any import that transitively loads libomp (e.g.
# importlib.import_module("dipy.workflows.segment") pulls in all of torch).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import re
import shutil
import socket
import socketserver
import subprocess
import sys
import time
import traceback
import urllib.parse
import webbrowser

import numpy as np

from dipy.data import fetch_30_bundle_atlas_hcp842
from dipy.io.bids import (
    BIDS_ENTITY_ORDER,
    build_entity_name,
    derivative_path,
    find_dwi_runs,
    list_sessions,
    list_subjects,
    looks_like_dataset,
    phase_encoding_info,
    split_extension,
    validate_dataset,
    write_dataset_description,
)
from dipy.io.image import load_nifti, load_nifti_data
from dipy.utils.logging import file_handler, logger
from dipy.utils.optpkg import optional_package
from dipy.workflows import interactive, report, templates
from dipy.workflows.cli import cli_flows
from dipy.workflows.qc import qc_result, run_stage_qc
from dipy.workflows.stage_isolation import execute_stage_isolated
from dipy.workflows.workflow import Workflow

try:  # standard module since Python 3.11
    import tomllib as toml
except ImportError:
    toml, have_toml, _ = optional_package("tomli")


class PipelineError(RuntimeError):
    """Raised when a pipeline cannot be built or fails during execution.

    Callers running a single pipeline translate this into ``sys.exit(1)``;
    callers running a batch of pipelines catch it to isolate the failure to
    one dataset entry.

    Parameters
    ----------
    message : str
        Human readable description of the failure.
    stage : str, optional
        Name of the pipeline stage that failed, when the failure happened
        during the execution of a stage.

    Attributes
    ----------
    stage : str or None
        Name of the failing stage.
    """

    def __init__(self, message, *, stage=None):
        super().__init__(message)
        self.stage = stage


def _pipeline_fingerprint(*, pipeline_stages, io_config):
    """Compute a short hash identifying a pipeline and its input configuration.

    Parameters
    ----------
    pipeline_stages : list
        List of stage configuration dicts from ``[[pipeline]]`` sections.
    io_config : dict
        I/O configuration dict (``[io]`` section), including input file paths
        and output directory.

    Returns
    -------
    str
        16-character hex digest of the combined pipeline and I/O configuration.
    """
    content = json.dumps({"pipeline": pipeline_stages, "io": io_config}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _load_timing_cache(*, out_dir, fingerprint):
    """Load per-stage timing from disk.

    Parameters
    ----------
    out_dir : str
        Pipeline output directory.
    fingerprint : str
        Pipeline fingerprint to validate cache freshness.

    Returns
    -------
    dict
        Mapping of stage name to duration in seconds. Returns ``{}`` if the
        cache is missing or belongs to a different pipeline.
    """
    path = os.path.join(out_dir, "pipeline_timing.json")
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("_fingerprint") != fingerprint:
            logger.debug("Timing cache fingerprint mismatch — discarding.")
            return {}
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception:
        return {}


def _save_timing_cache(*, out_dir, fingerprint, timing_data):
    """Persist per-stage timing with a pipeline fingerprint.

    Parameters
    ----------
    out_dir : str
        Pipeline output directory.
    fingerprint : str
        Pipeline fingerprint to embed in the cache file.
    timing_data : dict
        Mapping of stage name to duration in seconds.
    """
    path = os.path.join(out_dir, "pipeline_timing.json")
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"_fingerprint": fingerprint, **timing_data}, f, indent=2)
    except Exception as e:
        logger.debug(f"Could not save timing cache: {e}")


def format_toml_config(config):
    """Format configuration as readable TOML string.

    This creates a human-readable TOML format matching the template style,
    with proper multiline formatting for [[pipeline]] sections.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    str
        Formatted TOML string.
    """
    lines = []

    if "General" in config:
        lines.append("[General]")
        for key, value in config["General"].items():
            if isinstance(value, str):
                # Escape backslashes for Windows paths
                escaped_value = value.replace("\\", "\\\\")
                lines.append(f'{key} = "{escaped_value}"')
            else:
                lines.append(f"{key} = {value}")
        lines.append("")

    if "io" in config:
        lines.append("[io]")
        for key, value in config["io"].items():
            if isinstance(value, str):
                # Escape backslashes for Windows paths
                escaped_value = value.replace("\\", "\\\\")
                lines.append(f'{key} = "{escaped_value}"')
            else:
                lines.append(f"{key} = {value}")
        lines.append("")

    if "pipeline" in config:
        for stage in config["pipeline"]:
            lines.append("[[pipeline]]")
            if "name" in stage:
                # Escape backslashes for Windows paths
                escaped_name = stage["name"].replace("\\", "\\\\")
                lines.append(f'name = "{escaped_name}"')
            if "cli" in stage:
                # Escape backslashes for Windows paths
                escaped_cli = stage["cli"].replace("\\", "\\\\")
                lines.append(f'cli = "{escaped_cli}"')

            for key, value in stage.items():
                if key in ("name", "cli"):
                    continue
                if isinstance(value, str):
                    # Escape backslashes for Windows paths
                    escaped_value = value.replace("\\", "\\\\")
                    lines.append(f'{key} = "{escaped_value}"')
                elif isinstance(value, bool):
                    lines.append(f"{key} = {str(value).lower()}")
                elif isinstance(value, (int, float)):
                    lines.append(f"{key} = {value}")
                else:
                    lines.append(f"{key} = {value}")
            lines.append("")

    return "\n".join(lines)


# =============================================================================
# Semantic Pipeline Directed Acyclic Graph (DAG)  Functions
# =============================================================================


def extract_variable_references(value):
    """Extract variable references from a value.

    Finds all ${...} patterns in strings.

    Parameters
    ----------
    value : any
        Value to search (typically string).

    Returns
    -------
    set
        Set of variable references (e.g., {'io.dwi', 'denoise.out_denoised'}).
    """
    if not isinstance(value, str):
        return set()
    return set(re.findall(r"\$\{([^}]+)\}", value))


def introspect_workflow_outputs(cli_name: str):
    """Introspect a workflow to discover its output parameters.

    Parameters
    ----------
    cli_name : str
        CLI command name (e.g., 'dipy_denoise_nlmeans').

    Returns
    -------
    set
        Set of output parameter names (e.g., {'out_denoised'}).
    """
    if cli_name not in cli_flows:
        raise ValueError(f"Unknown CLI: {cli_name}")

    module_name, class_name = cli_flows[cli_name]
    module = importlib.import_module(module_name)
    workflow_class = getattr(module, class_name)

    sig = inspect.signature(workflow_class.run)

    outputs = set()
    for param_name in sig.parameters:
        if param_name.startswith("out_") and param_name != "out_dir":
            outputs.add(param_name)

    return outputs


def _stage_parameter_errors(*, stage):
    """List the parameters of a stage that its workflow cannot accept.

    A stage may set any argument of the workflow ``run`` or ``__init__``
    methods, at its top level or under its ``inputs`` and ``params`` tables,
    and must set every ``run`` argument that has no default.

    Parameters
    ----------
    stage : dict
        Stage configuration from a ``[[pipeline]]`` section, with a ``cli``
        known to ``cli_flows``.

    Returns
    -------
    list of str
        One message per unknown or missing parameter.
    """
    module_name, class_name = cli_flows[stage["cli"]]
    workflow_class = getattr(importlib.import_module(module_name), class_name)
    run_params = inspect.signature(workflow_class.run).parameters
    init_params = inspect.signature(workflow_class.__init__).parameters
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)

    given = _stage_run_params(stage)

    unknown = sorted(
        key for key in given if key not in run_params and key not in init_params
    )
    missing = [
        name
        for name, param in run_params.items()
        if name != "self"
        and param.default is inspect.Parameter.empty
        and param.kind not in variadic
        and name not in given
    ]

    errors = []
    if unknown:
        errors.append(
            f"Stage '{stage['name']}': Unknown parameters for {stage['cli']}: {unknown}"
        )
    if missing:
        errors.append(
            f"Stage '{stage['name']}': Missing required parameters for "
            f"{stage['cli']}: {missing}"
        )
    return errors


def validate_pipeline_config(pipeline_stages):
    """Validate pipeline configuration for correct input/output references.

    Checks that every stage names a known CLI, sets only parameters its
    workflow accepts along with all the required ones, and references
    existing outputs of other stages.

    Parameters
    ----------
    pipeline_stages : list
        List of stage configurations from [[pipeline]] sections.

    Returns
    -------
    tuple
        (is_valid, list of error messages)
    """
    errors = []
    stage_outputs = {}

    for stage in pipeline_stages:
        stage_name = stage["name"]
        cli_name = stage.get("cli")

        if not cli_name:
            errors.append(f"Stage '{stage_name}' missing 'cli' parameter")
            continue

        try:
            outputs = introspect_workflow_outputs(cli_name)
            stage_outputs[stage_name] = outputs
            errors.extend(_stage_parameter_errors(stage=stage))
        except ValueError as e:
            errors.append(f"Stage '{stage_name}': {str(e)}")
        except Exception as e:
            errors.append(
                f"Stage '{stage_name}': Failed to introspect CLI '{cli_name}': {str(e)}"
            )

    for stage in pipeline_stages:
        stage_name = stage["name"]

        for param_name, param_value in stage.items():
            if param_name in ["name", "cli"]:
                continue

            refs = extract_variable_references(param_value)

            for ref in refs:
                if "." not in ref:
                    errors.append(
                        f"Stage '{stage_name}': Invalid reference "
                        f"'${{{ref}}}' (missing dot notation)"
                    )
                    continue

                ref_stage, ref_output = ref.split(".", 1)

                if ref_stage == "io":
                    continue

                if ref_stage not in stage_outputs:
                    errors.append(
                        f"Stage '{stage_name}': References unknown stage "
                        f"'{ref_stage}' in '${{{ref}}}'"
                    )
                    continue

                available_outputs = stage_outputs[ref_stage]
                if ref_output == "out_dir":
                    pass  # out_dir is always a valid reference on any stage
                elif ref_output not in available_outputs:
                    errors.append(
                        f"Stage '{stage_name}': Unknown output '{ref_output}' "
                        f"from stage '{ref_stage}'. Available: "
                        f"{sorted(available_outputs)}"
                    )

    return len(errors) == 0, errors


def _unset_io_inputs(*, stage, io_config):
    """List the ``[io]`` entries a stage references but that hold no value.

    Parameters
    ----------
    stage : dict
        Stage configuration from a ``[[pipeline]]`` section.
    io_config : dict
        Values of the ``[io]`` section.

    Returns
    -------
    list of str
        Sorted ``[io]`` keys, excluding the ``out_*`` ones.
    """
    refs = set()
    pending = [value for key, value in stage.items() if key not in ("name", "cli")]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        else:
            refs.update(extract_variable_references(value))

    keys = {ref.split(".", 1)[1] for ref in refs if ref.startswith("io.")}
    return sorted(
        key
        for key in keys
        if not key.startswith("out_") and not str(io_config.get(key, "") or "").strip()
    )


def optional_stages_to_skip(*, pipeline_stages, execution_order, io_config):
    """Find the optional stages that cannot run with the inputs at hand.

    A stage marked ``optional = true`` is skipped when an ``[io]`` entry it
    references is empty, for instance a T1 registration run without a T1
    image. An optional stage reading the outputs of a skipped stage is
    skipped too.

    Parameters
    ----------
    pipeline_stages : list
        List of stage configurations from [[pipeline]] sections.
    execution_order : list of str
        Stage names in execution order, from :func:`topological_sort`.
    io_config : dict
        Values of the ``[io]`` section.

    Returns
    -------
    dict
        Mapping of skipped stage name to the reason it is skipped.

    Raises
    ------
    PipelineError
        If a stage that is not optional reads the outputs of a skipped stage.
    """
    stage_map = {stage["name"]: stage for stage in pipeline_stages}
    dependencies = build_dependency_graph(pipeline_stages)
    skipped = {}

    for name in execution_order:
        stage = stage_map[name]
        unset = _unset_io_inputs(stage=stage, io_config=io_config)
        upstream = sorted(dependencies.get(name, set()) & skipped.keys())

        if stage.get("optional", False) and unset:
            skipped[name] = f"no value for {', '.join(f'io.{k}' for k in unset)}"
        elif upstream and stage.get("optional", False):
            skipped[name] = f"needs the skipped stage(s) {', '.join(upstream)}"
        elif upstream:
            raise PipelineError(
                f"Stage '{name}' needs the outputs of the skipped optional "
                f"stage(s) {', '.join(upstream)}; mark it optional or provide "
                "the missing inputs",
                stage=name,
            )

    return skipped


def build_dependency_graph(pipeline_stages):
    """Build dependency graph from pipeline stages.

    Parameters
    ----------
    pipeline_stages : list
        List of stage configurations from [[pipeline]] sections.

    Returns
    -------
    dict
        Mapping of stage_name -> set of stages it depends on.
    """
    stage_names = {stage["name"] for stage in pipeline_stages}
    dependencies = defaultdict(set)

    for stage in pipeline_stages:
        name = stage["name"]
        refs = set()

        for key, value in stage.items():
            if key in ("name", "cli"):
                continue

            if isinstance(value, dict):
                for nested_value in value.values():
                    refs.update(extract_variable_references(nested_value))
            elif isinstance(value, list):
                for item in value:
                    refs.update(extract_variable_references(item))
            else:
                refs.update(extract_variable_references(value))

        for ref in refs:
            parts = ref.split(".")
            if len(parts) >= 2:
                dep_stage = parts[0]
                if dep_stage in stage_names:
                    dependencies[name].add(dep_stage)

    return dependencies


def topological_sort(stages, dependencies):
    """Compute topological order of stages using Kahn's algorithm.

    Parameters
    ----------
    stages : list
        List of stage configurations.
    dependencies : dict
        Stage dependencies from build_dependency_graph().

    Returns
    -------
    list
        List of stage names in execution order.

    Raises
    ------
    ValueError
        If the graph contains cycles.
    """
    stage_names = [stage["name"] for stage in stages]

    in_degree = dict.fromkeys(stage_names, 0)
    for name, deps in dependencies.items():
        in_degree[name] = len(deps)

    queue = deque([name for name in stage_names if in_degree[name] == 0])
    result = []

    while queue:
        stage = queue.popleft()
        result.append(stage)

        for name, deps in dependencies.items():
            if stage in deps:
                in_degree[name] -= 1
                if in_degree[name] == 0:
                    queue.append(name)

    if len(result) != len(stage_names):
        raise ValueError(
            f"Pipeline contains cycles! Processed "
            f"{len(result)}/{len(stage_names)} stages."
        )

    return result


def visualize_pipeline_dag(
    stages,
    dependencies,
    execution_order,
):
    """Generate ASCII visualization of pipeline DAG.

    Parameters
    ----------
    stages : list
        Pipeline stages.
    dependencies : dict
        Stage dependencies.
    execution_order : list
        Execution order from topological_sort().

    Returns
    -------
    str
        Formatted pipeline visualization.
    """
    lines = ["Your Diffusion Pipeline:", "=" * 60]

    for stage_name in execution_order:
        # Get CLI for this stage
        stage = next(s for s in stages if s["name"] == stage_name)
        cli_name = stage.get("cli", "unknown")

        deps = dependencies.get(stage_name, set())
        if deps:
            dep_str = ", ".join(sorted(deps))
            lines.append(f"  {stage_name} ({cli_name}) ← [{dep_str}]")
        else:
            lines.append(f"  {stage_name} ({cli_name}) ← [no dependencies]")

    lines.append("=" * 60)
    return "\n".join(lines)


def resolve_stage_parameters(stage_config, resolved_outputs, io_config):
    """Resolve variable references in stage parameters.

    Parameters
    ----------
    stage_config : dict
        Stage configuration with potential ${} references.
    resolved_outputs : dict
        Mapping of stage_name -> {output_param -> file_path}.
    io_config : dict
        IO configuration from [io] section.

    Returns
    -------
    dict
        Stage configuration with all variables resolved.
    """

    def resolve_value(value):
        """Recursively resolve variables."""
        if isinstance(value, str):
            matches = re.findall(r"\$\{([^}]+)\}", value)
            for match in matches:
                parts = match.split(".")

                if parts[0] == "io":
                    io_key = ".".join(parts[1:])
                    replacement = io_config.get(io_key, "")
                    value = value.replace(f"${{{match}}}", str(replacement))

                elif parts[0] in resolved_outputs:
                    stage_name = parts[0]
                    output_param = ".".join(parts[1:])

                    if output_param in resolved_outputs[stage_name]:
                        replacement = resolved_outputs[stage_name][output_param]
                        logger.debug(f"Resolving ${{{match}}} -> {replacement}")
                        value = value.replace(f"${{{match}}}", str(replacement))
                    else:
                        available = list(resolved_outputs[stage_name].keys())
                        raise ValueError(
                            f"Unknown output '{output_param}' from "
                            f"stage '{stage_name}'. Available: {available}"
                        )

        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(v) for v in value]

        return value

    return resolve_value(stage_config)


# =============================================================================
# Output Conflict Detection and Resolution
# =============================================================================


def get_workflow_output_params(*, cli_command):
    """Dynamically discover output parameters for a workflow CLI.

    Uses inspect to examine the workflow's run() method signature
    and identify output parameters whose names start with ``out_``.

    Parameters
    ----------
    cli_command : str
        CLI command name (e.g., 'dipy_fit_csa').

    Returns
    -------
    list[str]
        List of output parameter names.

    Examples
    --------
    >>> get_workflow_output_params(cli_command="dipy_fit_csa")
    ['out_pam', 'out_shm', 'out_peaks_dir', 'out_peaks_values', 'out_peaks_indices', 'out_sphere', 'out_gfa', 'out_b', 'out_qa']
    """
    try:
        # Get workflow class from cli_flows
        if cli_command not in cli_flows:
            logger.debug(f"Unknown CLI command: {cli_command}")
            return []

        workflow_info = cli_flows[cli_command]

        # cli_flows contains tuples of (module_name, class_name)
        # Import the workflow class dynamically
        if isinstance(workflow_info, tuple):
            module_name, class_name = workflow_info
            module = importlib.import_module(module_name)
            workflow_class = getattr(module, class_name)
        else:
            # Fallback if it's already a class
            workflow_class = workflow_info

        # Inspect the run() method signature
        run_method = getattr(workflow_class, "run", None)
        if not run_method:
            logger.debug(f"No run() method found for {cli_command}")
            return []

        sig = inspect.signature(run_method)

        # Identify output parameters
        # Convention: parameters starting with 'out_' are outputs
        # Exclude 'out_dir' as it's a directory parameter, not an output file
        output_params = [
            param_name
            for param_name in sig.parameters
            if param_name.startswith("out_") and param_name != "out_dir"
        ]

        return output_params

    except Exception as e:
        logger.debug(f"Error inspecting {cli_command}: {e}")
        return []


def detect_output_conflicts(*, pipeline_stages):
    """Detect if multiple stages would produce conflicting output filenames.

    Only considers it a conflict if stages write to the same directory.
    Stages with different out_dir values do not conflict.

    Parameters
    ----------
    pipeline_stages : list
        List of stage configurations from [[pipeline]] sections.

    Returns
    -------
    dict
        Mapping of output_param -> list of stage_names that would conflict.
        Only includes parameters with actual conflicts (len > 1).

    Examples
    --------
    >>> stages = [
    ...     {"name": "csa_fit", "cli": "dipy_fit_csa"},
    ...     {"name": "csd_fit", "cli": "dipy_fit_csd"}
    ... ]
    >>> conflicts = detect_output_conflicts(pipeline_stages=stages)
    >>> conflicts
    {'out_pam': ['csa_fit', 'csd_fit'], 'out_shm': ['csa_fit', 'csd_fit'], 'out_peaks_dir': ['csa_fit', 'csd_fit'], 'out_peaks_values': ['csa_fit', 'csd_fit'], 'out_peaks_indices': ['csa_fit', 'csd_fit'], 'out_sphere': ['csa_fit', 'csd_fit'], 'out_gfa': ['csa_fit', 'csd_fit'], 'out_b': ['csa_fit', 'csd_fit'], 'out_qa': ['csa_fit', 'csd_fit']}
    """
    # Track which stages produce which output parameters
    # Map: (output_param, out_dir) -> list of (stage_name, cli)
    output_producers = defaultdict(list)

    for stage in pipeline_stages:
        stage_name = stage.get("name")
        cli = stage.get("cli")

        if not cli:
            continue

        # Get the output directory for this stage (None means use default)
        stage_out_dir = stage.get("out_dir", None)

        # Dynamically get output parameters for this CLI
        output_params = get_workflow_output_params(cli_command=cli)

        for output_param in output_params:
            # Skip if stage already has explicit output parameter
            # (user manually specified to avoid conflict)
            if output_param in stage:
                continue

            # Key by both output_param AND out_dir
            # Only stages writing to the same directory can conflict
            key = (output_param, stage_out_dir)
            output_producers[key].append((stage_name, cli))

    # Find actual conflicts (multiple stages producing same output to same dir)
    conflicts = {}
    for (output_param, _), producers in output_producers.items():
        if len(producers) > 1:
            # Extract just the stage names
            stage_names = [stage_name for stage_name, _ in producers]
            conflicts[output_param] = stage_names

    return conflicts


def resolve_output_conflicts(*, config, conflicts):
    """Resolve output filename conflicts by adding stage-specific names.

    Modifies the config in-place to add explicit output parameters with
    stage-specific filenames following the pattern:
    {base_name}_{stage_name}.{original_extension}

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary.
    conflicts : dict
        Conflict mapping from detect_output_conflicts().

    Returns
    -------
    list[str]
        List of warning messages about renamed outputs.

    Examples
    --------
    >>> config = {"pipeline": [
    ...     {"name": "csa_fit", "cli": "dipy_fit_csa"},
    ...     {"name": "csd_fit", "cli": "dipy_fit_csd"}
    ... ]}
    >>> conflicts = {"out_pam": ["csa_fit", "csd_fit"]}
    >>> warnings = resolve_output_conflicts(config=config, conflicts=conflicts)
    >>> config["pipeline"][0]["out_pam"]
    'pam_csa_fit.pam5'
    >>> config["pipeline"][1]["out_pam"]
    'pam_csd_fit.pam5'
    """
    warnings = []
    pipeline_stages = config.get("pipeline", [])

    # Build stage lookup for quick access
    stage_map = {stage["name"]: stage for stage in pipeline_stages}

    # For each conflicting output parameter
    for output_param, conflicting_stages in conflicts.items():
        # Determine base filename from parameter name
        # out_pam -> pam, out_fa -> fa, etc.
        base_name = output_param.replace("out_", "")

        # Add explicit output parameter to each conflicting stage
        for stage_name in conflicting_stages:
            if stage_name not in stage_map:
                continue

            stage = stage_map[stage_name]
            cli_name = stage.get("cli")

            # Get original extension from workflow default
            original_extension = "nii.gz"  # default fallback
            try:
                if cli_name and cli_name in cli_flows:
                    module_name, class_name = cli_flows[cli_name]
                    module = importlib.import_module(module_name)
                    workflow_class = getattr(module, class_name)
                    sig = inspect.signature(workflow_class.run)
                    param = sig.parameters.get(output_param)

                    if param and param.default != inspect.Parameter.empty:
                        default_filename = param.default
                        # Extract extension from default filename
                        # e.g., "peaks.pam5" -> "pam5"
                        if "." in default_filename:
                            original_extension = default_filename.split(".", 1)[1]
            except Exception as e:
                logger.debug(
                    f"Could not get extension for {output_param} in {cli_name}: {e}"
                )

            # Create stage-specific filename preserving original extension
            # Pattern: {base_name}_{stage_name}.{original_extension}
            # Example: pam_csa_fit.pam5, fa_dti_fit.nii.gz
            new_filename = f"{base_name}_{stage_name}.{original_extension}"

            # Add explicit output parameter to stage config
            stage[output_param] = new_filename

            # Log warning about renaming
            cli = stage.get("cli", "unknown")
            warning_msg = (
                f"Stage '{stage_name}' ({cli}): "
                f"Renamed output '{output_param}' to '{new_filename}' "
                f"to avoid conflict with other stages"
            )
            warnings.append(warning_msg)

    return warnings


# Stages built on PyTorch run through stage_isolation, away from the libomp
# that DIPY's OpenMP extensions load in this process.
_ISOLATED_STAGE_CLIS = frozenset(["dipy_brain_mask"])

_STAGE_KEYS = ("name", "cli", "inputs", "params", "optional")


def _stage_run_params(resolved_config):
    """Merge the ``inputs``, ``params`` and top-level keys of a stage config.

    Parameters
    ----------
    resolved_config : dict
        Stage configuration with variables resolved.

    Returns
    -------
    dict
        Workflow parameters, ``param -> value``.
    """
    return {
        **resolved_config.get("inputs", {}),
        **resolved_config.get("params", {}),
        **{
            key: value
            for key, value in resolved_config.items()
            if key not in _STAGE_KEYS
        },
    }


def _open_command(path):
    """Shell command that opens a file with the platform's default viewer.

    Parameters
    ----------
    path : str
        File to open.

    Returns
    -------
    str
        Command to paste in a terminal.
    """
    opener = {"darwin": "open", "win32": "start"}.get(sys.platform, "xdg-open")
    return f'{opener} "{path}"'


def _write_stage_qc(
    *,
    record,
    stage_map,
    resolved_outputs,
    io_config,
    stage_names,
    pipeline_name,
    error=None,
):
    """Run the QC of a stage and write its page, never raising.

    Parameters
    ----------
    record : dict
        Stage info (``name``, ``cli``, ``duration``, ``outputs``, ...).
    stage_map : dict
        Mapping of stage name -> stage configuration.
    resolved_outputs : dict
        Mapping of stage_name -> {output_param -> file_path}.
    io_config : dict
        IO configuration.
    stage_names : list of str
        All stage names in execution order.
    pipeline_name : str
        Pipeline name shown in the page header.
    error : str, optional
        Traceback of a failed stage; the page then reports a FAIL.
    """
    name = record["name"]
    index = stage_names.index(name) + 1
    reports_dir = os.path.join(io_config.get("out_dir", "."), "reports")
    try:
        previous = report.load_stage_record(
            reports_dir=reports_dir, index=index, stage_name=name
        )
        if error is not None:
            qc = qc_result(error=error, status="FAIL")
        elif record.get("skipped") == "restart" and previous is not None:
            qc = previous["qc"]
        elif record.get("skipped") == "optional":
            qc = qc_result(status="INFO")
        else:
            try:
                resolved = resolve_stage_parameters(
                    stage_map[name], resolved_outputs, io_config
                )
            except ValueError:
                resolved = stage_map[name]
            inputs = _stage_run_params(resolved)
            qc = run_stage_qc(
                cli=record.get("cli"),
                outputs=record.get("outputs", {}),
                inputs=inputs,
                io_config=io_config,
            )
        page = report.write_stage_report(
            reports_dir=reports_dir,
            stage_names=stage_names,
            index=index,
            record={**record, "qc": qc},
            pipeline_name=pipeline_name,
        )
        report.write_index(
            reports_dir=reports_dir,
            stage_names=stage_names,
            pipeline_name=pipeline_name,
        )
        logger.info(f"QC report [{qc['status']}]: {page}")
        logger.info(f"  Open it now: {_open_command(page)}  (or double-click it)")
    except Exception:
        logger.warning(
            f"Could not write the QC report of stage '{name}':\n"
            f"{traceback.format_exc()}"
        )


def execute_pipeline_stage(
    *, stage_name, stage_config, resolved_outputs, io_config, force=False
):
    """Execute a single pipeline stage.

    Parameters
    ----------
    stage_name : str
        Name of the stage.
    stage_config : dict
        Stage configuration.
    resolved_outputs : dict
        Previously resolved outputs from upstream stages.
    io_config : dict
        IO configuration.
    force : bool
        If ``True``, overwrite existing output files. Overrides any
        stage-level ``force`` setting from the configuration.

    Returns
    -------
    dict
        Mapping of output_param -> file_path for this stage.
    """
    cli_name = stage_config["cli"]

    if cli_name not in cli_flows:
        raise ValueError(f"Unknown CLI: {cli_name}")

    workflow_params = _stage_run_params(
        resolve_stage_parameters(stage_config, resolved_outputs, io_config)
    )

    module_name, class_name = cli_flows[cli_name]
    module = importlib.import_module(module_name)
    workflow_class = getattr(module, class_name)

    # Extract __init__ parameters (mix_names, force, output_strategy, skip)
    init_sig = inspect.signature(workflow_class.__init__)
    init_params = {}
    init_param_names = {"mix_names", "force", "output_strategy", "skip"}

    for param_name in init_param_names:
        if param_name in workflow_params:
            init_params[param_name] = workflow_params.pop(param_name)
            logger.debug(
                f"Using __init__ param: {param_name}={init_params[param_name]}"
            )
        elif param_name in init_sig.parameters:
            # Use default from signature
            param = init_sig.parameters[param_name]
            if param.default != inspect.Parameter.empty:
                init_params[param_name] = param.default

    # Pipeline-level --force overrides any stage-level force setting
    if force and "force" in init_sig.parameters:
        init_params["force"] = True
        logger.debug(f"Stage '{stage_name}': force overwrite enabled")

    # Get run() parameters
    sig = inspect.signature(workflow_class.run)
    valid_params = {
        name: param.default if param.default is not inspect.Parameter.empty else None
        for name, param in sig.parameters.items()
        if name != "self"
    }

    out_dir = io_config.get("out_dir", ".")
    if "out_dir" in valid_params and "out_dir" not in workflow_params:
        workflow_params["out_dir"] = out_dir

    invalid_params = set(workflow_params.keys()) - set(valid_params.keys())
    if invalid_params:
        raise ValueError(
            f"Invalid parameters for {cli_name}: {invalid_params}. "
            f"Valid: {list(valid_params.keys())}"
        )

    final_params = {**valid_params, **workflow_params}

    isolated = cli_name in _ISOLATED_STAGE_CLIS
    if isolated:
        logger.debug(
            f"Stage '{stage_name}' uses PyTorch — running in isolated subprocess"
        )

    logger.info(f"Executing stage '{stage_name}' using {cli_name}...")
    t_start = time.perf_counter()

    try:
        if isolated:
            last_outputs = execute_stage_isolated(
                module_name=module_name,
                class_name=class_name,
                init_params=init_params,
                run_params=final_params,
            )
        else:
            workflow = workflow_class(**init_params)
            workflow.run(**final_params)
            last_outputs = getattr(workflow, "last_generated_outputs", None)

        stage_outputs = {}
        if last_outputs is not None:
            if isinstance(last_outputs, dict):
                stage_outputs = last_outputs
                logger.debug(f"Stage '{stage_name}' outputs (dict): {stage_outputs}")
            elif isinstance(last_outputs, list):
                outputs = introspect_workflow_outputs(cli_name)
                sorted_params = sorted(outputs)
                logger.debug(
                    f"Stage '{stage_name}' - sorted params: {sorted_params}, "
                    f"files: {last_outputs}"
                )
                for out_param, out_file in zip(sorted_params, last_outputs):
                    stage_outputs[out_param] = out_file
                logger.debug(f"Stage '{stage_name}' outputs (zipped): {stage_outputs}")

        if "out_dir" in final_params:
            stage_outputs["out_dir"] = final_params["out_dir"]

        duration = time.perf_counter() - t_start
        logger.info(f"Completed stage '{stage_name}' in {duration:.2f} seconds")

        return stage_outputs

    except Exception as e:
        logger.error(f"Error executing stage '{stage_name}': {e}")
        raise


def discover_stage_outputs(*, stage_name, stage_config, io_config):
    """Discover outputs from a previously executed stage.

    Parameters
    ----------
    stage_name : str
        Name of the stage.
    stage_config : dict
        Stage configuration.
    io_config : dict
        IO configuration.

    Returns
    -------
    dict
        Dictionary mapping output parameter names to file paths.
    """
    cli_name = stage_config.get("cli")
    if not cli_name or cli_name not in cli_flows:
        return {}

    outputs = introspect_workflow_outputs(cli_name)
    # The stage out_dir still carries its ${io.*} references at this point.
    out_dir = resolve_stage_parameters(
        stage_config.get("out_dir"), {}, io_config
    ) or io_config.get("out_dir", ".")
    discovered = {}

    # Strategy 1: Check if output filenames are explicitly specified in config
    for output_param in outputs:
        if output_param in stage_config:
            specified_file = stage_config[output_param]
            full_path = os.path.join(out_dir, specified_file)
            if os.path.exists(full_path):
                discovered[output_param] = full_path
                logger.debug(f"Discovered {output_param} from config: {full_path}")

    # Strategy 2: Get default filenames from workflow signature
    try:
        module_name, class_name = cli_flows[cli_name]
        module = importlib.import_module(module_name)
        workflow_class = getattr(module, class_name)
        sig = inspect.signature(workflow_class.run)

        for output_param in outputs:
            if output_param in discovered:
                continue

            param = sig.parameters.get(output_param)
            if param and param.default != inspect.Parameter.empty:
                default_filename = param.default
                full_path = os.path.join(out_dir, default_filename)
                if os.path.exists(full_path):
                    discovered[output_param] = full_path
                    logger.debug(
                        f"Discovered {output_param} using default: {full_path}"
                    )
    except Exception as e:
        logger.debug(f"Could not introspect defaults for {cli_name}: {e}")

    # Strategy 3: Pattern matching for outputs not yet discovered
    for output_param in outputs:
        if output_param in discovered:
            continue

        potential_patterns = []

        if "out_" in output_param:
            suffix = output_param.replace("out_", "")
            # Try exact patterns first (without wildcards)
            potential_patterns.extend(
                [
                    os.path.join(out_dir, f"{suffix}.nii.gz"),
                    os.path.join(out_dir, f"{suffix}.nii"),
                    os.path.join(out_dir, f"{suffix}.trk"),
                    os.path.join(out_dir, f"{suffix}.tck"),
                    os.path.join(out_dir, f"{suffix}.txt"),
                    os.path.join(out_dir, f"{suffix}.pam5"),
                ]
            )
            # Then try wildcards as fallback
            potential_patterns.extend(
                [
                    os.path.join(out_dir, f"*{suffix}.nii.gz"),
                    os.path.join(out_dir, f"*{suffix}.nii"),
                ]
            )

        cli_workflow_patterns = {
            "dipy_denoise": ["dwi_denoised.nii.gz"],
            "dipy_denoise_nlmeans": ["dwi_nlmeans.nii.gz"],
            "dipy_denoise_mppca": ["dwi_mppca.nii.gz"],
            "dipy_denoise_patch2self": ["dwi_patch2self.nii.gz"],
            "dipy_gibbs_ringing": ["dwi_unring.nii.gz", "*_unring.nii.gz"],
            "dipy_correct_motion": ["dwi_moved.nii.gz", "*_moved.nii.gz"],
            "dipy_extract_b0": ["b0.nii.gz", "*_b0.nii.gz"],
        }

        if cli_name in cli_workflow_patterns:
            potential_patterns.extend(
                os.path.join(out_dir, pattern)
                for pattern in cli_workflow_patterns[cli_name]
            )

        stage_specific = stage_name.lower().replace("_", "")
        potential_patterns.extend(
            [
                os.path.join(out_dir, f"{stage_specific}.nii.gz"),
                os.path.join(out_dir, f"{stage_specific}.nii"),
                os.path.join(out_dir, f"*{stage_specific}*.nii.gz"),
            ]
        )

        for pattern in potential_patterns:
            if "*" in pattern:
                matches = glob.glob(pattern)
                if matches:
                    discovered[output_param] = matches[0]
                    break
            elif os.path.exists(pattern):
                discovered[output_param] = pattern
                break

    if out_dir:
        discovered["out_dir"] = out_dir

    return discovered


def estimate_pipeline_disk_usage(*, pipeline_stages, io_config):
    """Estimate total disk space required by pipeline output files.

    Uses a header-only load of the DWI file (no data read into memory) to
    determine voxel dimensions and dtype.  For each output parameter in each
    stage a size multiplier is applied depending on the parameter name or the
    default filename extension.

    Parameters
    ----------
    pipeline_stages : list
        List of stage configurations from [[pipeline]] sections.
    io_config : dict
        IO configuration dictionary with at least ``dwi`` and ``out_dir``.

    Returns
    -------
    tuple
        ``(total_bytes, itemized)`` where *total_bytes* is the integer sum of
        all estimated output sizes and *itemized* is a list of
        ``(stage_name, filename, bytes)`` tuples — one per output parameter.
    """
    # Multipliers in units of "one compressed 3-D volume"
    _PARAM_MULTIPLIERS = {
        "out_denoised": None,  # → n_vols
        "out_tensor": 6,
        "out_evec": 9,
        "out_eval": 3,
        "out_shm": None,  # → n_vols
        "out_peaks_dir": 15,
        "out_peaks_values": 5,
        "out_peaks_indices": 5,
    }
    _EXT_MULTIPLIERS = {
        "pam5": 10,
        "trk": 20,
        "trx": 20,
        "txt": 0,
        "mat": 0,
        "npz": 0,
    }

    dwi_path = io_config.get("dwi", "")
    single_3d_bytes = 0
    n_vols = 1

    if dwi_path and os.path.exists(dwi_path):
        try:
            _, _, img = load_nifti(dwi_path, return_img=True, as_ndarray=False)
            shape = img.shape
            itemsize = np.dtype(img.header.get_data_dtype()).itemsize
            x, y, z = shape[0], shape[1], shape[2]
            n_vols = int(shape[3]) if len(shape) > 3 else 1
            # 0.30 ≈ typical gzip compression ratio for brain MRI
            single_3d_bytes = int(x * y * z * itemsize * 0.30)
        except Exception as exc:
            logger.debug(f"Could not read DWI header for disk estimation: {exc}")

    itemized = []
    total_bytes = 0

    for stage in pipeline_stages:
        stage_name = stage.get("name", "unknown")
        cli_name = stage.get("cli")

        if not cli_name or cli_name not in cli_flows:
            continue

        try:
            module_name, class_name = cli_flows[cli_name]
            module = importlib.import_module(module_name)
            workflow_class = getattr(module, class_name)
            sig = inspect.signature(workflow_class.run)
        except Exception as exc:
            logger.debug(f"Could not introspect {cli_name} for disk estimation: {exc}")
            continue

        for param_name, param in sig.parameters.items():
            if not param_name.startswith("out_") or param_name == "out_dir":
                continue

            # Resolve default filename
            if param.default is inspect.Parameter.empty:
                base = param_name.replace("out_", "", 1)
                filename = f"{base}.nii.gz"
            else:
                filename = str(param.default)

            # Choose multiplier: param name → extension → default (1 scalar vol)
            if param_name in _PARAM_MULTIPLIERS:
                mult = _PARAM_MULTIPLIERS[param_name]
                if mult is None:
                    mult = n_vols
            else:
                ext = filename.split(".", 1)[1].lower() if "." in filename else "nii.gz"
                mult = _EXT_MULTIPLIERS.get(ext, 1)

            item_bytes = mult * single_3d_bytes
            total_bytes += item_bytes
            itemized.append((stage_name, filename, item_bytes))

    return total_bytes, itemized


def execute_semantic_pipeline(
    *,
    config,
    io_config,
    config_file_path=None,
    report_path=None,
    start=None,
    dry_run=False,
    force=False,
):
    """Execute pipeline using semantic DAG-based approach.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    io_config : dict
        IO configuration.
    config_file_path : str, optional
        Path to configuration file for report linking.
    report_path : str, optional
        Path for the HTML report file. Defaults to
        ``pipeline_report.html`` in the output directory.
    start : str, optional
        Stage name to start execution from. Earlier stages will be skipped,
        but their outputs must exist on disk.
    dry_run : bool
        If True, only show execution plan.
    force : bool
        If ``True``, overwrite existing output files for every stage.

    Returns
    -------
    dict
        Execution information with the keys ``stages``, ``total_time``,
        ``dag_visualization``, ``execution_order``, ``optional_skips``
        (skipped optional stage name to reason), ``report`` and ``dry_run``.

    Raises
    ------
    PipelineError
        If the configuration is invalid, the execution order cannot be
        computed, a required stage depends on a skipped optional stage, the
        output directory lacks free space, or a stage fails.
    """
    pipeline_stages = config.get("pipeline", [])
    if not pipeline_stages:
        raise ValueError("No [[pipeline]] sections found in configuration")

    stage_names = [s["name"] for s in pipeline_stages]

    if start is not None and start not in stage_names:
        suggestions = difflib.get_close_matches(start, stage_names, n=3, cutoff=0.6)
        msg = f"Stage '{start}' not found in pipeline."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        msg += f" Available stages: {', '.join(stage_names)}"
        raise ValueError(msg)

    logger.info(f"Building pipeline with {len(pipeline_stages)} stages...")

    is_valid, validation_errors = validate_pipeline_config(pipeline_stages)
    if not is_valid:
        logger.error("=" * 70)
        logger.error("Pipeline Configuration Validation Failed")
        logger.error("=" * 70)
        for error in validation_errors:
            logger.error(f"  ✗ {error}")
        logger.error("=" * 70)
        logger.error("Please update your configuration file to fix these errors.")
        raise PipelineError(
            f"Pipeline configuration validation failed: {'; '.join(validation_errors)}"
        )

    logger.info("✓ Pipeline configuration validated successfully")

    dependencies = build_dependency_graph(pipeline_stages)

    try:
        execution_order = topological_sort(pipeline_stages, dependencies)
    except ValueError as e:
        logger.error(f"Failed to compute execution order: {e}")
        raise PipelineError(f"Failed to compute execution order: {e}") from e

    dag_viz = visualize_pipeline_dag(pipeline_stages, dependencies, execution_order)
    logger.info(dag_viz)
    logger.info(f"Execution order: {' → '.join(execution_order)}")

    optional_skips = optional_stages_to_skip(
        pipeline_stages=pipeline_stages,
        execution_order=execution_order,
        io_config=io_config,
    )
    for stage_name, reason in optional_skips.items():
        logger.info(f"Skipping optional stage '{stage_name}': {reason}")

    # Detect and resolve output filename conflicts
    conflicts = detect_output_conflicts(pipeline_stages=pipeline_stages)
    if conflicts:
        logger.info("=" * 60)
        logger.info("Detecting output filename conflicts...")
        logger.info("=" * 60)

        # Show detected conflicts
        for output_param, conflicting_stages in conflicts.items():
            logger.info(
                f"Conflict detected for '{output_param}': "
                f"{', '.join(conflicting_stages)}"
            )

        # Resolve conflicts by renaming outputs
        warnings = resolve_output_conflicts(config=config, conflicts=conflicts)

        # Log warnings about renamed outputs
        logger.warning("=" * 60)
        logger.warning("Output files renamed to avoid conflicts:")
        logger.warning("=" * 60)
        for warning in warnings:
            logger.warning(f"  ⚠ {warning}")
        logger.warning("=" * 60)

        # Save updated config file with resolved conflicts
        if config_file_path:
            formatted_toml = format_toml_config(config)
            with open(config_file_path, "w") as f:
                f.write(formatted_toml)
            logger.info(
                f"Configuration with resolved conflicts saved to: {config_file_path}"
            )

    # ------------------------------------------------------------------
    # Disk space estimation (always shown, including dry-run)
    # ------------------------------------------------------------------
    total_bytes, disk_itemized = estimate_pipeline_disk_usage(
        pipeline_stages=pipeline_stages, io_config=io_config
    )

    def _fmt_bytes(b):
        if b >= 1_000_000_000:
            return f"{b / 1_000_000_000:.1f} GB"
        if b >= 1_000_000:
            return f"{b / 1_000_000:.0f} MB"
        if b >= 1_000:
            return f"{b / 1_000:.0f} KB"
        return f"{b} B"

    if disk_itemized:
        logger.info("=" * 60)
        logger.info("Estimated disk usage:")
        col1 = max(len(s) for s, _, _ in disk_itemized)
        col2 = max(len(f) for _, f, _ in disk_itemized)
        for stage_name, filename, item_bytes in disk_itemized:
            size_str = f"~{_fmt_bytes(item_bytes)}" if item_bytes else "negligible"
            logger.info(f"  {stage_name:<{col1}}  {filename:<{col2}}  {size_str}")
        logger.info(f"  {'TOTAL':<{col1}}  {'': <{col2}}  ~{_fmt_bytes(total_bytes)}")
        out_dir_check = io_config.get("out_dir", ".")
        try:
            free_bytes = shutil.disk_usage(out_dir_check).free
            free_str = _fmt_bytes(free_bytes)
            avail_label = f"Available on {out_dir_check}:"
            pad = col1 + col2 + 4
            if free_bytes < total_bytes:
                logger.error(f"  {avail_label:<{pad}}  {free_str}")
                logger.error(
                    f"Insufficient disk space: need ~{_fmt_bytes(total_bytes)}, "
                    f"have {free_str} on '{out_dir_check}'"
                )
                raise PipelineError(
                    f"Insufficient disk space: need ~{_fmt_bytes(total_bytes)}, "
                    f"have {free_str} on '{out_dir_check}'"
                )
            elif free_bytes < int(total_bytes * 1.5):
                logger.warning(f"  {avail_label:<{pad}}  {free_str}  ⚠ LOW MARGIN")
            else:
                logger.info(f"  {avail_label:<{pad}}  {free_str}  ✓")
        except PipelineError:
            raise
        except Exception as exc:
            logger.debug(f"Could not check free disk space on '{out_dir_check}': {exc}")
        logger.info("=" * 60)

    if dry_run:
        logger.info("[Dry run mode] Pipeline plan shown above - no execution performed")
        return {
            "stages": [],
            "total_time": 0.0,
            "dag_visualization": dag_viz,
            "execution_order": list(execution_order),
            "optional_skips": optional_skips,
            "report": None,
            "dry_run": True,
        }

    logger.info("=" * 60)
    logger.info("Starting Diffusion Pipeline Execution")
    logger.info("=" * 60)

    resolved_outputs = {}
    stage_map = {stage["name"]: stage for stage in pipeline_stages}
    stages_info = []
    full_execution_order = list(execution_order)  # preserve before --start trimming

    out_dir = io_config.get("out_dir", ".")
    fingerprint = _pipeline_fingerprint(
        pipeline_stages=pipeline_stages, io_config=io_config
    )
    timing_cache = _load_timing_cache(out_dir=out_dir, fingerprint=fingerprint)
    reports_dir = os.path.join(out_dir, "reports")
    write_qc = partial(
        _write_stage_qc,
        stage_map=stage_map,
        resolved_outputs=resolved_outputs,
        io_config=io_config,
        stage_names=full_execution_order,
        pipeline_name=config.get("General", {}).get("name", "Pipeline"),
    )

    if start:
        start_idx = execution_order.index(start)
        skipped_stages = execution_order[:start_idx]

        if skipped_stages:
            logger.info("=" * 60)
            logger.info(f"Resuming from stage '{start}'")
            logger.info(f"Skipping stages: {' → '.join(skipped_stages)}")
            logger.info("=" * 60)

            for skipped_stage in skipped_stages:
                stage_config = stage_map[skipped_stage]
                discovered = discover_stage_outputs(
                    stage_name=skipped_stage,
                    stage_config=stage_config,
                    io_config=io_config,
                )

                if discovered:
                    resolved_outputs[skipped_stage] = discovered
                    logger.info(
                        f"Discovered outputs from '{skipped_stage}': "
                        f"{list(discovered.keys())}"
                    )
                else:
                    logger.warning(
                        f"No outputs found for skipped stage '{skipped_stage}'. "
                        f"This may cause errors if later stages depend on it."
                    )

                stages_info.append(
                    {
                        "name": skipped_stage,
                        "cli": stage_config.get("cli"),
                        "duration": timing_cache.get(skipped_stage),
                        "success": True,
                        "outputs": resolved_outputs.get(skipped_stage, {}),
                        "skipped": "restart",
                    }
                )
                write_qc(record=stages_info[-1])

        execution_order = execution_order[start_idx:]
        logger.info(f"Executing stages: {' → '.join(execution_order)}")
        logger.info("")

    # =====================================================================
    # User-provided mask priority injection
    # Pre-populate resolved_outputs for any stage that produces out_mask so
    # those stages are transparently skipped in favour of the user's mask.
    # This runs AFTER the --start block so it always wins over discovered
    # outputs as well.
    # =====================================================================

    user_mask_stages = set()
    if io_config.get("mask"):
        for stage in pipeline_stages:
            cli_name = stage.get("cli", "")
            if not cli_name:
                continue
            try:
                stage_outputs = introspect_workflow_outputs(cli_name)
            except Exception:
                continue
            if "out_mask" in stage_outputs:
                sname = stage["name"]
                resolved_outputs.setdefault(sname, {})["out_mask"] = io_config["mask"]
                user_mask_stages.add(sname)
                logger.info(
                    f"Skipping stage '{sname}' — using user-provided mask: "
                    f"{io_config['mask']}"
                )

    for stage_name in execution_order:
        stage_config = stage_map[stage_name]
        stage_start = time.perf_counter()

        if stage_name in optional_skips:
            stages_info.append(
                {
                    "name": stage_name,
                    "cli": stage_config.get("cli"),
                    "duration": 0.0,
                    "success": True,
                    "outputs": {},
                    "skipped": "optional",
                    "reason": optional_skips[stage_name],
                }
            )
            write_qc(record=stages_info[-1])
            continue

        if not force:
            already_done = discover_stage_outputs(
                stage_name=stage_name,
                stage_config=stage_config,
                io_config=io_config,
            )
            try:
                expected = introspect_workflow_outputs(stage_config.get("cli", ""))
            except Exception:
                expected = set()
            if already_done and expected and already_done.keys() >= expected:
                resolved_outputs[stage_name] = already_done
                stages_info.append(
                    {
                        "name": stage_name,
                        "cli": stage_config.get("cli"),
                        "duration": timing_cache.get(stage_name),
                        "success": True,
                        "outputs": already_done,
                        "skipped": "restart",
                    }
                )
                logger.info(f"Stage '{stage_name}' outputs already exist, skipping.")
                write_qc(record=stages_info[-1])
                continue

        if stage_name in user_mask_stages:
            stages_info.append(
                {
                    "name": stage_name,
                    "cli": stage_config.get("cli"),
                    "duration": 0.0,
                    "success": True,
                    "outputs": resolved_outputs.get(stage_name, {}),
                    "skipped": "user_mask",
                }
            )
            write_qc(record=stages_info[-1])
            continue

        try:
            outputs = execute_pipeline_stage(
                stage_name=stage_name,
                stage_config=stage_config,
                resolved_outputs=resolved_outputs,
                io_config=io_config,
                force=force,
            )
            resolved_outputs[stage_name] = outputs

            stage_duration = time.perf_counter() - stage_start
            stages_info.append(
                {
                    "name": stage_name,
                    "cli": stage_config.get("cli"),
                    "duration": stage_duration,
                    "success": True,
                    "outputs": outputs,
                }
            )
            write_qc(record=stages_info[-1])
            timing_cache[stage_name] = stage_duration
            _save_timing_cache(
                out_dir=out_dir, fingerprint=fingerprint, timing_data=timing_cache
            )

            if outputs:
                logger.debug(f"Stage '{stage_name}' outputs: {outputs}")

        except Exception as e:
            logger.error(f"Pipeline failed at stage '{stage_name}': {str(e)}")
            logger.error(traceback.format_exc())
            write_qc(
                record={
                    "name": stage_name,
                    "cli": stage_config.get("cli"),
                    "duration": time.perf_counter() - stage_start,
                    "success": False,
                    "outputs": {},
                },
                error=traceback.format_exc(),
            )
            raise PipelineError(
                f"Pipeline failed at stage '{stage_name}': {e}", stage=stage_name
            ) from e

    total_time = sum(
        s["duration"] for s in stages_info if s.get("duration") is not None
    )

    logger.info("=" * 60)
    logger.info("Pipeline Completed Successfully!")
    logger.info("=" * 60)

    execution_info = {
        "stages": stages_info,
        "total_time": total_time,
        "dag_visualization": dag_viz,
        "execution_order": full_execution_order,
        "optional_skips": optional_skips,
        "report": None,
        "dry_run": False,
    }

    if report_path is None:
        report_path = os.path.join(
            io_config.get("out_dir", "."), "pipeline_report.html"
        )

    try:
        report_path = report.generate_html_report(
            config=config,
            config_file_path=config_file_path or "config.toml",
            execution_info=execution_info,
            output_path=report_path,
            reports_dir=reports_dir,
        )
        execution_info["report"] = report_path
        logger.info(f"HTML report generated: {report_path}")
        logger.info(f"Per-stage QC reports: {os.path.join(reports_dir, 'index.html')}")
        logger.info("Open them in a browser, or serve with: dipy_auto <report.html>")
    except Exception as e:
        logger.warning(f"Failed to generate HTML report: {e}")
        logger.debug(traceback.format_exc())

    return execution_info


_ANALYSIS_LEVELS = ("participant", "group")

_BIDS_DERIVATIVE_MAP = {
    "out_fa": ("FA", "dwimap"),
    "out_md": ("MD", "dwimap"),
    "out_rd": ("RD", "dwimap"),
    "out_ad": ("AD", "dwimap"),
    "out_denoised": ("denoised", "dwi"),
    "out_mask": ("brain", "mask"),
    "out_tractogram": (None, "tractography"),
}


def _classify_inputs(*, input_files, bids_mode):
    """Sort the positional arguments of ``dipy_auto`` by kind.

    Recognised by extension: ``.toml`` configurations, DWI NIfTI files,
    b-value and b-vector files and HTML reports. A directory holding a
    ``dataset_description.json`` or a ``sub-*`` entry is a BIDS dataset
    root; in BIDS mode a second directory is the output directory and the
    literal ``participant`` or ``group`` is the analysis level.

    Parameters
    ----------
    input_files : sequence of str
        Positional arguments as given on the command line.
    bids_mode : bool
        Whether ``--bids`` was passed, which enables the BIDS-App style
        positional arguments.

    Returns
    -------
    dict
        Keys ``config_file``, ``dwi``, ``bvals``, ``bvecs``, ``report``,
        ``bids_dir``, ``output_dir`` and ``analysis_level``.
    """
    found = {
        "config_file": None,
        "dwi": None,
        "bvals": None,
        "bvecs": None,
        "report": None,
        "bids_dir": None,
        "output_dir": None,
        "analysis_level": None,
    }

    for entry in input_files:
        path = Path(entry)
        name_lower = path.name.lower()
        full_suffix = "".join(path.suffixes).lower()

        if re.match(r".*report.*\.html$", name_lower):
            found["report"] = found["report"] or entry
        elif full_suffix == ".toml":
            found["config_file"] = entry
        elif full_suffix in (".nii.gz", ".nii"):
            found["dwi"] = entry
        elif path.suffix.lower() in (".bval", ".bvals"):
            found["bvals"] = entry
        elif path.suffix.lower() in (".bvec", ".bvecs"):
            found["bvecs"] = entry
        elif name_lower in _ANALYSIS_LEVELS and not path.exists():
            found["analysis_level"] = name_lower
        elif found["bids_dir"] is None and looks_like_dataset(path=entry):
            found["bids_dir"] = entry
        elif bids_mode and found["output_dir"] is None:
            found["output_dir"] = entry
        else:
            logger.warning(f"Unrecognized file type, ignoring: {entry}")

    return found


def _fill_io_section(*, config, overrides, out_dir, out_report):
    """Apply input overrides to the ``[io]`` section of a configuration.

    Parameters
    ----------
    config : dict
        Pipeline configuration, modified in place.
    overrides : dict
        Values to write into the ``[io]`` section. Empty values are
        ignored, so a configuration file keeps its own paths.
    out_dir : str
        Output directory of the run.
    out_report : str
        Name of the report file inside ``out_dir``.

    Returns
    -------
    dict
        The ``[io]`` section, with an absolute ``out_dir`` that exists.
    """
    config.setdefault("io", {})
    io_config = config["io"]

    for key, value in overrides.items():
        if value in (None, ""):
            continue
        existing = io_config.get(key, "")
        if existing and existing != value:
            logger.warning(f"Overriding {key}: {existing} → {value}")
        io_config[key] = value

    io_config["out_dir"] = os.path.abspath(out_dir or io_config.get("out_dir", "."))
    io_config["out_report"] = os.path.join(io_config["out_dir"], out_report)
    os.makedirs(io_config["out_dir"], exist_ok=True)

    return io_config


def _write_pipeline_config(*, config, out_dir, timestamp, pipeline_name):
    """Write the configuration a run is about to execute.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    out_dir : str
        Directory to write into.
    timestamp : str
        Timestamp prefixed to the file name.
    pipeline_name : str
        Pipeline name used in the file name.

    Returns
    -------
    str
        Absolute path of the written configuration.
    """
    config_file = os.path.join(out_dir, f"{timestamp}_{pipeline_name}_pipeline.toml")

    with open(config_file, "w") as f:
        f.write(format_toml_config(config))

    logger.info(f"Configuration saved to: {config_file}")
    return config_file


def _resolve_atlas_paths(*, config, io_config):
    """Download the HCP atlas when a stage needs it and no path was given.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    io_config : dict
        The ``[io]`` section, updated in place with the atlas paths.

    Raises
    ------
    PipelineError
        If the atlas is required but cannot be fetched.
    """
    needs_atlas = any(
        stage.get("cli", "") in ("dipy_slr", "dipy_recobundles")
        for stage in config.get("pipeline", [])
    )

    if not needs_atlas:
        return

    atlas_tractogram = io_config.get("atlas_tractogram", "")
    bundle_atlas_dir = io_config.get("bundle_atlas_dir", "")

    if atlas_tractogram and bundle_atlas_dir:
        return

    logger.info("Atlas required for SLR/RecoBundles but not provided.")
    logger.info("Downloading HCP 30-bundle atlas...")

    try:
        _, atlas_base = fetch_30_bundle_atlas_hcp842()
    except Exception as e:
        logger.error(f"Failed to download atlas: {e}")
        logger.error(
            "Please provide atlas paths manually using --atlas_tractogram "
            "and --bundle_atlas_dir"
        )
        raise PipelineError(f"Failed to download atlas: {e}") from e

    if not atlas_tractogram:
        io_config["atlas_tractogram"] = os.path.join(
            atlas_base, "Atlas_30_Bundles", "whole_brain", "whole_brain_MNI.trk"
        )
        logger.info(f"Atlas tractogram: {io_config['atlas_tractogram']}")

    if not bundle_atlas_dir:
        io_config["bundle_atlas_dir"] = os.path.join(
            atlas_base, "Atlas_30_Bundles", "bundles"
        )
        logger.info(f"Bundle atlas directory: {io_config['bundle_atlas_dir']}")


def _bids_run_out_dirs(*, runs, deriv_root):
    """Give each acquisition its own output directory.

    Several acquisitions of the same session share a datatype directory and
    would overwrite each other, since stage output names carry no entity.
    Those get a subdirectory named after the entities that tell them apart;
    a lone acquisition writes straight into the datatype directory.

    Parameters
    ----------
    runs : sequence of DWIRun
        Acquisitions to process.
    deriv_root : str
        Root of the derivative dataset.

    Returns
    -------
    list of str
        One output directory per acquisition, in the same order.
    """
    datatype_dirs = [derivative_path(deriv_root=deriv_root, run=run) for run in runs]
    counts = Counter(datatype_dirs)
    out_dirs = []

    for run, datatype_dir in zip(runs, datatype_dirs):
        if counts[datatype_dir] == 1:
            out_dirs.append(datatype_dir)
            continue

        extra = "_".join(
            f"{key}-{run.entities[key]}"
            for key in BIDS_ENTITY_ORDER
            if key in run.entities and key not in ("sub", "ses")
        )
        out_dirs.append(
            os.path.join(datatype_dir, extra or split_extension(filename=run.dwi)[0])
        )

    return out_dirs


def _write_bids_derivative_links(*, run, deriv_root, execution_info):
    """Expose headline outputs under BIDS-style names.

    A symlink (a copy where symlinks are unavailable) is created at the
    datatype level for each output of ``_BIDS_DERIVATIVE_MAP``, with a JSON
    sidecar recording its provenance. Nothing is moved, so stage
    resumption and the QC reports keep working on the original paths.

    Parameters
    ----------
    run : DWIRun
        Resolved diffusion acquisition.
    deriv_root : str
        Root of the derivative dataset.
    execution_info : dict or None
        Return value of :func:`execute_semantic_pipeline`.

    Returns
    -------
    list of str
        Paths of the links that were created.
    """
    datatype_dir = derivative_path(deriv_root=deriv_root, run=run)
    entities = {
        key: value
        for key, value in run.entities.items()
        if key not in ("suffix", "extension")
    }
    created = []

    for stage in (execution_info or {}).get("stages", []):
        for key, value in (stage.get("outputs") or {}).items():
            if key not in _BIDS_DERIVATIVE_MAP:
                continue
            if not isinstance(value, (str, os.PathLike)):
                continue

            path = os.fspath(value)
            if not os.path.isfile(path):
                continue

            description, bids_suffix = _BIDS_DERIVATIVE_MAP[key]
            _, extension = split_extension(filename=path)

            link_entities = dict(entities)
            link_entities["desc"] = description or stage.get("name", "dipy")

            name = build_entity_name(
                entities=link_entities, suffix=bids_suffix, extension=extension
            )
            link = os.path.join(datatype_dir, name)

            try:
                if os.path.lexists(link):
                    os.unlink(link)
                os.symlink(os.path.relpath(path, datatype_dir), link)
            except OSError:
                shutil.copy2(path, link)

            sidecar = {
                "Sources": [f"bids::{os.path.relpath(path, deriv_root)}"],
                "RawSources": [f"bids::{os.path.relpath(run.dwi, run.root)}"],
                "GeneratedByStage": {
                    "name": stage.get("name"),
                    "cli": stage.get("cli"),
                    "output": key,
                },
            }
            with open(
                os.path.join(datatype_dir, f"{name[: -len(extension)]}.json"), "w"
            ) as f:
                json.dump(sidecar, f, indent=4)

            created.append(link)
            logger.info(f"BIDS derivative: {os.path.basename(link)} → {path}")

    return created


def _write_batch_summary(*, results, deriv_root, timestamp, pipeline_name, bids_dir):
    """Write the JSON and HTML summaries of a BIDS batch.

    Parameters
    ----------
    results : list of dict
        One record per processed acquisition.
    deriv_root : str
        Root of the derivative dataset.
    timestamp : str
        Timestamp shared by every artifact of the batch.
    pipeline_name : str
        Name of the pipeline that was run.
    bids_dir : str
        Root of the source BIDS dataset.

    Returns
    -------
    json_path : str
        Path of the JSON summary.
    html_path : str or None
        Path of the HTML summary, None when it could not be generated.
    """
    json_path = os.path.join(deriv_root, f"{timestamp}_batch_summary.json")

    with open(json_path, "w") as f:
        json.dump(
            {
                "pipeline": pipeline_name,
                "bids_dir": bids_dir,
                "deriv_root": deriv_root,
                "timestamp": timestamp,
                "runs": results,
            },
            f,
            indent=4,
        )

    logger.info(f"Batch summary: {json_path}")
    html_path = os.path.join(deriv_root, f"{timestamp}_batch_summary.html")

    try:
        report.generate_batch_report(
            results=results,
            output_path=html_path,
            pipeline_name=pipeline_name,
            bids_dir=bids_dir,
        )
        logger.info(f"Batch report: {html_path}")
    except Exception as e:
        logger.warning(f"Failed to generate the batch report: {e}")
        logger.debug(traceback.format_exc())
        html_path = None

    return json_path, html_path


_NON_PATH_IO_KEYS = frozenset(
    {
        "pe_dir",
        "total_readout_time",
        "effective_echo_spacing",
        "echo_time",
        "repetition_time",
        "subject",
        "session",
        "entities",
        "out_report",
    }
)


def _validate_io_refs(*, config, io_config):
    """Check that every ``${io.x}`` reference of a pipeline resolves to a file.

    Keys listed in ``_NON_PATH_IO_KEYS`` hold scalar metadata rather than
    paths and are never checked against the filesystem. An empty entry used
    only by optional stages is not an error: those stages are skipped.

    Parameters
    ----------
    config : dict
        Full pipeline configuration.
    io_config : dict
        Values of the ``[io]`` section.

    Returns
    -------
    missing : list of tuple
        ``(reference, key)`` pairs whose ``[io]`` entry is absent or empty.
    invalid : list of tuple
        ``(reference, path)`` pairs whose ``[io]`` entry does not exist.
    """
    required_refs = set()
    optional_refs = set()

    for stage in config.get("pipeline", []):
        for key, value in stage.items():
            if key in ("name", "cli") or not isinstance(value, str):
                continue
            for ref in extract_variable_references(value):
                if not ref.startswith("io."):
                    continue
                io_key = ref.split(".", 1)[1]
                if io_key.startswith("out_") or io_key in _NON_PATH_IO_KEYS:
                    continue
                if stage.get("optional", False):
                    optional_refs.add((ref, io_key))
                else:
                    required_refs.add((ref, io_key))

    missing = []
    invalid = []

    for ref, io_key in sorted(required_refs | optional_refs):
        path = str(io_config.get(io_key, "") or "")
        if not path.strip():
            if (ref, io_key) in required_refs:
                missing.append((ref, io_key))
        elif not os.path.exists(path):
            invalid.append((ref, path))

    return missing, invalid


def _log_io_validation_errors(*, missing, invalid, label=None):
    """Report the outcome of :func:`_validate_io_refs` to the log.

    Parameters
    ----------
    missing : list of tuple
        ``(reference, key)`` pairs with no value in the ``[io]`` section.
    invalid : list of tuple
        ``(reference, path)`` pairs pointing at a missing file.
    label : str, optional
        Dataset entry the errors belong to, prefixed to the header when
        several entries are processed in one run.
    """
    header = (
        "Input Validation Failed"
        if label is None
        else (f"Input Validation Failed for {label}")
    )
    logger.error("=" * 70)
    logger.error(header)
    logger.error("=" * 70)

    if missing:
        logger.error("✗ Missing input file definitions in [io] section:")
        for ref, io_key in missing:
            logger.error(f"  • {io_key} (required by ${{{ref}}})")

        logger.error("Please provide input files using one of these methods:")
        logger.error("  1. Edit the config file and set paths in the [io] section")
        logger.error("  2. Pass them as positional arguments, for instance:")
        logger.error("     dipy_auto dwi.nii.gz dwi.bval dwi.bvec")
        logger.error("  3. Run over a BIDS dataset: dipy_auto <bids_dir> --bids")

    if invalid:
        logger.error("✗ Input files do not exist:")
        for ref, path in invalid:
            logger.error(f"  • {path} (from ${{{ref}}})")

    logger.error("=" * 70)


def _validate_mask_file(mask_file):
    """Validate that a mask file is binary and has sufficient brain coverage.

    Parameters
    ----------
    mask_file : str
        Path to the NIfTI mask file.

    Raises
    ------
    SystemExit
        If the file cannot be loaded, is not binary, or covers less than
        25 % of the volume.
    """
    try:
        data = load_nifti_data(mask_file)
    except Exception as e:
        logger.error(f"Cannot load mask file '{mask_file}': {e}")
        sys.exit(1)

    unique_vals = np.unique(data)
    non_zero = unique_vals[unique_vals != 0]
    if len(non_zero) > 1:
        logger.error(
            f"Mask '{mask_file}' is not binary: "
            f"found {len(unique_vals)} unique values ({unique_vals[:8]}...)."
        )
        sys.exit(1)

    coverage = float((data > 0).sum()) / data.size
    if coverage < 0.25:
        logger.error(
            f"Mask '{mask_file}' has insufficient coverage: "
            f"{coverage:.1%} of voxels are non-zero (minimum 25 % required)."
        )
        sys.exit(1)

    logger.info(f"✓ Mask validated: binary, {coverage:.1%} coverage — {mask_file}")


# =============================================================================
# AutoFlow Workflow
# =============================================================================
class _ReportHandler(http.server.SimpleHTTPRequestHandler):
    """Serve a report and launch Skyline on the files it lists.

    The pages call ``/qc-open?path=...``; anything else is served statically.
    Only files under the served directory can be opened, and the response
    body is the message the page shows when the launch fails.
    """

    def do_GET(self):  # noqa: N802 — http.server API
        if urllib.parse.urlparse(self.path).path != "/qc-open":
            super().do_GET()
            return

        query = urllib.parse.urlparse(self.path).query
        target = urllib.parse.parse_qs(query).get("path", [""])[0]
        skyline = shutil.which("dipy_skyline")
        root = Path(self.directory).resolve()
        path = Path(target).resolve()

        if skyline is None:
            self._reply(
                404,
                "Skyline (dipy_skyline) was not found in this environment. "
                "Install it, or run the command below yourself.",
            )
        elif not path.is_file() or root not in path.parents:
            self._reply(403, f"'{target}' is not a file served by this report.")
        else:
            subprocess.Popen([skyline, str(path)])
            self._reply(200, f"Skyline launched on {path.name}.")

    def _reply(self, code, message):
        body = message.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve_html_report(report_path):
    """Serve an HTML pipeline report via a local HTTP server and open it.

    The server is rooted at the pipeline output directory, so that a stage
    page under ``reports/`` can still reach the outputs it links to. It opens
    the default browser and blocks until the user presses Ctrl+C.

    Parameters
    ----------
    report_path : str or Path
        Path to the HTML report file (e.g., ``pipeline_report.html``).
    """
    report_path = Path(report_path).resolve()
    root = report_path.parent
    if root.name == "reports":
        root = root.parent
    report_dir = str(root)
    report_name = str(report_path.relative_to(root))

    host = "127.0.0.1"
    port = 8000
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                break
            except OSError:
                port += 1

    handler = partial(_ReportHandler, directory=report_dir)
    url = f"http://localhost:{port}/{report_name}"
    logger.info(f"Serving report at: {url}")
    logger.info("Press Ctrl+C to stop the server.")
    webbrowser.open(url)
    with socketserver.TCPServer((host, port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped.")


# =============================================================================
# AutoFlow Workflow
# =============================================================================
# TODO:
# - Done: handle maskfile, check binary, % of white voxels.
# - Add opposite phase encoding (AP/PA)
# - Done: handle incorrect --start name.
# - Done: Create dipy_brain_mask workflow and add to pipelines
#   (SynthSeg, median_otsu, PUMBA, EVAC+)
# - Done: Create dipy_denoise and use it in the pipelines
# - dipy cluster qbx


class AutoFlow(Workflow):
    """Automatic pipeline execution workflow with semantic DAG-based wiring.

    This workflow supports:
    - Semantic pipeline specification via [[pipeline]] TOML sections
    - Automatic DAG construction and topological execution
    - Interactive mode for dynamic pipeline construction
    - Predefined pipeline templates
    - Dry-run mode for testing configurations
    """

    @classmethod
    def get_short_name(cls):
        """Return short name for the workflow.

        Returns
        -------
        str
            Short name 'auto'.
        """
        return "auto"

    def run(
        self,
        input_files,
        t1_file=None,
        mask_file=None,
        dwi_opp_phase_file=None,
        bids=False,
        bids_folder=None,
        participant_label=None,
        session_label=None,
        acquisition_label=None,
        run_label=None,
        stop_on_error=False,
        atlas_tractogram=None,
        bundle_atlas_dir=None,
        interactive_mode=False,
        pipeline_type=None,
        start=None,
        dry_run=False,
        list_pipelines=False,
        out_dir="",
        out_report="reports.toml",
    ):
        """Run diffusion pipelines based on configuration files.

        Parameters
        ----------
        input_files : variable str
            Input files passed to the pipeline. File types are detected
            automatically from their extensions (case-insensitive):

            - ``.toml`` — pipeline configuration file.
            - ``.nii`` / ``.nii.gz`` — DWI NIfTI file.
            - ``.bval`` / ``.bvals`` — b-values file.
            - ``.bvec`` / ``.bvecs`` — b-vectors file.
            - ``*report*.html`` — HTML report; all other inputs are ignored
              and the report is served via a local HTTP server.

            With ``--bids``, the arguments follow the BIDS-App convention
            instead: ``dipy_auto <bids_dir> [<output_dir>] [participant]``.
        t1_file : str, optional
            Path to the T1 file. Resolved from the dataset with ``--bids``.
        mask_file : str, optional
            Path to the brain mask file.
        dwi_opp_phase_file : str, optional
            Path to the opposite phase DWI file (for phase correction).
            Resolved from the fieldmaps of the dataset with ``--bids``.
        bids : bool, optional
            Treat the inputs as a BIDS dataset and run the pipeline on every
            diffusion acquisition it contains, writing BIDS derivatives.
        bids_folder : str, optional
            Path to the BIDS dataset root, as an alternative to passing it
            positionally. Requires ``--bids``.
        participant_label : variable str, optional
            Subject labels to process, with or without the ``sub-`` prefix.
            All subjects are processed when omitted.
        session_label : variable str, optional
            Session labels to process, with or without the ``ses-`` prefix.
        acquisition_label : variable str, optional
            ``acq-`` labels to process.
        run_label : variable str, optional
            ``run-`` labels to process.
        stop_on_error : bool, optional
            Stop a BIDS batch at the first acquisition that fails instead of
            reporting it and carrying on.
        atlas_tractogram : str, optional
            Path to atlas tractogram for registration (dipy_slr).
            If not provided and dipy_slr is in the pipeline, will auto-download
            HCP 30-bundle atlas.
        bundle_atlas_dir : str, optional
            Path to bundle atlas directory for segmentation (dipy_recobundles).
            If not provided and dipy_recobundles is in the pipeline,
            will auto-download HCP 30-bundle atlas.
        interactive_mode : bool, optional
            Enable interactive mode for pipeline customization.
        pipeline_type : str, optional
            Predefined pipeline type (e.g., 'denoise', 'preprocessing', 'dti_only').
            Use --list-pipelines to see all available pipelines.
        start : str, optional
            Stage name to start execution from. Earlier stages will be skipped,
            but their outputs must exist on disk. Use this to resume a failed
            pipeline or re-run later stages with different parameters.
        dry_run : bool, optional
            Show execution plan without running commands.
        list_pipelines : bool, optional
            List all available predefined pipelines and exit.
        out_dir : str, optional
            Output directory. With ``--bids``, the root of the derivative
            dataset, defaulting to ``<bids_dir>/derivatives/dipy``.
        out_report : str, optional
            Path to the report file.
        """
        if list_pipelines:
            current_log_level = logger.getEffectiveLevel()
            logger.info(
                templates.list_pipelines_with_descriptions(log_level=current_log_level)
            )
            return

        inputs = _classify_inputs(input_files=input_files, bids_mode=bids)

        if inputs["report"]:
            logger.info(
                f"HTML report detected: {inputs['report']}. "
                "Ignoring all other inputs and opening report."
            )
            _serve_html_report(inputs["report"])
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bids_dir = bids_folder or inputs["bids_dir"]

        if bids_dir and not bids:
            logger.error("=" * 70)
            logger.error(
                f"BIDS dataset detected at {bids_dir} but --bids was not given"
            )
            logger.error("=" * 70)
            logger.error("Re-run in BIDS mode:")
            logger.error(f"  dipy_auto {bids_dir} <output_dir> participant --bids")
            logger.error(
                "Optional filters: --participant_label 01 02 --session_label 1 "
                "--acquisition_label b1000 --run_label 1"
            )
            logger.error("=" * 70)
            sys.exit(1)

        if bids:
            if not bids_dir:
                logger.error(
                    "--bids requires a BIDS dataset directory: "
                    "dipy_auto <bids_dir> [<output_dir>] participant --bids"
                )
                sys.exit(1)

            if inputs["analysis_level"] == "group":
                logger.error(
                    "Group level analysis is not implemented. Use the "
                    "'participant' analysis level."
                )
                sys.exit(1)

            conflicting = {
                "DWI file": inputs["dwi"],
                "b-values file": inputs["bvals"],
                "b-vectors file": inputs["bvecs"],
                "--t1_file": t1_file,
                "--mask_file": mask_file,
                "--dwi_opp_phase_file": dwi_opp_phase_file,
            }
            conflicting = {k: v for k, v in conflicting.items() if v}

            if conflicting:
                logger.error(
                    "Single subject inputs cannot be combined with --bids; "
                    "they are resolved from the dataset instead:"
                )
                for name, value in conflicting.items():
                    logger.error(f"  • {name}: {value}")
                sys.exit(1)

            if interactive_mode:
                logger.error("--interactive_mode is not supported with --bids")
                sys.exit(1)

            deriv_root = (
                inputs["output_dir"]
                or out_dir
                or os.path.join(bids_dir, "derivatives", "dipy")
            )

            return self._run_bids_batch(
                bids_dir=os.path.abspath(bids_dir),
                deriv_root=os.path.abspath(deriv_root),
                config_file=inputs["config_file"],
                pipeline_type=pipeline_type,
                participant_label=participant_label,
                session_label=session_label,
                acquisition_label=acquisition_label,
                run_label=run_label,
                atlas_tractogram=atlas_tractogram,
                bundle_atlas_dir=bundle_atlas_dir,
                start=start,
                dry_run=dry_run,
                stop_on_error=stop_on_error,
                out_report=out_report,
                timestamp=timestamp,
            )

        return self._run_single(
            config_file=inputs["config_file"],
            dwi_file=inputs["dwi"],
            bvals_file=inputs["bvals"],
            bvecs_file=inputs["bvecs"],
            t1_file=t1_file,
            mask_file=mask_file,
            dwi_opp_phase_file=dwi_opp_phase_file,
            atlas_tractogram=atlas_tractogram,
            bundle_atlas_dir=bundle_atlas_dir,
            interactive_mode=interactive_mode,
            pipeline_type=pipeline_type,
            start=start,
            dry_run=dry_run,
            out_dir=os.path.abspath(out_dir),
            out_report=out_report,
            timestamp=timestamp,
        )

    def _load_pipeline_config(
        self, *, config_file, pipeline_type, interactive_mode=False, bvals_file=None
    ):
        """Load the pipeline configuration from a file or a template.

        Parameters
        ----------
        config_file : str or None
            Path to a TOML configuration. When None, a predefined template
            is used.
        pipeline_type : str or None
            Name of the predefined pipeline. None selects ``full``.
        interactive_mode : bool, optional
            Build the pipeline interactively instead of using a template.
        bvals_file : str, optional
            b-values file analysed by the interactive builder.

        Returns
        -------
        config : dict
            Parsed configuration.
        pipeline_name : str
            Name used for the generated configuration file.
        """
        if config_file is not None:
            config_file = os.path.abspath(config_file)

            if not config_file.endswith(".toml"):
                logger.error(f"Configuration file must be TOML: {config_file}")
                sys.exit(1)

            logger.info(f"Running pipeline from {os.path.basename(config_file)}")

            try:
                with open(config_file, "rb") as f:
                    config = toml.load(f)
            except FileNotFoundError:
                logger.error(f"Configuration file not found: {config_file}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error decoding TOML file: {e}")
                sys.exit(1)

            name = config.get("General", {}).get("name", "custom")
            return config, name

        if interactive_mode and pipeline_type:
            logger.warning("Both --interactive and --pipeline_type specified.")
            choice = input(
                "Choose: [1] Use interactive mode, [2] Use pipeline_type: "
            ).strip()
            if choice == "1":
                pipeline_type = None
            elif choice == "2":
                interactive_mode = False
            else:
                logger.error("Invalid choice. Defaulting to pipeline_type.")
                interactive_mode = False

        if interactive_mode:
            if not bvals_file or not os.path.exists(bvals_file):
                logger.error(
                    "Interactive mode requires a b-values file to analyze "
                    "data characteristics."
                )
                sys.exit(1)

            try:
                data_chars = interactive.analyze_bvals(bvals_file=bvals_file)
                interactive.print_data_summary(data_chars=data_chars)
                selected_pipeline = interactive.interactive_pipeline_selection(
                    data_chars=data_chars
                )

                if selected_pipeline is None:
                    logger.info("Building custom pipeline interactively...")
                    config = interactive.build_interactive_pipeline_config(
                        data_chars=data_chars
                    )
                    return config, "interactive"

                pipeline_type = selected_pipeline
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                traceback.print_exc()
                sys.exit(1)

        if pipeline_type:
            try:
                logger.info(f"Loading predefined pipeline: {pipeline_type}")
                config_str = templates.get_predefined_pipeline(
                    pipeline_name=pipeline_type
                )
            except KeyError:
                logger.error(f"Unknown pipeline type: {pipeline_type}")
                current_log_level = logger.getEffectiveLevel()
                logger.info(
                    templates.list_pipelines_with_descriptions(
                        log_level=current_log_level
                    )
                )
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error loading predefined pipeline: {e}")
                sys.exit(1)
        else:
            logger.info("Using default 'full' pipeline")
            try:
                config_str = templates.get_predefined_pipeline(pipeline_name="full")
            except Exception as e:
                logger.error(f"Error loading default pipeline: {e}")
                sys.exit(1)

        return toml.loads(config_str), pipeline_type or "default"

    def _run_single(
        self,
        *,
        config_file,
        dwi_file,
        bvals_file,
        bvecs_file,
        t1_file,
        mask_file,
        dwi_opp_phase_file,
        atlas_tractogram,
        bundle_atlas_dir,
        interactive_mode,
        pipeline_type,
        start,
        dry_run,
        out_dir,
        out_report,
        timestamp,
    ):
        """Run one pipeline on one set of input files.

        Parameters
        ----------
        config_file : str or None
            Path to a TOML configuration, or None to use a template.
        dwi_file, bvals_file, bvecs_file : str or None
            Diffusion input files.
        t1_file, mask_file, dwi_opp_phase_file : str or None
            Optional companion images.
        atlas_tractogram, bundle_atlas_dir : str or None
            Atlas paths, downloaded when needed and not provided.
        interactive_mode : bool
            Build the pipeline interactively.
        pipeline_type : str or None
            Name of the predefined pipeline.
        start : str or None
            Stage to start execution from.
        dry_run : bool
            Show the execution plan without running it.
        out_dir : str
            Output directory.
        out_report : str
            Name of the report file inside ``out_dir``.
        timestamp : str
            Timestamp prefixed to the generated artifacts.
        """
        config, pipeline_name = self._load_pipeline_config(
            config_file=config_file,
            pipeline_type=pipeline_type,
            interactive_mode=interactive_mode,
            bvals_file=bvals_file,
        )

        if "pipeline" not in config or not isinstance(config["pipeline"], list):
            logger.error("Configuration missing [[pipeline]] sections")
            logger.error("Expected TOML structure with [[pipeline]] array sections")
            sys.exit(1)

        if mask_file:
            _validate_mask_file(mask_file)

        io_config = _fill_io_section(
            config=config,
            overrides={
                "dwi": dwi_file,
                "bvals": bvals_file,
                "bvecs": bvecs_file,
                "t1w": t1_file,
                "mask": mask_file,
                "dwi_opp_phase": dwi_opp_phase_file,
                "atlas_tractogram": atlas_tractogram,
                "bundle_atlas_dir": bundle_atlas_dir,
            },
            out_dir=out_dir,
            out_report=out_report,
        )

        log_file_path = os.path.join(out_dir, f"{timestamp}_dipy_auto.log")
        with file_handler(log_file_path, level=logger.getEffectiveLevel()):
            logger.info(f"Log file: {log_file_path}")

            if config_file is None:
                config_file = _write_pipeline_config(
                    config=config,
                    out_dir=out_dir,
                    timestamp=timestamp,
                    pipeline_name=pipeline_name,
                )

            try:
                _resolve_atlas_paths(config=config, io_config=io_config)
                missing, invalid = _validate_io_refs(config=config, io_config=io_config)
                if missing or invalid:
                    _log_io_validation_errors(missing=missing, invalid=invalid)
                    raise PipelineError("Input validation failed")

                logger.info("✓ All input files validated successfully")
                logger.info("Pipeline Configuration Summary:")
                logger.info(f"  Config file: {os.path.basename(config_file)}")
                logger.info(f"  Output directory: {io_config['out_dir']}")
                logger.info(f"  Number of stages: {len(config['pipeline'])}")
                logger.info(f"  Dry run: {dry_run}")
                logger.info("")

                execute_semantic_pipeline(
                    config=config,
                    io_config=io_config,
                    config_file_path=config_file,
                    report_path=os.path.join(
                        io_config["out_dir"], f"{timestamp}_pipeline_report.html"
                    ),
                    start=start,
                    dry_run=dry_run,
                    force=self._force_overwrite,
                )
            except PipelineError as e:
                logger.error(str(e))
                sys.exit(1)

    def _run_bids_batch(
        self,
        *,
        bids_dir,
        deriv_root,
        config_file,
        pipeline_type,
        participant_label,
        session_label,
        acquisition_label,
        run_label,
        atlas_tractogram,
        bundle_atlas_dir,
        start,
        dry_run,
        stop_on_error,
        out_report,
        timestamp,
    ):
        """Run one pipeline on every diffusion acquisition of a BIDS dataset.

        Each acquisition is processed independently into
        ``<deriv_root>/sub-X[/ses-Y]/dwi``, and a failure is reported in the
        batch summary instead of stopping the batch.

        Parameters
        ----------
        bids_dir : str
            Root of the BIDS dataset.
        deriv_root : str
            Root of the derivative dataset to write into.
        config_file : str or None
            Path to a TOML configuration, or None to use a template.
        pipeline_type : str or None
            Name of the predefined pipeline.
        participant_label, session_label, acquisition_label, run_label : \
sequence of str or None
            Entity filters restricting the acquisitions to process.
        atlas_tractogram, bundle_atlas_dir : str or None
            Atlas paths, downloaded when needed and not provided.
        start : str or None
            Stage to start execution from.
        dry_run : bool
            Show the execution plan of every run without running it.
        stop_on_error : bool
            Stop the batch at the first failing acquisition.
        out_report : str
            Name of the report file inside each run directory.
        timestamp : str
            Timestamp shared by every artifact of the batch.

        Returns
        -------
        list of dict
            One result record per acquisition.
        """
        try:
            description = validate_dataset(root=bids_dir)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

        logger.info(f"BIDS dataset: {description.get('Name', bids_dir)}")

        runs = find_dwi_runs(
            root=bids_dir,
            participant_label=participant_label,
            session_label=session_label,
            acquisition_label=acquisition_label,
            run_label=run_label,
        )

        if not runs:
            logger.error(f"No diffusion acquisition matched in {bids_dir}")
            for name, value in (
                ("--participant_label", participant_label),
                ("--session_label", session_label),
                ("--acquisition_label", acquisition_label),
                ("--run_label", run_label),
            ):
                if value:
                    logger.error(f"  {name} {' '.join(str(v) for v in value)}")
            subjects = list_subjects(root=bids_dir)
            logger.error(f"Subjects present: {', '.join(subjects) or 'none'}")
            for subject in subjects:
                sessions = list_sessions(root=bids_dir, subject=subject)
                if sessions:
                    logger.error(f"  sub-{subject} sessions: {', '.join(sessions)}")
            sys.exit(1)

        base_config, pipeline_name = self._load_pipeline_config(
            config_file=config_file, pipeline_type=pipeline_type
        )

        if "pipeline" not in base_config or not isinstance(
            base_config["pipeline"], list
        ):
            logger.error("Configuration missing [[pipeline]] sections")
            logger.error("Expected TOML structure with [[pipeline]] array sections")
            sys.exit(1)

        os.makedirs(deriv_root, exist_ok=True)
        write_dataset_description(
            path=deriv_root,
            source_datasets=[bids_dir],
            pipeline_name=pipeline_name,
        )

        batch_log = os.path.join(deriv_root, f"{timestamp}_dipy_auto.log")
        with file_handler(batch_log, level=logger.getEffectiveLevel()):
            logger.info(f"Batch log file: {batch_log}")
            logger.info(f"Derivatives root: {deriv_root}")
            logger.info(f"{len(runs)} diffusion acquisition(s) to process:")
            for run in runs:
                logger.info(f"  {run.label}")

            results = []
            run_out_dirs = _bids_run_out_dirs(runs=runs, deriv_root=deriv_root)

            for index, (run, run_out_dir) in enumerate(
                zip(runs, run_out_dirs), start=1
            ):
                logger.info("")
                logger.info("=" * 70)
                logger.info(f"[{index}/{len(runs)}] {run.label}")
                logger.info("=" * 70)

                result = self._run_bids_entry(
                    run=run,
                    run_out_dir=run_out_dir,
                    base_config=base_config,
                    deriv_root=deriv_root,
                    pipeline_name=pipeline_name,
                    atlas_tractogram=atlas_tractogram,
                    bundle_atlas_dir=bundle_atlas_dir,
                    start=start,
                    dry_run=dry_run,
                    out_report=out_report,
                    timestamp=timestamp,
                )
                results.append(result)

                if result["status"] == "failed" and stop_on_error:
                    logger.error(f"Stopping batch after {run.label} (--stop_on_error)")
                    break

            _write_batch_summary(
                results=results,
                deriv_root=deriv_root,
                timestamp=timestamp,
                pipeline_name=pipeline_name,
                bids_dir=bids_dir,
            )

            failed = [r for r in results if r["status"] == "failed"]
            succeeded = [r for r in results if r["status"] == "completed"]

            logger.info("")
            logger.info("=" * 70)
            logger.info(
                f"Batch finished: {len(results)} run(s), {len(succeeded)} succeeded, "
                f"{len(failed)} failed"
            )
            for result in failed:
                stage = result["failed_stage"]
                where = f" at stage '{stage}'" if stage else ""
                logger.error(f"  ✗ {result['label']} failed{where}: {result['error']}")
            logger.info("=" * 70)

            if failed:
                sys.exit(1)

            return results

    def _run_bids_entry(
        self,
        *,
        run,
        run_out_dir,
        base_config,
        deriv_root,
        pipeline_name,
        atlas_tractogram,
        bundle_atlas_dir,
        start,
        dry_run,
        out_report,
        timestamp,
    ):
        """Run the pipeline on one diffusion acquisition of a BIDS dataset.

        Parameters
        ----------
        run : DWIRun
            Resolved diffusion acquisition.
        run_out_dir : str
            Output directory of this acquisition.
        base_config : dict
            Configuration shared by the batch. Copied before use.
        deriv_root : str
            Root of the derivative dataset.
        pipeline_name : str
            Name used for the generated configuration file.
        atlas_tractogram, bundle_atlas_dir : str or None
            Atlas paths, downloaded when needed and not provided.
        start : str or None
            Stage to start execution from.
        dry_run : bool
            Show the execution plan without running it.
        out_report : str
            Name of the report file inside the run directory.
        timestamp : str
            Timestamp shared by every artifact of the batch.

        Returns
        -------
        dict
            Record with the keys ``label``, ``subject``, ``session``,
            ``entities``, ``out_dir``, ``status``, ``failed_stage``,
            ``error``, ``duration``, ``metadata`` and ``report``.
        """
        metadata = phase_encoding_info(run=run)
        record = {
            "label": run.label,
            "subject": run.subject,
            "session": run.session,
            "entities": dict(run.entities),
            "out_dir": run_out_dir,
            "status": "failed",
            "failed_stage": None,
            "error": None,
            "duration": 0.0,
            "metadata": metadata,
            "report": None,
        }

        started = time.perf_counter()
        os.makedirs(run_out_dir, exist_ok=True)

        try:
            with file_handler(
                os.path.join(run_out_dir, f"{timestamp}_dipy_auto.log"),
                level=logger.getEffectiveLevel(),
            ):
                config = copy.deepcopy(base_config)
                io_config = _fill_io_section(
                    config=config,
                    overrides={
                        "dwi": run.dwi,
                        "bvals": run.bval,
                        "bvecs": run.bvec,
                        "t1w": run.t1w,
                        "dwi_opp_phase": run.fmap_epi,
                        "bids_folder": run.root,
                        "subject": run.subject,
                        "session": run.session or "",
                        "entities": run.label,
                        "pe_dir": metadata.get("PhaseEncodingDirection", ""),
                        "total_readout_time": metadata.get("TotalReadoutTime", ""),
                        "effective_echo_spacing": metadata.get(
                            "EffectiveEchoSpacing", ""
                        ),
                        "echo_time": metadata.get("EchoTime", ""),
                        "repetition_time": metadata.get("RepetitionTime", ""),
                        "atlas_tractogram": atlas_tractogram,
                        "bundle_atlas_dir": bundle_atlas_dir,
                    },
                    out_dir=run_out_dir,
                    out_report=out_report,
                )

                if not run.is_complete:
                    raise PipelineError(
                        f"{run.label}: no b-value or b-vector file found next to "
                        f"{run.dwi}"
                    )

                config_path = _write_pipeline_config(
                    config=config,
                    out_dir=run_out_dir,
                    timestamp=timestamp,
                    pipeline_name=pipeline_name,
                )

                _resolve_atlas_paths(config=config, io_config=io_config)

                missing, invalid = _validate_io_refs(config=config, io_config=io_config)
                if missing or invalid:
                    _log_io_validation_errors(
                        missing=missing, invalid=invalid, label=run.label
                    )
                    raise PipelineError(f"{run.label}: input validation failed")

                execution_info = execute_semantic_pipeline(
                    config=config,
                    io_config=io_config,
                    config_file_path=config_path,
                    report_path=os.path.join(
                        run_out_dir, f"{timestamp}_pipeline_report.html"
                    ),
                    start=start,
                    dry_run=dry_run,
                    force=self._force_overwrite,
                )

                record["status"] = "completed"
                record["report"] = (execution_info or {}).get("report")

                if not dry_run:
                    _write_bids_derivative_links(
                        run=run,
                        deriv_root=deriv_root,
                        execution_info=execution_info,
                    )
        except PipelineError as e:
            logger.error(str(e))
            record["error"] = str(e)
            record["failed_stage"] = e.stage
        except Exception as e:
            logger.error(f"{run.label} failed: {e}")
            logger.debug(traceback.format_exc())
            record["error"] = str(e)

        record["duration"] = time.perf_counter() - started
        return record
