"""Semantic Pipeline Execution System for DIPY Workflows.

This module implements a flexible pipeline system based on semantic naming
and automatic DAG-based wiring. The pipeline:
- Uses semantic stage names
- Automatically infers data flow from output→input name matching
- Executes stages in topological order based on dependencies
- Supports interactive mode for dynamic pipeline construction
- Supports flexible TOML configuration via [[pipeline]] sections
"""

from collections import defaultdict, deque
from datetime import datetime
import importlib
import inspect
import os
import re
import sys
import time

from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
from dipy.workflows import templates
from dipy.workflows.cli import cli_flows
from dipy.workflows.workflow import Workflow

try:  # standard module since Python 3.11
    import tomllib as toml
except ImportError:
    toml, have_toml, _ = optional_package("tomli")


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


def validate_pipeline_config(pipeline_stages):
    """Validate pipeline configuration for correct input/output references.

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
        except ValueError as e:
            errors.append(f"Stage '{stage_name}': {str(e)}")
        except Exception as e:
            errors.append(
                f"Stage '{stage_name}': Failed to introspect CLI "
                f"'{cli_name}': {str(e)}"
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
                if ref_output not in available_outputs:
                    errors.append(
                        f"Stage '{stage_name}': Unknown output '{ref_output}' "
                        f"from stage '{ref_stage}'. Available: "
                        f"{sorted(available_outputs)}"
                    )

    return len(errors) == 0, errors


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
    and identify output parameters (those starting with 'out_').

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
    ['out_pam', 'out_shm', 'out_peaks_dir', 'out_peaks_values']
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
        output_params = []

        # Identify output parameters
        # Convention: parameters starting with 'out_' are outputs
        # Exclude 'out_dir' as it's a directory parameter, not an output file
        for param_name in sig.parameters:
            if param_name.startswith("out_") and param_name != "out_dir":
                output_params.append(param_name)

        return output_params

    except Exception as e:
        logger.debug(f"Error inspecting {cli_command}: {e}")
        return []


def detect_output_conflicts(*, pipeline_stages):
    """Detect if multiple stages would produce conflicting output filenames.

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
    ...     {"name": "csa_fit", "cli": "dipy_fit_csa", ...},
    ...     {"name": "csd_fit", "cli": "dipy_fit_csd", ...}
    ... ]
    >>> conflicts = detect_output_conflicts(pipeline_stages=stages)
    >>> conflicts
    {'out_pam': ['csa_fit', 'csd_fit'],
     'out_shm': ['csa_fit', 'csd_fit']}
    """
    # Track which stages produce which output parameters
    # Map: output_param -> list of (stage_name, cli)
    output_producers = defaultdict(list)

    for stage in pipeline_stages:
        stage_name = stage.get("name")
        cli = stage.get("cli")

        if not cli:
            continue

        # Dynamically get output parameters for this CLI
        output_params = get_workflow_output_params(cli_command=cli)

        for output_param in output_params:
            # Skip if stage already has explicit output parameter
            # (user manually specified to avoid conflict)
            if output_param in stage:
                continue

            output_producers[output_param].append((stage_name, cli))

    # Find actual conflicts (multiple stages producing same output)
    conflicts = {}
    for output_param, producers in output_producers.items():
        if len(producers) > 1:
            # Extract just the stage names
            stage_names = [stage_name for stage_name, _ in producers]
            conflicts[output_param] = stage_names

    return conflicts


def resolve_output_conflicts(*, config, conflicts):
    """Resolve output filename conflicts by adding stage-specific names.

    Modifies the config in-place to add explicit output parameters with
    stage-specific filenames following the pattern:
    {base_name}_{stage_name}.{extension}

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
    'pam_csa_fit.nii.gz'
    >>> config["pipeline"][1]["out_pam"]
    'pam_csd_fit.nii.gz'
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

            # Create stage-specific filename
            # Pattern: {base_name}_{stage_name}.nii.gz
            # Example: pam_csa_fit.nii.gz, pam_csd_fit.nii.gz
            new_filename = f"{base_name}_{stage_name}.nii.gz"

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


def execute_pipeline_stage(*, stage_name, stage_config, resolved_outputs, io_config):
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

    Returns
    -------
    dict
        Mapping of output_param -> file_path for this stage.
    """
    cli_name = stage_config["cli"]

    if cli_name not in cli_flows:
        raise ValueError(f"Unknown CLI: {cli_name}")

    resolved_config = resolve_stage_parameters(
        stage_config, resolved_outputs, io_config
    )

    workflow_params = {}

    if "inputs" in resolved_config:
        workflow_params.update(resolved_config["inputs"])

    if "params" in resolved_config:
        workflow_params.update(resolved_config["params"])

    for key, value in resolved_config.items():
        if key not in ("name", "cli", "inputs", "params"):
            workflow_params[key] = value

    module_name, class_name = cli_flows[cli_name]
    module = importlib.import_module(module_name)
    workflow_class = getattr(module, class_name)

    sig = inspect.signature(workflow_class.run)
    valid_params = {
        name: param.default if param.default is not inspect.Parameter.empty else None
        for name, param in sig.parameters.items()
        if name != "self"
    }

    out_dir = io_config.get("out_dir", ".")
    if "out_dir" in valid_params:
        workflow_params["out_dir"] = out_dir

    invalid_params = set(workflow_params.keys()) - set(valid_params.keys())
    if invalid_params:
        raise ValueError(
            f"Invalid parameters for {cli_name}: {invalid_params}. "
            f"Valid: {list(valid_params.keys())}"
        )

    final_params = {**valid_params, **workflow_params}

    logger.info(f"Executing stage '{stage_name}' using {cli_name}...")
    t_start = time.perf_counter()

    try:
        workflow = workflow_class()
        workflow.run(**final_params)

        stage_outputs = {}
        if hasattr(workflow, "last_generated_outputs"):
            if isinstance(workflow.last_generated_outputs, dict):
                stage_outputs = workflow.last_generated_outputs
            elif isinstance(workflow.last_generated_outputs, list):
                outputs = introspect_workflow_outputs(cli_name)
                for out_param, out_file in zip(
                    sorted(outputs), workflow.last_generated_outputs
                ):
                    stage_outputs[out_param] = out_file

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
    import glob

    cli_name = stage_config.get("cli")
    if not cli_name or cli_name not in cli_flows:
        return {}

    outputs = introspect_workflow_outputs(cli_name)
    out_dir = io_config.get("out_dir", ".")
    discovered = {}

    for output_param in outputs:
        potential_patterns = []

        if "out_" in output_param:
            suffix = output_param.replace("out_", "")
            potential_patterns.extend(
                [
                    os.path.join(out_dir, f"{suffix}.nii.gz"),
                    os.path.join(out_dir, f"{suffix}.nii"),
                    os.path.join(out_dir, f"*{suffix}*.nii.gz"),
                    os.path.join(out_dir, f"*{suffix}*.nii"),
                    os.path.join(out_dir, f"{suffix}.trk"),
                    os.path.join(out_dir, f"{suffix}.tck"),
                    os.path.join(out_dir, f"{suffix}.txt"),
                    os.path.join(out_dir, f"{suffix}.pam5"),
                ]
            )

        cli_workflow_patterns = {
            "dipy_denoise_nlmeans": ["dwi_nlmeans.nii.gz"],
            "dipy_denoise_mppca": ["dwi_mppca.nii.gz"],
            "dipy_denoise_patch2self": ["dwi_patch2self.nii.gz"],
            "dipy_median_otsu": ["*_masked.nii.gz", "*_mask.nii.gz"],
            "dipy_mask": ["mask.nii.gz"],
            "dipy_gibbs_ringing": ["*_unring.nii.gz"],
            "dipy_correct_motion": ["*_moved.nii.gz"],
            "dipy_extract_b0": ["*_b0.nii.gz"],
        }

        if cli_name in cli_workflow_patterns:
            for pattern in cli_workflow_patterns[cli_name]:
                potential_patterns.append(os.path.join(out_dir, pattern))

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

    return discovered


def execute_semantic_pipeline(
    *, config, io_config, config_file_path=None, start=None, dry_run=False
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
    start : str, optional
        Stage name to start execution from. Earlier stages will be skipped,
        but their outputs must exist on disk.
    dry_run : bool
        If True, only show execution plan.
    """
    pipeline_stages = config.get("pipeline", [])
    if not pipeline_stages:
        raise ValueError("No [[pipeline]] sections found in configuration")

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
        sys.exit(1)

    logger.info("✓ Pipeline configuration validated successfully")

    dependencies = build_dependency_graph(pipeline_stages)

    try:
        execution_order = topological_sort(pipeline_stages, dependencies)
    except ValueError as e:
        logger.error(f"Failed to compute execution order: {e}")
        sys.exit(1)

    dag_viz = visualize_pipeline_dag(pipeline_stages, dependencies, execution_order)
    logger.info(dag_viz)
    logger.info(f"Execution order: {' → '.join(execution_order)}")

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

    if dry_run:
        logger.info("[Dry run mode] Pipeline plan shown above - no execution performed")
        return

    logger.info("=" * 60)
    logger.info("Starting Diffusion Pipeline Execution")
    logger.info("=" * 60)

    resolved_outputs = {}
    stage_map = {stage["name"]: stage for stage in pipeline_stages}
    stages_info = []
    pipeline_start_time = time.perf_counter()

    if start:
        if start not in execution_order:
            logger.error(f"Start stage '{start}' not found in pipeline")
            logger.error(f"Available stages: {', '.join(execution_order)}")
            sys.exit(1)

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

        execution_order = execution_order[start_idx:]
        logger.info(f"Executing stages: {' → '.join(execution_order)}")
        logger.info("")

    for stage_name in execution_order:
        stage_config = stage_map[stage_name]
        stage_start = time.perf_counter()

        try:
            outputs = execute_pipeline_stage(
                stage_name=stage_name,
                stage_config=stage_config,
                resolved_outputs=resolved_outputs,
                io_config=io_config,
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

            if outputs:
                logger.debug(f"Stage '{stage_name}' outputs: {outputs}")

        except Exception as e:
            logger.error(f"Pipeline failed at stage '{stage_name}': {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            sys.exit(1)

    total_time = time.perf_counter() - pipeline_start_time

    logger.info("=" * 60)
    logger.info("Pipeline Completed Successfully!")
    logger.info("=" * 60)

    from dipy.workflows import report

    execution_info = {
        "stages": stages_info,
        "total_time": total_time,
        "dag_visualization": dag_viz,
        "execution_order": execution_order,
    }

    report_path = os.path.join(io_config.get("out_dir", "."), "pipeline_report.html")

    try:
        report_path = report.generate_html_report(
            config, config_file_path or "config.toml", execution_info, report_path
        )
        logger.info(f"HTML report generated: {report_path}")
        logger.info("")
        logger.info("=" * 70)
        logger.info("To view the report with interactive 3D viewers:")
        logger.info(f"  cd {io_config.get('out_dir', '.')}")
        logger.info("  python -m http.server 8000")
        logger.info("  Then open: http://localhost:8000/pipeline_report.html")
        logger.info("=" * 70)
    except Exception as e:
        logger.warning(f"Failed to generate HTML report: {e}")
        import traceback

        logger.debug(traceback.format_exc())


# =============================================================================
# AutoFlow Workflow
# =============================================================================


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
        config_file=None,
        dwi_file=None,
        bvals_file=None,
        bvecs_file=None,
        t1_file=None,
        bids_folder=None,
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
        config_file : str, optional
            Path to the TOML configuration file with [[pipeline]] sections.
        dwi_file : str, optional
            Path to the diffusion weighted images.
        bvals_file : str, optional
            Path to the bvals file.
        bvecs_file : str, optional
            Path to the bvec file.
        t1_file : str, optional
            Path to the T1 file.
        bids_folder : str, optional
            Path to the BIDS folder (reserved for future use).
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
            Output directory.
        out_report : str, optional
            Path to the report file.
        """

        if list_pipelines:
            current_log_level = logger.getEffectiveLevel()
            logger.info(
                templates.list_pipelines_with_descriptions(log_level=current_log_level)
            )
            return

        out_dir = os.path.abspath(out_dir) or os.getcwd()

        # =================================================================
        # Configuration Loading
        # =================================================================

        if config_file is None:
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
                        "Interactive mode requires --bvals_file to analyze "
                        "data characteristics."
                    )
                    sys.exit(1)

                try:
                    from dipy.workflows import interactive

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
                    else:
                        pipeline_type = selected_pipeline
                except Exception as e:
                    logger.error(f"Error in interactive mode: {e}")
                    import traceback

                    traceback.print_exc()
                    sys.exit(1)

            if "config" not in locals():
                if pipeline_type:
                    try:
                        logger.info(f"Loading predefined pipeline: {pipeline_type}")
                        config_str = templates.get_predefined_pipeline(
                            pipeline_name=pipeline_type
                        )
                        config = toml.loads(config_str)
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
                        config_str = templates.get_predefined_pipeline(
                            pipeline_name="full"
                        )
                        config = toml.loads(config_str)
                    except Exception as e:
                        logger.error(f"Error loading default pipeline: {e}")
                        sys.exit(1)

            config.setdefault("io", {})
            config["io"]["dwi"] = dwi_file or config.get("io", {}).get("dwi", "")
            config["io"]["bvals"] = bvals_file or config.get("io", {}).get("bvals", "")
            config["io"]["bvecs"] = bvecs_file or config.get("io", {}).get("bvecs", "")
            config["io"]["t1w"] = t1_file or config.get("io", {}).get("t1w", "")
            config["io"]["bids_folder"] = bids_folder or config.get("io", {}).get(
                "bids_folder", ""
            )
            config["io"]["atlas_tractogram"] = atlas_tractogram or config.get(
                "io", {}
            ).get("atlas_tractogram", "")
            config["io"]["bundle_atlas_dir"] = bundle_atlas_dir or config.get(
                "io", {}
            ).get("bundle_atlas_dir", "")
            config["io"]["out_dir"] = out_dir
            config["io"]["out_report"] = os.path.join(out_dir, out_report)

            os.makedirs(out_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pipeline_name = pipeline_type or "default"
            config_filename = f"{pipeline_name}_pipeline_{timestamp}.toml"
            config_file = os.path.join(out_dir, config_filename)

            formatted_toml = format_toml_config(config)
            with open(config_file, "w") as f:
                f.write(formatted_toml)
            logger.info(f"Configuration saved to: {config_file}")

        # =================================================================
        # Load Configuration File
        # =================================================================

        config_file = os.path.abspath(config_file)
        cfg_name = os.path.basename(config_file)
        logger.info(f"Running pipeline from {cfg_name}")

        if not config_file.endswith(".toml"):
            logger.error(f"Configuration file must be TOML: {config_file}")
            sys.exit(1)

        try:
            with open(config_file, "rb") as f:
                config = toml.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_file}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error decoding TOML file: {e}")
            sys.exit(1)

        # =================================================================
        # Interactive Mode Support
        # =================================================================

        if interactive_mode:
            logger.info("Interactive mode enabled")

            bvals_for_analysis = config.get("io", {}).get("bvals", bvals_file)

            if bvals_for_analysis and os.path.exists(bvals_for_analysis):
                try:
                    from dipy.workflows import interactive

                    data_chars = interactive.analyze_bvals(
                        bvals_file=bvals_for_analysis
                    )

                    logger.info("\nYou can customize the pipeline interactively.")
                    logger.info("Current configuration will be used as the base.")

                    # Save updated config
                    formatted_toml = format_toml_config(config)
                    with open(config_file, "w") as f:
                        f.write(formatted_toml)
                    logger.info(f"Updated configuration saved to: {config_file}")

                except Exception as e:
                    logger.warning(f"Interactive mode customization failed: {e}")

        # =================================================================
        # Validate Configuration
        # =================================================================

        if "pipeline" not in config or not isinstance(config["pipeline"], list):
            logger.error("Configuration missing [[pipeline]] sections")
            logger.error("Expected TOML structure with [[pipeline]] array sections")
            sys.exit(1)

        io_config = config.get("io", {})
        if out_dir:
            io_config["out_dir"] = out_dir
        io_config.setdefault("out_dir", ".")
        io_config["out_dir"] = os.path.abspath(io_config["out_dir"])
        os.makedirs(io_config["out_dir"], exist_ok=True)

        cli_overrides = {
            "dwi": dwi_file,
            "bvals": bvals_file,
            "bvecs": bvecs_file,
            "t1w": t1_file,
            "bids_folder": bids_folder,
            "atlas_tractogram": atlas_tractogram,
            "bundle_atlas_dir": bundle_atlas_dir,
        }

        for config_key, cli_value in cli_overrides.items():
            if cli_value:
                existing = io_config.get(config_key, "")
                if existing and existing != cli_value:
                    logger.warning(f"Overriding {config_key}: {existing} → {cli_value}")
                io_config[config_key] = cli_value

        # =================================================================
        # Auto-download Atlas if needed for SLR or RecoBundles
        # =================================================================

        # Check if dipy_slr or dipy_recobundles are in the pipeline
        needs_atlas = False
        for stage in config.get("pipeline", []):
            cli_name = stage.get("cli", "")
            if cli_name in ("dipy_slr", "dipy_recobundles"):
                needs_atlas = True
                break

        # If atlas is needed and paths are empty, download the atlas
        if needs_atlas:
            atlas_tractogram_path = io_config.get("atlas_tractogram", "")
            bundle_atlas_dir_path = io_config.get("bundle_atlas_dir", "")

            if not atlas_tractogram_path or not bundle_atlas_dir_path:
                logger.info("Atlas required for SLR/RecoBundles but not provided.")
                logger.info("Downloading HCP 30-bundle atlas...")

                try:
                    from dipy.data import fetch_30_bundle_atlas_hcp842

                    # fetch_30_bundle_atlas_hcp842 returns (files_dict, base_path)
                    _, atlas_base = fetch_30_bundle_atlas_hcp842()

                    # Set the atlas paths
                    if not atlas_tractogram_path:
                        atlas_tractogram_path = os.path.join(
                            atlas_base,
                            "Atlas_30_Bundles",
                            "whole_brain",
                            "whole_brain_MNI.trk",
                        )
                        io_config["atlas_tractogram"] = atlas_tractogram_path
                        logger.info(f"Atlas tractogram: {atlas_tractogram_path}")

                    if not bundle_atlas_dir_path:
                        bundle_atlas_dir_path = os.path.join(
                            atlas_base, "Atlas_30_Bundles", "bundles"
                        )
                        io_config["bundle_atlas_dir"] = bundle_atlas_dir_path
                        logger.info(f"Bundle atlas directory: {bundle_atlas_dir_path}")

                except Exception as e:
                    logger.error(f"Failed to download atlas: {e}")
                    logger.error(
                        "Please provide atlas paths manually using "
                        "--atlas_tractogram and --bundle_atlas_dir"
                    )
                    sys.exit(1)

        # =================================================================
        # Validate Input Files BEFORE Pipeline Execution
        # =================================================================

        required_io_refs = set()

        for stage in config.get("pipeline", []):
            for key, value in stage.items():
                if key in ("name", "cli"):
                    continue
                if isinstance(value, str):
                    refs = extract_variable_references(value)
                    for ref in refs:
                        if ref.startswith("io."):
                            io_key = ref.split(".", 1)[1]
                            if not io_key.startswith("out_"):
                                required_io_refs.add((ref, io_key))

        missing_or_empty = []
        invalid_paths = []

        for ref, io_key in required_io_refs:
            path = io_config.get(io_key, "")

            if not path or path.strip() == "":
                missing_or_empty.append((ref, io_key))
            elif not os.path.exists(path):
                invalid_paths.append((ref, path))

        if missing_or_empty or invalid_paths:
            logger.error("=" * 70)
            logger.error("Input Validation Failed")
            logger.error("=" * 70)

            if missing_or_empty:
                logger.error("✗ Missing input file definitions in [io] section:")
                for ref, io_key in missing_or_empty:
                    logger.error(f"  • {io_key} (required by ${{{ref}}})")

                logger.error("Please provide input files using one of these methods:")
                logger.error(
                    "  1. Edit the config file and set paths in the [io] section"
                )
                logger.error("  2. Use CLI arguments:")
                if any(k == "dwi" for _, k in missing_or_empty):
                    logger.error("     dipy_auto --dwi-file <path>")
                if any(k == "bvals" for _, k in missing_or_empty):
                    logger.error("     dipy_auto --bvals-file <path>")
                if any(k == "bvecs" for _, k in missing_or_empty):
                    logger.error("     dipy_auto --bvecs-file <path>")

            if invalid_paths:
                logger.error("✗ Input files do not exist:")
                for ref, path in invalid_paths:
                    logger.error(f"  • {path} (from ${{{ref}}})")

            logger.error("=" * 70)
            sys.exit(1)

        logger.info("✓ All input files validated successfully")

        # =================================================================
        # Execute Semantic Pipeline
        # =================================================================

        logger.info("Pipeline Configuration Summary:")
        logger.info(f"  Config file: {cfg_name}")
        logger.info(f"  Output directory: {io_config['out_dir']}")
        logger.info(f"  Number of stages: {len(config['pipeline'])}")
        logger.info(f"  Dry run: {dry_run}")
        logger.info("")

        execute_semantic_pipeline(
            config=config,
            io_config=io_config,
            config_file_path=config_file,
            start=start,
            dry_run=dry_run,
        )
