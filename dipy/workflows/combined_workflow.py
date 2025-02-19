import importlib
import inspect
import logging
import os
import re
import sys
import time

from dipy.utils.optpkg import optional_package
from dipy.workflows.cli import cli_flows
from dipy.workflows.workflow import Workflow

try:  # standard module since Python 3.11
    import tomllib as toml
except ImportError:
    # available for older Python via pip
    toml, have_toml, _ = optional_package("tomli")

tomli_w, have_tomli_w, _ = optional_package("tomli_w")


def flatten_dict(d, *, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, parent_key=new_key, sep=sep).items())
        elif isinstance(v, list):
            for _, item in enumerate(v):
                items.extend(flatten_dict(item, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep="."):
    """Reconstruct a nested dictionary from a flattened dictionary, restoring lists."""
    unflat = {}

    for key, value in d.items():
        keys = key.split(sep)
        d_ref = unflat

        for i, k in enumerate(keys):
            is_last = i == len(keys) - 1
            is_next_key_index = i < len(keys) - 1 and keys[i + 1].isdigit()

            # Convert numeric keys to integers for list handling
            k = int(k) if k.isdigit() else k

            if is_last:
                if isinstance(d_ref, list) and isinstance(k, int):
                    while len(d_ref) <= k:
                        d_ref.append(None)
                    d_ref[k] = value
                else:
                    d_ref[k] = value
            else:
                if isinstance(d_ref, list):
                    while len(d_ref) <= k:
                        d_ref.append({})
                    d_ref = d_ref[k]
                else:
                    if k not in d_ref:
                        d_ref[k] = [] if is_next_key_index else {}
                    d_ref = d_ref[k]

    def convert_lists(d):
        """Convert indexed dicts into proper lists."""
        if isinstance(d, dict):
            if all(isinstance(k, int) for k in d.keys()):
                return [d[k] for k in sorted(d.keys())]
            return {k: convert_lists(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_lists(v) for v in d]
        return d

    return convert_lists(unflat)


def resolve_variables(config):
    """Replace ${var} placeholders with their values from the dictionary.
    Ensure out_* keys are prefixed with their respective out_dir.
    """
    resolved = config.copy()
    current_dir = os.getcwd()  # Get current working directory

    def replace_var(value):
        """Replace placeholders in values recursively."""
        if isinstance(value, str):
            matches = re.findall(r"\${([^}]+)}", value)
            for match in matches:
                if match in resolved:
                    replacement = resolved[match]
                    replacement, _ = replace_var(
                        replacement
                    )  # Resolve nested references
                    value = value.replace(f"${{{match}}}", str(replacement))
                else:
                    error = f"Reference '{match}' not found."
                    return value, error
        return value, []

    # First pass: Resolve all ${} placeholders
    all_errors = []
    for key in resolved:
        resolved[key], errors = replace_var(resolved[key])
        if errors:
            all_errors.append(errors)

    # Second pass: Ensure all out_* keys are prefixed with their respective out_dir
    for key in resolved:
        if key.endswith(".out_dir"):
            # Make sure the out_dir is set properly (default to current directory)
            resolved[key] = resolved[key].strip() or current_dir

    for key in resolved:
        if ".out_" in key:
            step_prefix = ".".join(key.split(".")[:-1])  # Get step prefix
            out_dir_key = f"{step_prefix}.out_dir"  # Find corresponding out_dir
            out_dir = resolved.get(out_dir_key, current_dir)  # Default to current dir
            resolved[key] = os.path.join(out_dir, resolved[key])

    return resolved, all_errors


def add_dipy_auto_config(config):
    return config


class AutoFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "auto"

    def run(self,
            config_file=None,
            dwi_file=None,
            bvals_file=None,
            bvecs_file=None,
            t1_file=None,
            bids_folder=None,
            out_dir="",
            out_report="reports.toml"):
        """Run diffusion pipelines based on configuration files.

        Parameters
        ----------
        config_file : string
            Path to the configuration files.
        dwi_file : string, optional
            Path to the diffusion weighted images.
        bvals_file : string, optional
            Path to the bvals files. This path may contain wildcards to process
            multiple inputs at once.
        bvecs_file : string, optional
            Path to the bvec files. This path may contain wildcards to process
            multiple inputs at once.
        t1_file : string, optional
            Path to the T1 files. This path may contain wildcards to process
            multiple inputs at once.
        bids_folder : string, optional
            Path to the BIDS folder.
        out_dir : string, optional
            Output directory.
        out_report : string, optional
            Path to the directory where reports will be saved.
        """
        out_dir = os.path.abspath(out_dir) or os.getcwd()

        if config_file is None:
            logging.info("No configuration files provided. "
                         "Selecting default configuration.")
            try:
                config = toml.loads(DEFAULT_AUTO_CONFIG)
            except Exception as e:
                logging.error(f"Error decoding default TOML file: {e}")
                sys.exit(1)

            config.setdefault("General", {})
            config.setdefault("io", {})
            config["io"]["dwi"] = dwi_file or ""
            config["io"]["bvals"] = bvals_file or ""
            config["io"]["bvecs"] = bvecs_file or ""
            config["io"]["t1w"] = t1_file or ""
            config["io"]["bids_folder"] = bids_folder or ""
            config["io"]["out_dir"] = out_dir
            config["io"]["out_report"] = os.path.join(out_dir, out_report)

            config_file = os.path.join(out_dir, "dipy_auto.toml")
            if not have_tomli_w:
                logging.error("tomli_w is not installed. "
                              "Please install it to write the configuration file.\n"
                              "python -m pip install tomli_w")
                sys.exit(1)
            with open(config_file, "wb") as f:
                tomli_w.dump(config, f)

        config_file = os.path.abspath(config_file)
        cfg_name = os.path.basename(config_file)
        logging.info(f"Running pipeline on {cfg_name}")

        if not config_file.endswith(".toml"):
            logging.error(f"Configuration file is not a toml file: {config_file}")
            sys.exit(1)

        try:
            with open(config_file, "rb") as f:
                config = toml.load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {config_file}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error decoding TOML file: {e}")
            sys.exit(1)

        # Checks io section in the configuration file
        logging.info(f"Checking io in configuration file: {cfg_name}")
        if "io" not in config:
            logging.error(f"Missing 'io' section in configuration file: {cfg_name}")
            sys.exit(1)

        required_keys = {
            "dwi",
            "bvals",
            "bvecs",
            "t1w",
            "bids_folder",
        }

        # Update the required keys values based on cli
        for config_key, cli_key in zip(required_keys, [dwi_file, bvals_file, bvecs_file,
                                                       t1_file, bids_folder]):
            if cli_key:
                fpath = config.get("io", {}).get(config_key, "")
                if fpath:
                    logging.warning(
                        f"Overriding {config_key} file in configuration file: {fpath} with {cli_key}"
                    )
                config["io"]["config_key"] = cli_key

        # Check if all input paths are empty
        if all(not x for k, x in config.get("io", {}).items()
               if k in required_keys):
            logging.error(f"All input paths are empty in {config_file}.")
            sys.exit(1)

        # Check if io section have the required keys
        required_keys.add("out_dir")

        if not required_keys.issubset(config["io"].keys()):
            logging.error(f"Missing required keys in 'io' section: {required_keys}")
            sys.exit(1)

        # check if defined paths exist
        unknown_paths = []
        for k, ipath in config.get("io", {}).items():
            if k.startswith("out_"):
                continue
            if ipath and not os.path.exists(ipath):
                unknown_paths.append(ipath)
        if unknown_paths:
            logging.error(f"One or more file paths not found: {unknown_paths}")
            sys.exit(1)

        all_steps_keys = [key for key in config.keys() if key.startswith("step_")]
        if len(all_steps_keys) == 0:
            logging.info("Starting auto mode")
            add_dipy_auto_config(config)

        logging.info(f"Scanning configuration file: {config_file}")
        cli_errors = []
        # option to initialize the cli
        init_cli_keys = ["force", "skip", "output_strategy", "mix_names"]
        for k_step in all_steps_keys:
            cli_step = {}
            step = config.pop(k_step)
            config[k_step] = []
            for cli_name, cli_args in step.items():
                if cli_name not in cli_flows:
                    cli_errors.append(f"{cli_name} CLI not found in DIPY.")
                    continue

                # add try except block to handle bad import
                module = importlib.import_module(f"{cli_flows[cli_name][0]}")
                wflw = getattr(module, cli_flows[cli_name][1])
                sig = inspect.signature(wflw.run)
                sig_dict = {
                    name: param.default
                    if param.default is not inspect.Parameter.empty
                    else None
                    for name, param in sig.parameters.items()
                    if name != "self"
                }
                init_cli_dict = {
                    key: cli_args.pop(key)
                    for key in init_cli_keys if key in cli_args
                }
                # Check if all keys in cli_args are valid from the config file
                if not set(cli_args.keys()).issubset(sig_dict):
                    wrong_keys = set(cli_args.keys()) - set(sig_dict)
                    cli_errors.append(
                        f"Invalid keys in {cli_name} CLI: {wrong_keys}"
                    )
                    continue
                cli_step[cli_name] = {**sig_dict, **cli_args, "init": init_cli_dict}

            config[k_step].append(cli_step)

        flatten_config = flatten_dict(config)
        flatten_config, var_errors = resolve_variables(flatten_config)

        all_errors = cli_errors + var_errors
        if all_errors:
            logging.error(f"Errors found in configuration file {config_file}")
            for error in all_errors:
                logging.error(error)
            sys.exit(1)

        config = unflatten_dict(flatten_config)
        logging.info(f"Starting Pipeline from {config_file}")

        for k_step in all_steps_keys:
            msg = f"Running {k_step}"
            sep = "=" * len(msg)
            logging.info(f"\n{sep}\n{msg}\n{sep}")
            for cli_name, cli_args in config[k_step].items():
                t_start = time.perf_counter()
                try:
                    init_args = cli_args.pop("init", {})
                    module = importlib.import_module(f"{cli_flows[cli_name][0]}")
                    wflw = getattr(module, cli_flows[cli_name][1])
                    wflw(**init_args).run(**cli_args)
                except Exception as e:
                    logging.error(e)
                    logging.error(f"Error when running {cli_name}")
                    sys.exit(1)
                logging.info(f"Finished {cli_name} in {time.perf_counter() - t_start:.2f} seconds")


DEFAULT_AUTO_CONFIG = """
[General]
name =  "dipy_auto"
flow_name =  "dipy_auto"
summary = "Dipy is the parrot of the python scientific community"
description = ""
version = "0.1.0"
author = "Dipy Developers"

[io]
dwi = ""
bvals = ""
bvecs = ""
t1w = ""
bids_folder = ""
out_dir = "."

[step_0.dipy_denoise_nlmeans]
input_files = "${io.dwi}"


[step_0.dipy_median_otsu]
input_files = "${step_0.dipy_denoise_nlmeans.out_denoised}"
save_masked = true
vol_idx = "0, 1"

[step_1.dipy_mask]
input_files = "${step_0.dipy_denoise_nlmeans.out_denoised}"
lb = 15
force = "True"

[step_1.dipy_info]
input_files = "${step_1.dipy_mask.out_mask}"
"""
