"""DICOM-to-NIfTI conversion using the dcm2niix executable."""

import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from dipy.utils.deprecator import warning_for_keywords
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package

dcm2niix, have_dcm2niix, _ = optional_package("dcm2niix", min_version="1.0.20250506")

# dcm2niix exit status when some inputs were converted and others failed.
_SOME_CONVERTED = 8


def _dcm2niix_executable():
    """Return the packaged dcm2niix executable, else one on PATH, else None."""
    return dcm2niix.bin if have_dcm2niix else shutil.which("dcm2niix")


@warning_for_keywords()
def convert_dicom(
    input_path,
    *,
    out_dir,
    compress=True,
    filename="%3s_%p",
    bids=True,
    anonymize=True,
):
    """Convert DICOM images to NIfTI using dcm2niix.

    Parameters
    ----------
    input_path : str or Path
        DICOM directory, searched recursively, or a single DICOM file.
    out_dir : str or Path
        Output directory. It is created if it does not exist.
    compress : bool, optional
        Write ``.nii.gz`` instead of ``.nii`` files.
    filename : str, optional
        dcm2niix filename template, without directory or extension. The
        default is the zero-padded series number followed by the protocol name.
    bids : bool, optional
        Write BIDS JSON sidecars.
    anonymize : bool, optional
        Remove identifying fields from the BIDS JSON sidecars.

    Returns
    -------
    outputs : dict
        Sorted lists of the created files, keyed by ``nifti``, ``json``,
        ``bval`` and ``bvec``.

    Notes
    -----
    The executable of the ``dcm2niix`` Python package is used when installed,
    otherwise ``dcm2niix`` from PATH.

    Existing files are never overwritten: if any output name is taken, all
    outputs of the call receive the same numeric suffix. If only some series
    can be converted, the converted ones are kept and a warning is logged.

    Anonymization only applies to the JSON sidecars; filenames and images can
    still identify a participant. Avoid the ``%f``, ``%i``, ``%n`` and ``%t``
    template fields, which add folder names, patient details or dates.

    """
    input_path = Path(input_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"DICOM input does not exist: {input_path}")
    if not input_path.is_file() and not input_path.is_dir():
        raise ValueError(f"DICOM input must be a file or directory: {input_path}")
    if (
        not isinstance(filename, str)
        or filename in ("", ".", "..")
        or any(character in filename for character in ("/", "\\", "\0"))
    ):
        raise ValueError("filename must be a nonempty basename template, not a path")
    executable = _dcm2niix_executable()
    if executable is None:
        raise ImportError(
            "DICOM conversion requires dcm2niix. Install it with "
            "'pip install dcm2niix' or put a dcm2niix executable on PATH."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "-g",
        "i",
        "-e",
        "n",
        "-z",
        "i" if compress else "n",
        "-b",
        "y" if bids else "n",
        "-ba",
        "y" if anonymize else "n",
        "-f",
        filename,
        "-w",
        "2",
        "-d",
        "9",
        "-s",
        "y" if input_path.is_file() else "n",
    ]
    outputs = {"nifti": [], "json": [], "bval": [], "bvec": []}
    extensions = (".nii.gz", ".nii", ".json", ".bval", ".bvec")
    with TemporaryDirectory(prefix=".dipy-dcm2niix-", dir=out_dir) as staging:
        result = subprocess.run(
            [executable, *args, "-o", staging, str(input_path)],
            capture_output=True,
            text=True,
            errors="replace",
        )
        stdout = result.stdout.replace(staging, str(out_dir))
        stderr = result.stderr.replace(staging, str(out_dir))

        groups = {}
        for path in Path(staging).iterdir():
            suffix = next(
                (ext for ext in extensions if path.name.endswith(ext)), path.suffix
            )
            groups.setdefault(path.name.removesuffix(suffix), []).append((path, suffix))
        suffixes = {suffix for files in groups.values() for _, suffix in files}
        has_nifti = bool(suffixes & {".nii", ".nii.gz"})
        if result.returncode not in (0, _SOME_CONVERTED) or not has_nifti:
            raise RuntimeError(
                f"dcm2niix did not convert {input_path} "
                f"(exit status {result.returncode}).\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )

        # A single suffix for all outputs keeps related images' names aligned.
        suffixes.update(extensions)
        tag, index = "", 0
        while any(
            os.path.lexists(out_dir / f"{stem}{tag}{suffix}")
            for stem in groups
            for suffix in suffixes
        ):
            index += 1
            tag = f"_{index}"
        for stem, files in groups.items():
            for path, suffix in files:
                target = out_dir / f"{stem}{tag}{suffix}"
                path.rename(target)
                if suffix in (".nii", ".nii.gz"):
                    outputs["nifti"].append(target)
                elif suffix in (".json", ".bval", ".bvec"):
                    outputs[suffix[1:]].append(target)

    for paths in outputs.values():
        paths.sort()
    if result.returncode == _SOME_CONVERTED:
        logger.warning(
            f"dcm2niix converted only part of {input_path}; review its output:\n"
            f"{stdout}"
        )
    elif stdout:
        logger.info(f"dcm2niix stdout:\n{stdout}")
    if stderr:
        logger.warning(f"dcm2niix stderr:\n{stderr}")
    return outputs
