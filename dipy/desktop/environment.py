"""Requirements and interpreter of the Skyline desktop application."""

import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from dipy.desktop.constants import (
    LAUNCH_BOOTSTRAP,
    MAX_FILES,
    PYTHON_SAFE_PATH,
    SKYLINE_PACKAGES,
)

_SKYLINE_PACKAGE_NAMES = frozenset(map(canonicalize_name, SKYLINE_PACKAGES))
_DETACHED_PROCESS = (
    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    if os.name == "nt"
    else {"start_new_session": True}
)


def missing_viz_requirements():
    """Requirements of the ``viz`` extra that Skyline imports and are missing.

    Only the installed package metadata is read; no viewer package is imported.

    Returns
    -------
    list of str
        Requirement specifiers such as ``"fury>=2.1.0"``. Empty when the extra
        is satisfied, or when DIPY has no installed metadata to read.

    Notes
    -----
    ``optional_package`` is not used here: it imports each package, which
    would load numpy and the viewer stack in a module that must stay light, and
    it returns no requirement specifier to hand to the installer. The metadata
    holds both, so the ``viz`` extra of ``pyproject.toml`` stays the only list.
    """
    try:
        specifiers = importlib.metadata.requires("dipy") or []
    except importlib.metadata.PackageNotFoundError:
        return []

    missing = []
    for specifier in specifiers:
        requirement = Requirement(specifier)
        marker = requirement.marker
        if (
            canonicalize_name(requirement.name) not in _SKYLINE_PACKAGE_NAMES
            or marker is None
            or not marker.evaluate({"extra": "viz"})
            or marker.evaluate({"extra": ""})
        ):
            continue
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is None or not requirement.specifier.contains(
            installed, prereleases=True
        ):
            missing.append(f"{requirement.name}{requirement.specifier}")
    return missing


def launcher_command():
    """Command prefix that opens a file in Skyline with this interpreter.

    Returns
    -------
    list of str
        Command to which the files to open are appended.
    """
    return [sys.executable, PYTHON_SAFE_PATH, "-c", LAUNCH_BOOTSTRAP]


def open_in_skyline(paths, *, launcher=None):
    """Open files in a new Skyline process, without waiting for it.

    Parameters
    ----------
    paths : sequence of str or Path
        Files to display, from 1 to ``MAX_FILES``.
    launcher : list of str, optional
        Command to which the absolute paths are appended. By default,
        ``launcher_command()``.

    Raises
    ------
    ValueError
        If the number of paths is outside 1 to ``MAX_FILES``.
    OSError
        If the process cannot be started.
    """
    if not 1 <= len(paths) <= MAX_FILES:
        raise ValueError(f"paths must name 1 to {MAX_FILES} files, got {len(paths)}")
    launcher = launcher_command() if launcher is None else launcher
    subprocess.Popen(
        [*launcher, *(str(Path(path).absolute()) for path in paths)],
        **_DETACHED_PROCESS,
    )
