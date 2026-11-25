"""Skyline as a desktop application.

This module opens files in Skyline from ``dipy-skyline://`` links, the
application icon or "Open with". A link asks for confirmation, naming every file,
before Skyline opens it. Links to network paths (UNC paths, ``file:`` URLs with
a remote host) and to files of other formats are refused. The remaining
exposure is the file readers (NIfTI, TRK, VTK, ...) parsing a local file that
someone else placed on the disk, for instance in Downloads; the confirmation
dialog, which names every file and defaults to cancel, guards against it.

Launches that start together, such as one for each file selected in a file
manager, merge into one. When Skyline windows are already open, a dialog asks
whether the files open in the one used last or in a new window.
``dipy_skyline`` typed in a terminal does not go through this module, so it
always opens a new window.

This module must stay light: it never imports numpy, nibabel, FURY or the
report modules, so that it can run, and explain itself, before the viewer
packages are installed.
"""

from pathlib import Path
import sys

from dipy import __version__ as dipy_version
from dipy.desktop.constants import APP_LOG_FILE, MAX_FILES, SKYLINE_HOME
from dipy.desktop.dialogs import (
    CONFIRM_LOCK,
    acquire_confirmation_slot,
    allow_foreground_window,
    confirm_open,
    report_error,
    use_last_window,
)
from dipy.desktop.environment import missing_viz_requirements
from dipy.desktop.ipc import collect_batch, live_windows, send_open
from dipy.desktop.links import (
    checked_file,
    is_skyline_link,
    skyline_argument,
    skyline_targets,
)
from dipy.utils.logging import CustomHandler, file_handler, logger
from dipy.workflows import cli


def parse_arguments(argv):
    """Check what the launcher was asked to open.

    Parameters
    ----------
    argv : list of str
        Command-line arguments: ``dipy-skyline://open`` link, ``file:`` URLs
        or file paths.

    Returns
    -------
    targets : list of Path
        Files to open, without duplicates. Empty when ``argv`` holds a link
        that is refused.
    errors : list of str
        One message for each argument or rule that failed.
    from_link : bool
        True when an argument is a ``dipy-skyline://`` link, which must be
        confirmed.
    """
    from_link = any(is_skyline_link(argument) for argument in argv)
    if from_link and len(argv) != 1:
        return [], ["A dipy-skyline link must come alone."], True
    if len(argv) > MAX_FILES:
        return [], [f"Skyline opens at most {MAX_FILES} files at once."], from_link

    targets = []
    errors = []
    for argument in argv:
        try:
            targets.extend(skyline_targets(argument))
        except ValueError as err:
            errors.append(str(err))
    return list(dict.fromkeys(targets)), errors, from_link


def _restore_drive(destination, drive_root, share_root):
    """Name a file on a mapped drive by its drive letter again.

    Parameters
    ----------
    destination : Path
        Resolved path of the file.
    drive_root : Path
        Root of the drive the file was named on, such as ``Z:\\``.
    share_root : Path
        Resolved ``drive_root``, which is a network path for a mapped drive.

    Returns
    -------
    Path
        ``destination`` under ``drive_root`` when it lies in the share that
        ``drive_root`` maps; ``destination`` unchanged otherwise, so that a
        link leading to another share stays a network path.
    """
    if share_root != drive_root and destination.is_relative_to(share_root):
        return drive_root / destination.relative_to(share_root)
    return destination


def _resolve(target):
    """Follow the symbolic links of a path, keeping its mapped drive letter.

    Parameters
    ----------
    target : Path
        Absolute path of a file.

    Returns
    -------
    Path
        Where the path leads.
    """
    destination = target.resolve()
    if sys.platform == "win32" and target.drive and not target.drive.startswith("\\\\"):
        drive_root = Path(f"{target.drive}\\")
        destination = _restore_drive(destination, drive_root, drive_root.resolve())
    return destination


def resolve_targets(targets):
    """Follow the symbolic links of the files to open and check where they lead.

    The files are resolved once, before any confirmation, and the resolved
    paths are the ones opened, so that a link swapped while a dialog is open
    changes nothing.

    Parameters
    ----------
    targets : list of Path
        Files that passed ``checked_file`` under their own names.

    Returns
    -------
    shown : list of Path
        The targets whose destination is acceptable.
    resolved : list of Path
        Destination of each of ``shown``, in the same order.
    errors : list of str
        One message for each target whose destination fails ``checked_file``
        or cannot be read, for instance a link to a network path or to a file
        of another format.

    Notes
    -----
    A file on a mapped network drive keeps its drive letter. Resolving a
    symbolic link that leads to a share contacts that share before the link is
    refused, so a link planted on the disk can still make Windows reach out to
    a server.
    """
    shown = []
    resolved = []
    errors = []
    for target in targets:
        try:
            destination = _resolve(target)
            checked_file(str(destination))
        except ValueError as err:
            errors.append(str(err))
            continue
        except OSError as err:
            errors.append(f"Skyline cannot open {str(target)!r}: {err}")
            continue
        shown.append(target)
        resolved.append(destination)
    return shown, resolved, errors


def _wants_last_window(batch):
    """Ask whether files open in the window used last, when a window is open.

    Parameters
    ----------
    batch : sequence of str
        Paths to open.

    Returns
    -------
    bool
        False when the user asks for a new window. True otherwise, also when
        another dialog of the launcher is open, so that launches never pile
        up questions.
    """
    if not acquire_confirmation_slot():
        return True
    try:
        return use_last_window([Path(path).name for path in batch])
    finally:
        CONFIRM_LOCK.unlink(missing_ok=True)


def main(argv):
    """Open Skyline, on files when given, reporting problems in a dialog.

    Parameters
    ----------
    argv : list of str
        Command-line arguments: nothing, any number of file paths or ``file:``
        URLs (up to ``MAX_FILES``), or one ``dipy-skyline://open`` link.

    Returns
    -------
    int
        Exit code: 0 on success or when the user cancels a link, 1 when Skyline
        cannot run, 2 for bad arguments.
    """
    # The DIPY logger binds ``sys.stdout`` when it is imported, which is None
    # without a console (``pythonw``): log to the file and drop that handler.
    if sys.stdout is None or sys.stderr is None:
        SKYLINE_HOME.mkdir(parents=True, exist_ok=True)
        log = open(APP_LOG_FILE, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
        sys.stdout = log if sys.stdout is None else sys.stdout
        sys.stderr = log if sys.stderr is None else sys.stderr
        for handler in list(logger.handlers):
            if isinstance(handler, CustomHandler) and handler.stream is None:
                logger.removeHandler(handler)
    SKYLINE_HOME.mkdir(parents=True, exist_ok=True)
    with file_handler(APP_LOG_FILE):
        logger.info(f"DIPY {dipy_version} Skyline launcher: {sys.executable}")

        targets, errors, from_link = parse_arguments(argv)
        shown, resolved, resolve_errors = resolve_targets(targets)
        errors += resolve_errors
        if errors:
            if from_link and not acquire_confirmation_slot():
                logger.info("A dialog is already open; ignored this link.")
                return 2
            try:
                report_error("\n".join(errors))
            finally:
                if from_link:
                    CONFIRM_LOCK.unlink(missing_ok=True)
            if from_link or not resolved:
                return 2

        if from_link:
            if not acquire_confirmation_slot():
                logger.info("A confirmation is already open; ignored this link.")
                return 0
            try:
                confirmed = confirm_open(shown, resolved)
            except RuntimeError as err:
                report_error(str(err))
                return 1
            finally:
                CONFIRM_LOCK.unlink(missing_ok=True)
            if not confirmed:
                logger.info(f"Opening {len(shown)} file(s) was cancelled.")
                return 0

        targets = list(dict.fromkeys(resolved))

        missing = missing_viz_requirements()
        if missing:
            report_error(
                f"Skyline needs packages that are not installed in {sys.executable}: "
                f'{", ".join(missing)}. Run "dipy_skyline --setup" in a terminal '
                "of this environment."
            )
            return 1

        if targets:
            batch = collect_batch([str(target) for target in targets])
            if batch is None:
                logger.info(
                    f"Handed {len(targets)} file(s) to the Skyline launch "
                    "already collecting files."
                )
                return 0
            windows = live_windows()
            if windows and _wants_last_window(batch):
                window = max(windows, key=lambda window: window.last_active)
                allow_foreground_window(window.pid)
                if send_open(window, batch):
                    return 0
                logger.warning(
                    "The window used last did not take the files; opening a new window."
                )
            targets = batch

        arguments = [skyline_argument(path=target) for target in targets]
        sys.argv = ["dipy_skyline", *(arguments or ["run"])]
        try:
            cli.run()
        except Exception:  # noqa: BLE001
            logger.exception("Skyline stopped with an error")
            report_error(f"Skyline stopped with an error; details in {APP_LOG_FILE}.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
