"""Skyline as a desktop application.

This module opens files in Skyline from ``dipy-skyline://`` links, the
application icon or "Open with". A link asks for confirmation, naming the file,
before Skyline opens it. Links to network paths (UNC paths, ``file:`` URLs with
a remote host) and to files of other formats are refused. The remaining
exposure is the file readers (NIfTI, TRK, VTK, ...) parsing a local file that
someone else placed on the disk, for instance in Downloads; the confirmation
dialog, which names the file and defaults to cancel, guards against it.

This module must stay light: it never imports numpy, nibabel, FURY or the
report modules, so that it can run, and explain itself, before the viewer
packages are installed.
"""

import sys
import urllib.parse

from dipy import __version__ as dipy_version
from dipy.desktop.constants import (
    APP_LOG_FILE,
    SKYLINE_HOME,
    SKYLINE_URL_SCHEME,
)
from dipy.desktop.dialogs import (
    CONFIRM_LOCK,
    acquire_confirmation_slot,
    confirm_open,
    report_error,
)
from dipy.desktop.environment import missing_viz_requirements
from dipy.desktop.links import skyline_argument, skyline_target
from dipy.utils.logging import CustomHandler, file_handler, logger
from dipy.workflows import cli


def main(argv):
    """Open Skyline, on a file when one is given, reporting problems in a dialog.

    Parameters
    ----------
    argv : list of str
        Command-line arguments: nothing, or one ``dipy-skyline://open`` link,
        ``file:`` URL or file path.

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

        if len(argv) > 1:
            report_error("Skyline opens one file at a time.")
            return 2

        target = None
        if argv:
            try:
                target = skyline_target(argv[0])
            except ValueError as err:
                report_error(str(err))
                return 2

            if urllib.parse.urlsplit(argv[0]).scheme == SKYLINE_URL_SCHEME:
                if not acquire_confirmation_slot():
                    logger.info("A confirmation is already open; ignored this link.")
                    return 0
                try:
                    confirmed = confirm_open(target)
                except RuntimeError as err:
                    report_error(str(err))
                    return 1
                finally:
                    CONFIRM_LOCK.unlink(missing_ok=True)
                if not confirmed:
                    logger.info(f"Opening {target} was cancelled.")
                    return 0

        missing = missing_viz_requirements()
        if missing:
            report_error(
                f"Skyline needs packages that are not installed in {sys.executable}: "
                f'{", ".join(missing)}. Run "dipy_skyline --setup" in a terminal '
                "of this environment."
            )
            return 1

        sys.argv = [
            "dipy_skyline",
            "run" if target is None else skyline_argument(path=target),
        ]
        try:
            cli.run()
        except Exception:  # noqa: BLE001
            logger.exception("Skyline stopped with an error")
            report_error(f"Skyline stopped with an error; details in {APP_LOG_FILE}.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
