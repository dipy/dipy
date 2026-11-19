"""Native dialogs of the Skyline launcher."""

import ctypes
import html
import os
import shutil
import subprocess
import sys
import time

from dipy.desktop.constants import APP_NAME, SKYLINE_HOME
from dipy.utils.logging import logger

_LINUX_DIALOGS = (
    ("zenity", ("--error", "--no-markup", "--title", APP_NAME, "--text")),
    ("kdialog", ("--title", APP_NAME, "--error")),
    ("notify-send", (APP_NAME,)),
)
_LINUX_CONFIRM_DIALOGS = (
    (
        "zenity",
        (
            "--question",
            "--no-markup",
            "--default-cancel",
            "--title",
            APP_NAME,
            "--text",
        ),
    ),
    ("kdialog", ("--title", APP_NAME, "--yesno")),
)
CONFIRM_LOCK = SKYLINE_HOME / "confirm.lock"
CONFIRM_LOCK_SECONDS = 120


def applescript_string(text):
    """Quote text as an AppleScript string literal.

    Parameters
    ----------
    text : str
        Text to quote.

    Returns
    -------
    str
        Double-quoted literal with backslashes and double quotes escaped.
    """
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def linux_dialog_text(name, message):
    """Prepare a message for the Linux dialog tool ``name``.

    Parameters
    ----------
    name : str
        Dialog tool: ``zenity``, ``kdialog`` or ``notify-send``.
    message : str
        Text to show.

    Returns
    -------
    str
        The message itself for ``zenity``, which is run with ``--no-markup``;
        the message with its markup escaped for ``notify-send``; rich text
        with the message escaped for ``kdialog``, which may read plain text as
        markup and expands backslash sequences.
    """
    if name == "zenity":
        return message
    escaped = html.escape(message, quote=False)
    if name == "notify-send":
        return escaped
    body = escaped.replace("\\", "\\\\").replace("\n", "<br>")
    return f"<qt>{body}</qt>"


def acquire_confirmation_slot(*, lock=CONFIRM_LOCK):
    """Claim the right to show a confirmation dialog, so that links cannot flood.

    Parameters
    ----------
    lock : Path, optional
        Lock file; the caller removes it once the dialog is closed.

    Returns
    -------
    bool
        False when another confirmation started less than
        ``CONFIRM_LOCK_SECONDS`` ago and has not finished; True otherwise, with
        the lock file written.
    """
    try:
        age = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        age = None
    if age is not None:
        if age < CONFIRM_LOCK_SECONDS:
            return False
        lock.unlink(missing_ok=True)
    try:
        with open(lock, "x", encoding="utf-8") as lock_file:
            lock_file.write(str(os.getpid()))
    except FileExistsError:
        return False
    return True


def report_error(message):
    """Log an error and show it in a native dialog.

    Parameters
    ----------
    message : str
        Text of the error.
    """
    logger.error(message)
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10)
    elif sys.platform == "darwin":
        script = (
            f"display alert {applescript_string(APP_NAME)} "
            f"message {applescript_string(message)}"
        )
        subprocess.run(["osascript", "-e", script], check=False)
    else:
        for name, arguments in _LINUX_DIALOGS:
            tool = shutil.which(name)
            if tool is not None:
                text = linux_dialog_text(name, message)
                subprocess.run([tool, *arguments, text], check=False)
                break


def confirm_open(path):
    """Ask the user to confirm that a link may open a file.

    Parameters
    ----------
    path : Path
        File that the link asks Skyline to open.

    Returns
    -------
    bool
        True when the user agrees to open the file.

    Raises
    ------
    RuntimeError
        On Linux, if neither ``zenity`` nor ``kdialog`` is installed.
    """
    resolved = path.resolve()
    shown = str(path)
    if os.path.normcase(resolved) != os.path.normcase(path):
        shown += f"\n(a link to {resolved})"
    message = f"A link asks {APP_NAME} to open:\n\n{shown}\n\nOpen this file?"
    if sys.platform == "win32":
        # MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON2: No is the default; 6 is IDYES.
        return ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x124) == 6
    if sys.platform == "darwin":
        script = (
            f"display dialog {applescript_string(message)} "
            f"with title {applescript_string(APP_NAME)} "
            'buttons {"Cancel", "Open"} default button "Cancel" '
            'cancel button "Cancel" with icon caution'
        )
        return subprocess.run(["osascript", "-e", script], check=False).returncode == 0
    for name, arguments in _LINUX_CONFIRM_DIALOGS:
        tool = shutil.which(name)
        if tool is not None:
            text = linux_dialog_text(name, message)
            return subprocess.run([tool, *arguments, text], check=False).returncode == 0
    raise RuntimeError(
        "Opening files from links needs zenity or kdialog to ask for "
        f"confirmation; open {path} from the file manager instead."
    )
