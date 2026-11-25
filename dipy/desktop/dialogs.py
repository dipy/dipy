"""Native dialogs of the Skyline launcher."""

import ctypes
import html
import os
import shutil
import subprocess
import sys
import time

from dipy.desktop.constants import APP_NAME, SKYLINE_HOME, SUMMARY_FILES
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
_LINUX_WINDOW_DIALOGS = (
    (
        "zenity",
        (
            "--question",
            "--no-markup",
            "--title",
            APP_NAME,
            "--ok-label",
            "Last window",
            "--cancel-label",
            "New window",
            "--text",
        ),
    ),
    (
        "kdialog",
        (
            "--title",
            APP_NAME,
            "--yes-label",
            "Last window",
            "--no-label",
            "New window",
            "--yesno",
        ),
    ),
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


def display_text(text):
    """Make text safe to show in a dialog.

    Parameters
    ----------
    text : str
        Text that may come from a link, such as a file name.

    Returns
    -------
    str
        ``text`` with every character that is not printable, apart from the
        space, replaced by its ``\\uXXXX`` escape (``\\UXXXXXXXX`` beyond the
        Basic Multilingual Plane). Control, format, separator and unassigned
        characters can neither fake dialog lines nor reorder or hide the text.
    """
    escaped = []
    for char in text:
        if char == " " or char.isprintable():
            escaped.append(char)
        elif ord(char) <= 0xFFFF:
            escaped.append(f"\\u{ord(char):04x}")
        else:
            escaped.append(f"\\U{ord(char):08x}")
    return "".join(escaped)


def files_summary(names):
    """Summarize the files a window displays, for its title.

    Parameters
    ----------
    names : sequence of str
        File names, in the order displayed.

    Returns
    -------
    str
        The first ``SUMMARY_FILES`` names, made safe by ``display_text`` and
        joined by commas, then ``(+N)`` for the N others. Empty without names.
    """
    summary = ", ".join(display_text(name) for name in names[:SUMMARY_FILES])
    if len(names) > SUMMARY_FILES:
        summary += f" (+{len(names) - SUMMARY_FILES})"
    return summary


def window_title(names):
    """Build the title of a Skyline window.

    Parameters
    ----------
    names : sequence of str
        Names of the files the window displays.

    Returns
    -------
    str
        The application name, followed by ``files_summary(names)`` when the
        window displays files.
    """
    summary = files_summary(names)
    return f"{APP_NAME} - {summary}" if summary else APP_NAME


def confirm_message(paths, resolved):
    """Build the text that asks to confirm the files a link opens.

    Parameters
    ----------
    paths : sequence of Path
        Files that the link asks Skyline to open, at most ``MAX_LINK_FILES``.
    resolved : sequence of Path
        Where each of ``paths`` leads once symbolic links are followed, in the
        same order.

    Returns
    -------
    str
        Question naming every path, and the target of each path that is a
        link to another file.
    """
    entries = []
    for path, target in zip(paths, resolved):
        entry = display_text(str(path))
        if os.path.normcase(target) != os.path.normcase(path):
            entry += f"\n(a link to {display_text(str(target))})"
        entries.append(entry)
    count = len(paths)
    subject = "this file" if count == 1 else f"these {count} files"
    listing = "\n".join(entries)
    return f"A link asks {APP_NAME} to open:\n\n{listing}\n\nOpen {subject}?"


def confirm_open(paths, resolved):
    """Ask the user to confirm that a link may open files.

    Parameters
    ----------
    paths : sequence of Path
        Files that the link asks Skyline to open.
    resolved : sequence of Path
        Where each of ``paths`` leads once symbolic links are followed, in the
        same order. These are the files opened if the user agrees.

    Returns
    -------
    bool
        True when the user agrees to open the files.

    Raises
    ------
    RuntimeError
        On Linux, if neither ``zenity`` nor ``kdialog`` is installed.
    """
    message = confirm_message(paths, resolved)
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
        "confirmation; open the files from the file manager instead."
    )


def window_choice_message(names):
    """Build the text that asks where to open files when a window is open.

    Parameters
    ----------
    names : sequence of str
        File names to open.

    Returns
    -------
    str
        Question naming the first files, made safe by ``display_text``.
    """
    return (
        f"{APP_NAME} is already open.\n\n"
        f"Open {files_summary(names)} in the window used last?"
    )


def use_last_window(names):
    """Ask whether files open in the window used last or in a new window.

    Parameters
    ----------
    names : sequence of str
        File names to open, shown in the question.

    Returns
    -------
    bool
        True to open the files in the window used last. False to open a new
        window, which is also the answer when the user closes the dialog.
        True without asking when Linux has neither ``zenity`` nor ``kdialog``.
    """
    message = window_choice_message(names)
    if sys.platform == "win32":
        message += "\n\nYes: the window used last\nNo: a new window"
        # MB_YESNO | MB_ICONQUESTION | MB_SETFOREGROUND; 6 is IDYES.
        return ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10024) == 6
    if sys.platform == "darwin":
        script = (
            f"display dialog {applescript_string(message)} "
            f"with title {applescript_string(APP_NAME)} "
            'buttons {"New window", "Last window"} default button "Last window"'
        )
        result = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, check=False
        )
        return result.returncode == 0 and "Last window" in result.stdout
    for name, arguments in _LINUX_WINDOW_DIALOGS:
        tool = shutil.which(name)
        if tool is not None:
            text = linux_dialog_text(name, message)
            return subprocess.run([tool, *arguments, text], check=False).returncode == 0
    return True


def allow_foreground_window(pid):
    """Let a process bring its window to the front, on Windows.

    Windows only lets the process that has the foreground hand it over.

    Parameters
    ----------
    pid : int
        Process id of the window that will be asked to come forward. Nothing
        happens on other systems.
    """
    if sys.platform == "win32":
        ctypes.windll.user32.AllowSetForegroundWindow(pid)
