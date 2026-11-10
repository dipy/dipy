"""Skyline as a desktop application.

This module opens files in Skyline from ``dipy-skyline://`` links, the
application icon or "Open with", and registers that application with the
operating system. A link asks for confirmation, naming the file, before
Skyline opens it. It must stay light: it never imports numpy, nibabel, FURY or
the report modules, so that it can run, and explain itself, before the viewer
packages are installed.
"""

import ctypes
import glob
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path, PureWindowsPath
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from dipy import __version__ as dipy_version
from dipy.utils.assets import asset_home, fetch_asset
from dipy.utils.logging import CustomHandler, file_handler, logger
from dipy.workflows import cli

if sys.platform == "win32":
    import winreg

APP_NAME = "DIPY SKYLINE"
SKYLINE_URL_SCHEME = "dipy-skyline"
ASSET_APP = "dipy-skyline"
SKYLINE_HOME = asset_home(ASSET_APP)
APP_LOG_FILE = SKYLINE_HOME / "skyline_app.log"
SKYLINE_APP_ICONS = {"win32": "icons/skyline.ico", "darwin": "icons/skyline.icns"}
SKYLINE_APP_ICON_DEFAULT = "icons/skyline.png"

FILE_PROGID = "DIPY.Skyline.File"
BUNDLE_ID = "org.dipy.skyline"
MODULE_NAME = "dipy.workflows.skyline_app"
NO_REGISTER_ENV_VAR = "DIPY_SKYLINE_NO_REGISTER"
SKYLINE_PACKAGES = ("fury", "imgui_bundle")
_SKYLINE_PACKAGE_NAMES = frozenset(map(canonicalize_name, SKYLINE_PACKAGES))

# Launched from a menu, a link or a report, nothing activated the environment:
# its folders are missing from PATH, and an editable install cannot find the
# ``ninja`` it needs to import DIPY. The code holds no double quote or percent
# sign, so it survives the registry, shortcut and desktop entry quoting.
LAUNCH_BOOTSTRAP = (
    "import os,sys,runpy;"
    "d=os.path.dirname(sys.executable);"
    "os.environ['PATH']=os.pathsep.join([d,os.path.join(d,'Scripts'),"
    "os.path.join(d,'Library','bin'),os.environ.get('PATH','')]);"
    f"runpy.run_module('{MODULE_NAME}',run_name='__main__')"
)

# Formats the Skyline viewer can load; anything else opens in the browser.
SKYLINE_FORMATS = (
    ".nii",
    ".nii.gz",
    ".trx",
    ".trk",
    ".tck",
    ".pam5",
    ".vtk",
    ".vtp",
    ".fib",
    ".dpy",
    ".nrrd",
    ".mgh",
    ".mgz",
    ".gii",
    ".gii.gz",
    ".pial",
)

_WINDOWS_CLASSES = r"Software\Classes"
_CSIDL_PROGRAMS = 0x0002
_CSIDL_DESKTOPDIRECTORY = 0x0010
_SHCNE_ASSOCCHANGED = 0x08000000
_SHORTCUT_SCRIPT = (
    "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:DIPY_LNK);"
    "$s.TargetPath = $env:DIPY_TARGET;"
    "$s.Arguments = $env:DIPY_ARGS;"
    "$s.IconLocation = $env:DIPY_ICON;"
    "$s.WorkingDirectory = $env:USERPROFILE;"
    f"$s.Description = '{APP_NAME} viewer';"
    "$s.Save()"
)

_LSREGISTER = (
    "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
    "LaunchServices.framework/Support/lsregister"
)

_LINUX_DIALOGS = (
    ("zenity", ("--error", "--title", APP_NAME, "--text")),
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
_LINUX_ICON_NAME = "dipy-skyline"


def skyline_argument(*, path):
    """Command-line argument that makes Skyline load exactly ``path``.

    Skyline expands its input files as glob patterns, so the characters
    ``*``, ``?`` and ``[`` of a real file name are escaped.

    Parameters
    ----------
    path : str
        File to display.

    Returns
    -------
    str
        Glob-escaped path.
    """
    return glob.escape(str(path))


def skyline_url(*, path):
    """Link that makes the registered Skyline application open ``path``.

    Parameters
    ----------
    path : str or Path
        File to display.

    Returns
    -------
    str
        ``dipy-skyline://open?path=...`` with the absolute path percent-encoded.
    """
    query = urllib.parse.urlencode(
        {"path": str(Path(path).absolute())}, quote_via=urllib.parse.quote
    )
    return f"{SKYLINE_URL_SCHEME}://open?{query}"


def _cannot_open(argument, reason):
    """Build the error raised when Skyline is asked to open something invalid.

    Parameters
    ----------
    argument : str
        What Skyline was asked to open.
    reason : str
        Why it cannot be opened.

    Returns
    -------
    ValueError
        Error naming the argument and the reason.
    """
    return ValueError(f"Skyline cannot open {argument!r}: {reason}")


def _is_network_path(location):
    """Tell whether a path points to a network share, which is never opened.

    Parameters
    ----------
    location : str
        File path as received from a link or a ``file:`` URL.

    Returns
    -------
    bool
        True on Windows for a UNC path (``\\\\server\\share``), False elsewhere.
    """
    return sys.platform == "win32" and PureWindowsPath(location).drive.startswith(
        "\\\\"
    )


def skyline_target(argument):
    """File Skyline is asked to open by a link, a ``file:`` URL or a path.

    Parameters
    ----------
    argument : str
        ``dipy-skyline://open?path=...`` link, ``file:`` URL or file path.

    Returns
    -------
    Path
        Absolute path of an existing file of a format Skyline can display.

    Raises
    ------
    ValueError
        If the argument is not one of the accepted forms, or does not name an
        existing file whose name ends with one of ``SKYLINE_FORMATS``.
    """
    parts = urllib.parse.urlsplit(argument)
    if parts.scheme == SKYLINE_URL_SCHEME:
        if parts.netloc != "open" or parts.path not in ("", "/"):
            raise _cannot_open(argument, f"not a {SKYLINE_URL_SCHEME}://open link")
        query = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
        values = query.get("path", [])
        if (
            list(query) != ["path"]
            or len(values) != 1
            or not Path(values[0]).is_absolute()
        ):
            raise _cannot_open(
                argument, "the link must carry exactly one absolute path"
            )
        if _is_network_path(values[0]):
            raise _cannot_open(argument, "network paths are not allowed")
        path = Path(values[0])
    elif parts.scheme == "file":
        if parts.netloc not in ("", "localhost"):
            raise _cannot_open(argument, "network paths are not allowed")
        location = urllib.request.url2pathname(parts.path)
        if _is_network_path(location):
            raise _cannot_open(argument, "network paths are not allowed")
        path = Path(location)
    elif "://" in argument:
        raise _cannot_open(argument, f"not a {SKYLINE_URL_SCHEME}://open link")
    else:
        path = Path(argument).absolute()

    if not path.is_file():
        raise _cannot_open(argument, "no such file")
    if not path.name.endswith(SKYLINE_FORMATS):
        raise _cannot_open(argument, "unsupported file type")
    return path


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
        Command to which the file to open is appended.
    """
    return [sys.executable, "-c", LAUNCH_BOOTSTRAP]


def _applescript_string(text):
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


def _report_error(message):
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
            f"display alert {_applescript_string(APP_NAME)} "
            f"message {_applescript_string(message)}"
        )
        subprocess.run(["osascript", "-e", script], check=False)
    else:
        for name, arguments in _LINUX_DIALOGS:
            tool = shutil.which(name)
            if tool is not None:
                subprocess.run([tool, *arguments, message], check=False)
                break


def _confirm_open(path):
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
    message = f"A link asks {APP_NAME} to open:\n\n{path}\n\nOpen this file?"
    if sys.platform == "win32":
        # MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON2: No is the default; 6 is IDYES.
        return ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x124) == 6
    if sys.platform == "darwin":
        script = (
            f"display dialog {_applescript_string(message)} "
            f"with title {_applescript_string(APP_NAME)} "
            'buttons {"Cancel", "Open"} default button "Cancel" '
            'cancel button "Cancel" with icon caution'
        )
        return subprocess.run(["osascript", "-e", script], check=False).returncode == 0
    for name, arguments in _LINUX_CONFIRM_DIALOGS:
        tool = shutil.which(name)
        if tool is not None:
            return (
                subprocess.run([tool, *arguments, message], check=False).returncode == 0
            )
    raise RuntimeError(
        "Opening files from links needs zenity or kdialog to ask for "
        f"confirmation; open {path} from the file manager instead."
    )


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
            _report_error("Skyline opens one file at a time.")
            return 2

        target = None
        if argv:
            try:
                target = skyline_target(argv[0])
            except ValueError as err:
                _report_error(str(err))
                return 2

            if urllib.parse.urlsplit(argv[0]).scheme == SKYLINE_URL_SCHEME:
                try:
                    confirmed = _confirm_open(target)
                except RuntimeError as err:
                    _report_error(str(err))
                    return 1
                if not confirmed:
                    logger.info(f"Opening {target} was cancelled.")
                    return 0

        missing = missing_viz_requirements()
        if missing:
            _report_error(
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
            _report_error(f"Skyline stopped with an error; details in {APP_LOG_FILE}.")
            return 1
    return 0


def _app_command():
    """Command that starts the Skyline application with this interpreter.

    Returns
    -------
    list of str
        Interpreter (``pythonw`` on Windows, to avoid a console window), then
        ``-c`` and the bootstrap that prepares ``PATH`` before importing DIPY.
    """
    python = Path(sys.executable)
    if sys.platform == "win32" and python.with_name("pythonw.exe").exists():
        python = python.with_name("pythonw.exe")
    return [str(python), "-c", LAUNCH_BOOTSTRAP]


def _extensions():
    """Last suffix of each uncompressed Skyline format, as operating systems see it.

    Returns
    -------
    list of str
        Sorted extensions without a leading dot.

    Notes
    -----
    Compressed formats are left out because Windows and macOS associate by the
    last suffix only, so registering ``gz`` would claim every ``.gz`` file;
    ``.nii.gz`` and ``.gii.gz`` get no "Open with" entry there. Linux registers
    the full globs.
    """
    return sorted(
        {fmt.rsplit(".", 1)[-1] for fmt in SKYLINE_FORMATS if not fmt.endswith(".gz")}
    )


def _run_tool(name, *arguments, step):
    """Run an external tool, warning instead of failing when it cannot run.

    Parameters
    ----------
    name : str
        Executable name or path.
    *arguments : str
        Arguments of the tool.
    step : str
        What the tool does, for the warning.
    """
    tool = shutil.which(name)
    if tool is None:
        logger.warning(f"{name} not found; skipped {step}.")
        return
    result = subprocess.run([tool, *arguments], check=False)
    if result.returncode != 0:
        logger.warning(f"{name} exited with code {result.returncode} while {step}.")


def _windows_set_value(*, key_path, value, name="", value_type=None):
    """Set a value of a key below ``HKEY_CURRENT_USER``, creating the key.

    Parameters
    ----------
    key_path : str
        Path of the key below ``HKEY_CURRENT_USER``.
    value : str or bytes
        Value to store.
    name : str, optional
        Name of the value; the default value of the key when empty.
    value_type : int, optional
        ``winreg`` value type; ``REG_SZ`` when None.
    """
    value_type = winreg.REG_SZ if value_type is None else value_type
    with winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE
    ) as key:
        winreg.SetValueEx(key, name, 0, value_type, value)


def _windows_delete_tree(key_path):
    """Delete a key below ``HKEY_CURRENT_USER`` with all its subkeys.

    Parameters
    ----------
    key_path : str
        Path of the key below ``HKEY_CURRENT_USER``; nothing happens if absent.
    """
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_ALL_ACCESS
        )
    except FileNotFoundError:
        return
    with key:
        while True:
            try:
                child = winreg.EnumKey(key, 0)
            except OSError:
                break
            _windows_delete_tree(f"{key_path}\\{child}")
    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)


def _windows_shortcuts():
    """Locate the shortcut files of the application.

    Returns
    -------
    list of Path
        Shortcut in the Start menu programs folder, then on the Desktop.
    """
    shortcuts = []
    for csidl in (_CSIDL_PROGRAMS, _CSIDL_DESKTOPDIRECTORY):
        buffer = ctypes.create_unicode_buffer(260)
        ctypes.windll.shell32.SHGetFolderPathW(None, csidl, None, 0, buffer)
        shortcuts.append(Path(buffer.value) / f"{APP_NAME}.lnk")
    return shortcuts


def _windows_notify_shell():
    """Tell the Windows shell that file associations changed."""
    ctypes.windll.shell32.SHChangeNotify(_SHCNE_ASSOCCHANGED, 0, None, None)


def _windows_register(*, icon):
    """Register Skyline in the Windows registry and create its shortcuts.

    Parameters
    ----------
    icon : Path
        Application icon file.

    Returns
    -------
    list of str
        Registry keys and shortcut files created.
    """
    command = f'{subprocess.list2cmdline(_app_command())} "%1"'
    icon_value = f'"{icon}",0'

    scheme_key = rf"{_WINDOWS_CLASSES}\{SKYLINE_URL_SCHEME}"
    _windows_set_value(key_path=scheme_key, value=f"URL:{APP_NAME}")
    _windows_set_value(key_path=scheme_key, name="URL Protocol", value="")
    _windows_set_value(key_path=rf"{scheme_key}\DefaultIcon", value=icon_value)
    _windows_set_value(key_path=rf"{scheme_key}\shell\open\command", value=command)

    progid_key = rf"{_WINDOWS_CLASSES}\{FILE_PROGID}"
    _windows_set_value(key_path=progid_key, value=f"{APP_NAME} data")
    _windows_set_value(key_path=rf"{progid_key}\DefaultIcon", value=icon_value)
    application_key = rf"{progid_key}\Application"
    _windows_set_value(key_path=application_key, name="ApplicationName", value=APP_NAME)
    _windows_set_value(
        key_path=application_key, name="ApplicationIcon", value=icon_value
    )
    _windows_set_value(
        key_path=rf"{progid_key}\shell\open", name="FriendlyAppName", value=APP_NAME
    )
    _windows_set_value(key_path=rf"{progid_key}\shell\open\command", value=command)

    for extension in _extensions():
        _windows_set_value(
            key_path=rf"{_WINDOWS_CLASSES}\.{extension}\OpenWithProgids",
            name=FILE_PROGID,
            value=b"",
            value_type=winreg.REG_NONE,
        )

    locations = [
        rf"HKEY_CURRENT_USER\{scheme_key}",
        rf"HKEY_CURRENT_USER\{progid_key}",
    ]
    python, *arguments = _app_command()
    for link in _windows_shortcuts():
        environment = {
            **os.environ,
            "DIPY_LNK": str(link),
            "DIPY_TARGET": python,
            "DIPY_ARGS": subprocess.list2cmdline(arguments),
            "DIPY_ICON": f"{icon},0",
        }
        try:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.unlink(missing_ok=True)
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    _SHORTCUT_SCRIPT,
                ],
                env=environment,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as err:
            logger.warning(f"Could not create the shortcut {link}: {err}")
        else:
            locations.append(str(link))
    _windows_notify_shell()
    return locations


def _windows_unregister():
    """Remove the Windows registry keys and shortcuts of Skyline."""
    _windows_delete_tree(rf"{_WINDOWS_CLASSES}\{SKYLINE_URL_SCHEME}")
    _windows_delete_tree(rf"{_WINDOWS_CLASSES}\{FILE_PROGID}")
    for extension in _extensions():
        key_path = rf"{_WINDOWS_CLASSES}\.{extension}\OpenWithProgids"
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE
            ) as key:
                winreg.DeleteValue(key, FILE_PROGID)
        except FileNotFoundError:
            pass
    for link in _windows_shortcuts():
        link.unlink(missing_ok=True)
    _windows_notify_shell()


def _windows_registered_interpreter():
    """Read the interpreter of the registered ``dipy-skyline://`` handler.

    Returns
    -------
    str or None
        First token of the registered command, or None when it is not registered.
    """
    key_path = rf"{_WINDOWS_CLASSES}\{SKYLINE_URL_SCHEME}\shell\open\command"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            command = winreg.QueryValueEx(key, "")[0]
    except OSError:
        return None
    command = command.strip()
    if command.startswith('"'):
        return command[1:].split('"', 1)[0]
    return command.split(" ", 1)[0]


def _linux_data_dir():
    """Locate the XDG data folder of the user.

    Returns
    -------
    Path
        ``$XDG_DATA_HOME``, or ``~/.local/share`` when it is not set.
    """
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def _linux_desktop_dir():
    """Locate the Desktop folder of the user.

    Returns
    -------
    Path
        Folder reported by ``xdg-user-dir``, or ``~/Desktop``.
    """
    tool = shutil.which("xdg-user-dir")
    if tool is not None:
        result = subprocess.run(
            [tool, "DESKTOP"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())
    return Path.home() / "Desktop"


def _linux_files():
    """Locate the MIME package and desktop entry files of Skyline.

    Returns
    -------
    tuple of Path
        MIME package file and desktop entry file.
    """
    data = _linux_data_dir()
    return (
        data / "mime" / "packages" / "dipy-skyline.xml",
        data / "applications" / "dipy-skyline.desktop",
    )


def _linux_theme_icon():
    """Locate the application icon in the user's icon theme.

    Returns
    -------
    Path
        ``hicolor`` 256x256 icon file.
    """
    return (
        _linux_data_dir()
        / "icons"
        / "hicolor"
        / "256x256"
        / "apps"
        / f"{_LINUX_ICON_NAME}.png"
    )


def _linux_refresh_databases():
    """Refresh the MIME and desktop databases of the user."""
    data = _linux_data_dir()
    _run_tool(
        "update-mime-database", str(data / "mime"), step="updating the MIME database"
    )
    _run_tool(
        "update-desktop-database",
        str(data / "applications"),
        step="updating the desktop database",
    )


def _linux_mime_search_dirs():
    """Locate the folders whose ``mime/globs2`` database is searched.

    Returns
    -------
    list of Path
        The XDG data folder of the user, then ``XDG_DATA_DIRS``.
    """
    system_dirs = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    return [_linux_data_dir(), *map(Path, system_dirs.split(":"))]


def _linux_existing_mime_types(*, data_dirs):
    """Find the MIME types that other applications already give Skyline formats.

    Parameters
    ----------
    data_dirs : list of Path
        Folders that may hold a ``mime/globs2`` database.

    Returns
    -------
    dict
        For each entry of ``SKYLINE_FORMATS`` that has a glob in a database, the
        sorted list of its MIME types. Types made by a previous Skyline
        registration are left out.
    """
    formats_by_glob = {f"*{fmt}": fmt for fmt in SKYLINE_FORMATS}
    found = {}
    for data_dir in data_dirs:
        try:
            text = (data_dir / "mime" / "globs2").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        for line in text.splitlines():
            fields = line.split(":")
            if line.startswith("#") or len(fields) < 3:
                continue
            mime_type, pattern = fields[1], fields[2]
            fmt = formats_by_glob.get(pattern.lower())
            if fmt is not None and not mime_type.startswith("application/x-dipy-"):
                found.setdefault(fmt, set()).add(mime_type)
    return {fmt: sorted(types) for fmt, types in found.items()}


def _desktop_entry_quote(argument):
    """Quote an argument of the ``Exec`` line of a desktop entry.

    Parameters
    ----------
    argument : str
        Command-line argument to quote.

    Returns
    -------
    str
        Double-quoted argument. Reserved characters are backslash-escaped, every
        backslash is doubled for the string escaping that the specification
        applies first, and ``%`` is doubled so it is not read as a field code.
    """
    escaped = re.sub(r'(["`$\\])', r"\\\1", argument)
    return '"' + escaped.replace("\\", "\\\\").replace("%", "%%") + '"'


def _desktop_entry_unquote(text):
    """Undo :func:`_desktop_entry_quote` on the text between the quotes.

    Parameters
    ----------
    text : str
        Quoted argument without its surrounding double quotes.

    Returns
    -------
    str
        The original argument.
    """
    text = text.replace("%%", "%").replace("\\\\", "\\")
    return re.sub(r'\\(["`$\\])', r"\1", text)


def _linux_register(*, icon):
    """Install the Skyline MIME types, desktop entry and icon for the user.

    Parameters
    ----------
    icon : Path
        Application icon file.

    Returns
    -------
    list of str
        Files created.
    """
    existing_types = _linux_existing_mime_types(data_dirs=_linux_mime_search_dirs())
    mime_types = {
        fmt: f"application/x-dipy-{fmt.lstrip('.').replace('.', '-')}"
        for fmt in SKYLINE_FORMATS
        if fmt not in existing_types
    }
    desktop_mime_types = dict.fromkeys(
        [
            f"x-scheme-handler/{SKYLINE_URL_SCHEME}",
            *mime_types.values(),
            *(mime_type for types in existing_types.values() for mime_type in types),
        ]
    )
    mime_entries = ""
    for fmt, mime_type in mime_types.items():
        # A compressed format is a subclass of ``application/gzip``: GIO sniffs the
        # gzip magic bytes of such a file and only keeps a glob match whose type
        # is a subclass of what it finds.
        subclass = (
            '    <sub-class-of type="application/gzip"/>\n'
            if fmt.endswith(".gz")
            else ""
        )
        mime_entries += (
            f'  <mime-type type="{mime_type}">\n'
            f"    <comment>{APP_NAME} data</comment>\n"
            f"{subclass}"
            f'    <icon name="{_LINUX_ICON_NAME}"/>\n'
            # Without it the theme's generic file icon outranks the icon in hicolor.
            f'    <generic-icon name="{_LINUX_ICON_NAME}"/>\n'
            f'    <glob pattern="*{fmt}"/>\n'
            "  </mime-type>\n"
        )
    mime_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">\n'
        f"{mime_entries}"
        "</mime-info>\n"
    )

    quoted_arguments = [_desktop_entry_quote(argument) for argument in _app_command()]
    entry = "\n".join(
        [
            "[Desktop Entry]",
            "Type=Application",
            f"Name={APP_NAME}",
            "Comment=Diffusion MRI and tractography viewer",
            f"Exec={' '.join(quoted_arguments)} %u",
            f"Icon={_LINUX_ICON_NAME}",
            "Terminal=false",
            "Categories=Science;MedicalSoftware;Graphics;",
            "MimeType=" + ";".join(desktop_mime_types) + ";",
            "",
        ]
    )

    mime_file, desktop_file = _linux_files()
    theme_icon = _linux_theme_icon()
    theme_icon.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(icon, theme_icon)
    for path, text in ((mime_file, mime_xml), (desktop_file, entry)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    _linux_refresh_databases()
    _run_tool(
        "xdg-mime",
        "default",
        desktop_file.name,
        f"x-scheme-handler/{SKYLINE_URL_SCHEME}",
        step=f"registering the {SKYLINE_URL_SCHEME}:// handler",
    )

    locations = [mime_file, desktop_file]
    desktop_dir = _linux_desktop_dir()
    if desktop_dir.is_dir():
        shortcut = desktop_dir / desktop_file.name
        shortcut.write_text(entry, encoding="utf-8")
        shortcut.chmod(0o755)
        _run_tool(
            "gio",
            "set",
            str(shortcut),
            "metadata::trusted",
            "true",
            step="trusting the desktop icon",
        )
        locations.append(shortcut)
    return [str(location) for location in locations]


def _linux_unregister():
    """Remove the MIME package, desktop entries and icon of Skyline."""
    mime_file, desktop_file = _linux_files()
    for path in (
        mime_file,
        desktop_file,
        _linux_desktop_dir() / desktop_file.name,
        _linux_theme_icon(),
    ):
        path.unlink(missing_ok=True)
    _linux_refresh_databases()


def _linux_registered_interpreter():
    """Read the interpreter of the registered desktop entry.

    Returns
    -------
    str or None
        Interpreter in the ``Exec`` line, or None when it is not registered.
    """
    _, desktop_file = _linux_files()
    try:
        text = desktop_file.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'^Exec="((?:[^"\\]|\\.)*)"', text, flags=re.MULTILINE)
    if match is None:
        return None
    return _desktop_entry_unquote(match.group(1))


def _macos_app_path():
    """Locate the Skyline application bundle of the user.

    Returns
    -------
    Path
        ``~/Applications/<APP_NAME>.app``.
    """
    return Path.home() / "Applications" / f"{APP_NAME}.app"


def _macos_bundle_is_ours(app):
    """Tell whether an application bundle was created by Skyline.

    Parameters
    ----------
    app : Path
        Application bundle to inspect.

    Returns
    -------
    bool
        True when its bundle identifier is ``BUNDLE_ID``; False when it is
        another application, or has no readable ``Info.plist``.
    """
    try:
        with open(app / "Contents" / "Info.plist", "rb") as plist_file:
            return plistlib.load(plist_file).get("CFBundleIdentifier") == BUNDLE_ID
    except (OSError, plistlib.InvalidFileException):
        return False


def _macos_owns_desktop_link(shortcut, app):
    """Tell whether a Desktop entry is a symbolic link to the Skyline bundle.

    Parameters
    ----------
    shortcut : Path
        Desktop entry to inspect.
    app : Path
        Skyline application bundle.

    Returns
    -------
    bool
        True when ``shortcut`` is a symbolic link whose target is ``app``.
    """
    return shortcut.is_symlink() and shortcut.readlink() == app


def _macos_register(*, icon):
    """Build, register and link the Skyline application bundle.

    Parameters
    ----------
    icon : Path
        Application icon file in ``.icns`` format.

    Returns
    -------
    list of str
        Application bundle and Desktop link created.

    Raises
    ------
    RuntimeError
        If ``app`` exists and is not a Skyline application, or if
        ``osacompile`` cannot create the application.
    """
    python = _applescript_string(shlex.quote(sys.executable))
    module = _applescript_string(f" -c {shlex.quote(LAUNCH_BOOTSTRAP)}")
    background = _applescript_string(" > /dev/null 2>&1 &")
    space = _applescript_string(" ")
    start = f"do shell script {python} & {module}"
    script = [
        "on run",
        f"    {start} & {background}",
        "end run",
        "on open location theURL",
        f"    {start} & {space} & quoted form of theURL & {background}",
        "end open location",
        "on open theFiles",
        "    repeat with f in theFiles",
        (f"        {start} & {space} & quoted form of POSIX path of f & {background}"),
        "    end repeat",
        "end open",
    ]

    app = _macos_app_path()
    app.parent.mkdir(parents=True, exist_ok=True)
    if app.exists():
        if not _macos_bundle_is_ours(app):
            raise RuntimeError(
                f"{app} exists and is not a {APP_NAME} application; move it away "
                "and run dipy_skyline --setup again."
            )
        shutil.rmtree(app)
    script_lines = [argument for line in script for argument in ("-e", line)]
    try:
        subprocess.run(["osacompile", "-o", str(app), *script_lines], check=True)
    except (OSError, subprocess.CalledProcessError) as err:
        raise RuntimeError(f"Could not create {app} with osacompile: {err}") from err

    try:
        resources = app / "Contents" / "Resources"
        # The applet ships its own icon, named by CFBundleIconFile and possibly by
        # CFBundleIconName in an asset catalog, which macOS prefers over the file.
        shutil.copy2(icon, resources / "applet.icns")
        (resources / "Assets.car").unlink(missing_ok=True)
        (resources / "dipy-python.txt").write_text(sys.executable, encoding="utf-8")

        plist_path = app / "Contents" / "Info.plist"
        with open(plist_path, "rb") as plist_file:
            plist = plistlib.load(plist_file)
        plist.update(
            {
                "CFBundleIdentifier": BUNDLE_ID,
                "CFBundleName": APP_NAME,
                "CFBundleIconFile": "applet",
                "CFBundleURLTypes": [
                    {
                        "CFBundleURLName": BUNDLE_ID,
                        "CFBundleURLSchemes": [SKYLINE_URL_SCHEME],
                    }
                ],
                "CFBundleDocumentTypes": [
                    {
                        "CFBundleTypeName": f"{APP_NAME} data",
                        "CFBundleTypeRole": "Viewer",
                        "LSHandlerRank": "Alternate",
                        "CFBundleTypeExtensions": _extensions(),
                    }
                ],
            }
        )
        plist.pop("CFBundleIconName", None)
        with open(plist_path, "wb") as plist_file:
            plistlib.dump(plist, plist_file)
    except OSError:
        shutil.rmtree(app, ignore_errors=True)
        raise

    _run_tool(
        "codesign", "--force", "--deep", "--sign", "-", str(app), step="signing the app"
    )
    _run_tool(_LSREGISTER, "-f", str(app), step="registering the app")

    locations = [app]
    shortcut = Path.home() / "Desktop" / app.name
    if _macos_owns_desktop_link(shortcut, app):
        shortcut.unlink()
    if shortcut.exists() or shortcut.is_symlink():
        logger.warning(f"{shortcut} is not a link to {app}; skipped the Desktop link.")
    elif shortcut.parent.is_dir():
        shortcut.symlink_to(app)
        locations.append(shortcut)
    return [str(location) for location in locations]


def _macos_unregister():
    """Unregister and remove the Skyline application bundle and its Desktop link.

    A bundle or Desktop link that Skyline did not create is left in place.
    """
    app = _macos_app_path()
    if _macos_bundle_is_ours(app):
        _run_tool(_LSREGISTER, "-u", str(app), step="unregistering the app")
        shutil.rmtree(app, ignore_errors=True)
    elif app.exists():
        logger.warning(f"{app} is not a {APP_NAME} application; left it in place.")
    shortcut = Path.home() / "Desktop" / app.name
    if _macos_owns_desktop_link(shortcut, app):
        shortcut.unlink()


def _macos_registered_interpreter():
    """Read the interpreter stored in the Skyline application bundle.

    Returns
    -------
    str or None
        Stored interpreter, or None when the bundle is missing.
    """
    marker = _macos_app_path() / "Contents" / "Resources" / "dipy-python.txt"
    try:
        return marker.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def registered_interpreter():
    """Interpreter that the registered Skyline application runs with.

    Returns
    -------
    str or None
        Path of the interpreter stored in the operating system registration,
        or None when Skyline is not registered or that interpreter no longer
        exists.
    """
    if sys.platform == "win32":
        interpreter = _windows_registered_interpreter()
    elif sys.platform == "darwin":
        interpreter = _macos_registered_interpreter()
    else:
        interpreter = _linux_registered_interpreter()
    if interpreter is None or not Path(interpreter).exists():
        return None
    return interpreter


def register_app():
    """Register Skyline as an application that runs with this interpreter.

    An existing registration, of this or another environment, is replaced.

    Returns
    -------
    list of str
        Locations created or updated.

    Raises
    ------
    RuntimeError
        If the application icon is unavailable, or the application cannot be
        built.
    """
    relative_path = SKYLINE_APP_ICONS.get(sys.platform, SKYLINE_APP_ICON_DEFAULT)
    try:
        icon = fetch_asset(ASSET_APP, relative_path)
    except OSError as err:
        raise RuntimeError(
            f"Could not get the Skyline icon {relative_path} from the DIPY data "
            f"repository: {err}. Check the network connection and run "
            "dipy_skyline --setup again."
        ) from err
    previous = registered_interpreter()
    current = _app_command()[0]
    if previous is not None and previous != current:
        logger.info(f"Switching {APP_NAME} from {previous} to {current}.")

    if sys.platform == "win32":
        return _windows_register(icon=icon)
    if sys.platform == "darwin":
        return _macos_register(icon=icon)
    return _linux_register(icon=icon)


def unregister_app():
    """Remove the Skyline application registration, whichever environment made it."""
    previous = registered_interpreter()
    if sys.platform == "win32":
        _windows_unregister()
    elif sys.platform == "darwin":
        _macos_unregister()
    else:
        _linux_unregister()

    if previous is None:
        logger.info("Nothing to remove.")
    else:
        logger.info(f"Removed {APP_NAME} registered for {previous}.")


def try_register_app():
    """Register Skyline as an application, logging instead of raising on failure.

    A partial registration is removed when the operating system refuses a write.

    Returns
    -------
    list of str or None
        Locations created or updated, or None when the registration failed.
    """
    try:
        locations = register_app()
    except RuntimeError as err:
        logger.error(str(err))
        return None
    except OSError as err:
        unregister_app()
        logger.error(f"Could not register {APP_NAME}: {err}")
        return None

    logger.info(f"{APP_NAME} is installed as an application for {sys.executable}.")
    for location in locations:
        logger.info(f"  {location}")
    return locations


def _has_display():
    """Tell whether a graphical session is available to register an application.

    Returns
    -------
    bool
        Always True on Windows and macOS; on Linux, True when ``DISPLAY`` or
        ``WAYLAND_DISPLAY`` is set.
    """
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _interpreter_has_dipy(interpreter):
    """Tell whether DIPY is installed for an interpreter.

    Parameters
    ----------
    interpreter : str
        Path of the Python interpreter to ask.

    Returns
    -------
    bool
        True when ``interpreter`` is the running one, has ``dipy`` installed, or
        did not answer within 10 seconds (unknown state); False when it cannot
        be run or has no ``dipy``.
    """
    if interpreter == _app_command()[0]:
        return True
    try:
        result = subprocess.run(
            [interpreter, "-c", "import importlib.metadata as m;m.version('dipy')"],
            capture_output=True,
            cwd=Path(interpreter).parent,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return True
    except OSError:
        return False
    return result.returncode == 0


def ensure_app_registered():
    """Register Skyline as an application unless a working registration exists.

    The opt-out ``DIPY_SKYLINE_NO_REGISTER`` (non-empty, as in the DIPY test
    suite) is checked first. A registration made by any environment is kept
    while its interpreter exists and finds DIPY. Nothing is registered on Linux
    without a graphical session. Nothing is asked and nothing is installed;
    failures are logged, never raised.

    Returns
    -------
    bool
        True when a working registration exists afterwards.
    """
    if os.environ.get(NO_REGISTER_ENV_VAR):
        return False
    interpreter = registered_interpreter()
    if interpreter is not None and _interpreter_has_dipy(interpreter):
        return True
    if not _has_display():
        logger.info(
            "No graphical session; skipped registering Skyline. "
            "Run dipy_skyline --setup from a desktop session."
        )
        return False

    try:
        locations = try_register_app()
    except Exception as err:  # noqa: BLE001
        logger.warning(f"Could not register {APP_NAME}: {err}")
        locations = None
    if locations is None:
        logger.info("To register Skyline yourself, run: dipy_skyline --setup")
        return False

    missing = missing_viz_requirements()
    if missing:
        logger.warning(
            f"Skyline needs packages that are missing from {sys.executable}: "
            f"{', '.join(missing)}. Run dipy_skyline --setup to install them."
        )
    return True


def setup_app():
    """Install the missing viewer packages and register Skyline as an application.

    Returns
    -------
    bool
        True when the viewer packages are installed and Skyline is registered.
    """
    missing = missing_viz_requirements()
    if missing:
        logger.info(
            f"Skyline needs packages that are missing from {sys.executable}: "
            f"{', '.join(missing)}"
        )
        if importlib.util.find_spec("pip") is not None:
            command = [sys.executable, "-m", "pip", "install", *missing]
        else:
            uv = shutil.which("uv")
            if uv is None:
                logger.error(
                    "Install them with your package manager into "
                    f"{sys.executable}: {' '.join(missing)}"
                )
                return False
            command = [uv, "pip", "install", "--python", sys.executable, *missing]

        if sys.platform == "win32":
            manual = subprocess.list2cmdline(command)
        else:
            manual = shlex.join(command)
        confirmed = False
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                answer = input("Install them now? [Y/n] ")
                confirmed = not answer.strip().lower().startswith("n")
            except EOFError:
                pass
        if not confirmed:
            logger.error(
                f"Run this command, then run dipy_skyline --setup again: {manual}"
            )
            return False

        if subprocess.run(command, check=False).returncode != 0:
            logger.error(
                "Package installation failed. Close every running DIPY program "
                f"(Skyline viewers, dipy_auto report servers) and run: {manual}"
            )
            return False

        importlib.invalidate_caches()
        still_missing = missing_viz_requirements()
        if still_missing:
            logger.error(
                f"Still missing after the installation: {', '.join(still_missing)}"
            )
            return False

    return try_register_app() is not None


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
