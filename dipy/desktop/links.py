"""Links, ``file:`` URLs and paths that ask Skyline to open files."""

import glob
from pathlib import Path, PureWindowsPath
import sys
import urllib.parse
import urllib.request

from dipy.desktop.constants import SKYLINE_FORMATS, SKYLINE_URL_SCHEME


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
    shown = argument if len(argument) <= 200 else f"{argument[:200]}..."
    return ValueError(f"Skyline cannot open {shown!r}: {reason}")


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
