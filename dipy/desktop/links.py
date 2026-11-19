"""Links, ``file:`` URLs and paths that ask Skyline to open files."""

import glob
from pathlib import Path, PureWindowsPath
import sys
import urllib.parse
import urllib.request

from dipy.desktop.constants import MAX_LINK_FILES, SKYLINE_FORMATS, SKYLINE_URL_SCHEME


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


def skyline_url(*, paths):
    """Link that makes the registered Skyline application open ``paths``.

    Parameters
    ----------
    paths : sequence of str or Path
        Files to display, at most ``MAX_LINK_FILES``.

    Returns
    -------
    str
        ``dipy-skyline://open?path=...`` with each absolute path
        percent-encoded.

    Raises
    ------
    ValueError
        If ``paths`` is empty or names more than ``MAX_LINK_FILES`` files.
    """
    if not paths:
        raise ValueError("paths must name at least one file")
    if len(paths) > MAX_LINK_FILES:
        raise ValueError(
            f"paths names {len(paths)} files; a link opens at most {MAX_LINK_FILES}"
        )
    query = urllib.parse.urlencode(
        [("path", str(Path(path).absolute())) for path in paths],
        quote_via=urllib.parse.quote,
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


def checked_file(text):
    """Check that a path names a file Skyline may open.

    Parameters
    ----------
    text : str
        Absolute path as received from a link, a ``file:`` URL or another
        launch.

    Returns
    -------
    Path
        Path of an existing file of a format Skyline can display.

    Raises
    ------
    ValueError
        If ``text`` is a URL, is not absolute, is a network path, cannot be
        read, names no existing file, or does not end with one of
        ``SKYLINE_FORMATS``.
    """
    if text[:5].lower() == "file:" or "://" in text:
        raise _cannot_open(text, "not a file path")
    path = Path(text)
    if not path.is_absolute():
        raise _cannot_open(text, "the path must be absolute")
    if _is_network_path(text):
        raise _cannot_open(text, "network paths are not allowed")
    try:
        is_file = path.is_file()
    except OSError as err:
        raise _cannot_open(text, "the file cannot be read") from err
    if not is_file:
        raise _cannot_open(text, "no such file")
    if not path.name.endswith(SKYLINE_FORMATS):
        raise _cannot_open(text, "unsupported file type")
    return path


def _split_url(argument):
    """Split an argument as a URL.

    Parameters
    ----------
    argument : str
        Link, ``file:`` URL or path.

    Returns
    -------
    urllib.parse.SplitResult
        The parts of ``argument``.

    Raises
    ------
    ValueError
        If ``argument`` cannot be split, for instance ``dipy-skyline://[``.
    """
    try:
        return urllib.parse.urlsplit(argument)
    except ValueError as err:
        raise _cannot_open(argument, "not a valid URL") from err


def is_skyline_link(argument):
    """Tell whether an argument is a ``dipy-skyline://`` link.

    Parameters
    ----------
    argument : str
        Command-line argument of the launcher.

    Returns
    -------
    bool
        True when the scheme is ``dipy-skyline``, including for a link too
        malformed to split, so that it is reported like any bad link.
    """
    try:
        return urllib.parse.urlsplit(argument).scheme == SKYLINE_URL_SCHEME
    except ValueError:
        return argument.lstrip().lower().startswith(f"{SKYLINE_URL_SCHEME}:")


def skyline_targets(argument):
    """Files Skyline is asked to open by a link, a ``file:`` URL or a path.

    Parameters
    ----------
    argument : str
        ``dipy-skyline://open?path=...`` link with up to ``MAX_LINK_FILES``
        ``path`` values, ``file:`` URL or file path.

    Returns
    -------
    list of Path
        Absolute paths of existing files of a format Skyline can display, in
        the order given and without duplicates.

    Raises
    ------
    ValueError
        If the argument is not one of the accepted forms, or one of its paths
        fails ``checked_file``; a link with any bad path is refused whole.
    """
    parts = _split_url(argument)
    if parts.scheme == SKYLINE_URL_SCHEME:
        if parts.netloc != "open" or parts.path not in ("", "/"):
            raise _cannot_open(argument, f"not a {SKYLINE_URL_SCHEME}://open link")
        query = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
        values = query.get("path", [])
        if list(query) != ["path"] or not values:
            raise _cannot_open(
                argument, "the link must carry one or more absolute paths"
            )
        if len(values) > MAX_LINK_FILES:
            raise _cannot_open(argument, f"a link opens at most {MAX_LINK_FILES} files")
        paths = [checked_file(value) for value in values]
    elif parts.scheme == "file":
        if parts.netloc not in ("", "localhost"):
            raise _cannot_open(argument, "network paths are not allowed")
        paths = [checked_file(urllib.request.url2pathname(parts.path))]
    elif "://" in argument:
        raise _cannot_open(argument, f"not a {SKYLINE_URL_SCHEME}://open link")
    else:
        paths = [checked_file(str(Path(argument).absolute()))]
    return list(dict.fromkeys(paths))
