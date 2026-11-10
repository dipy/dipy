"""Per-user cache of the assets an application downloads from the DIPY data repository."""

from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
from urllib.request import urlopen

from dipy.utils.logging import logger

DIPY_DATA_MIRROR = "https://github.com/dipy/dipy_data/raw/refs/heads/master"
_DOWNLOAD_TIMEOUT = 30


def check_app(app):
    """Check that an application name can be used as a folder name.

    Parameters
    ----------
    app : str
        Application name to validate.

    Raises
    ------
    ValueError
        If ``app`` is empty or contains a path separator.
    """
    if not app or "/" in app or "\\" in app:
        raise ValueError(
            f"app must be a non-empty name without path separators, got {app!r}"
        )


def asset_home(app):
    """Return the per-user folder of an application.

    Parameters
    ----------
    app : str
        Application name, for instance ``"dipy-skyline"``. It must not contain
        path separators.

    Returns
    -------
    Path
        ``~/.<app>``.

    Raises
    ------
    ValueError
        If ``app`` is empty or contains a path separator.
    """
    check_app(app)
    return Path("~").expanduser() / f".{app}"


def fetch_asset(app, relative_path):
    """Return the cached asset of an application, downloading it when missing.

    Parameters
    ----------
    app : str
        Application name. It is also the folder of the data repository that
        holds the application assets.
    relative_path : str
        Forward-slash path of the asset below ``<app>/assets`` in the data
        repository, for instance ``"fonts/Inter_18pt-Regular.ttf"``.

    Returns
    -------
    Path
        Location of the asset below ``~/.<app>/assets``.

    Raises
    ------
    ValueError
        If ``app`` is empty or contains a path separator, or if
        ``relative_path`` is empty, absolute, has a backslash or drive, or
        contains ``..``.
    urllib.error.URLError
        If the asset is missing locally and the download fails.
    OSError
        If the asset cannot be written to the cache.
    """
    check_app(app)
    posix_path = PurePosixPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or PureWindowsPath(relative_path).drive
        or posix_path.is_absolute()
        or ".." in posix_path.parts
    ):
        raise ValueError(
            f"relative_path must stay inside the asset folder, got {relative_path!r}"
        )

    path = asset_home(app) / "assets" / relative_path
    if path.exists():
        return path

    logger.info(f"Downloading {app} asset {relative_path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    url = f"{DIPY_DATA_MIRROR}/{app}/assets/{relative_path}"
    try:
        with (
            urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as response,
            open(partial, "wb") as out,
        ):
            shutil.copyfileobj(response, out)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path
