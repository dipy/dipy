"""Per-user cache of the assets an application downloads from the DIPY data repository."""

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from dipy.utils.logging import logger

DIPY_DATA_REVISION = "d26d3e9cc1ae4140a59fb5b66af69078b7d7b530"
DIPY_DATA_MIRROR = f"https://github.com/dipy/dipy_data/raw/{DIPY_DATA_REVISION}"
ASSET_SHA256 = {
    "dipy-skyline/fonts/Inter_18pt-Regular.ttf": (
        "3e5f90a0138b38de4cf4d779ad78391974ea1df776b9164842bdcbb60ce383c5"
    ),
    "dipy-skyline/fonts/fontawesome-webfont.ttf": (
        "d2f0593540b0e33ba6de255a54f272d466e31144806956bea8cfdbf7edffc9bd"
    ),
    "dipy-skyline/images/dipy-logo.png": (
        "4439b4506243c9832d48a6ebe7efa7f2f71555cc6b5348fdef10e5bcea6c21b2"
    ),
    "dipy-skyline/icons/skyline.ico": (
        "f18b17a5857511c527089585a3e64724acfcc2a92b6ceb9bbfeed5e91d69a4f8"
    ),
    "dipy-skyline/icons/skyline.icns": (
        "0844c4bc7b4399e46cddca8ee4229136855b088b0c3ac89eab31d8c470098ee1"
    ),
    "dipy-skyline/icons/skyline.png": (
        "631c7f57aff619d5d1a9ec4cb612ac975834958681761b37b03c7629d944f783"
    ),
}
_DOWNLOAD_TIMEOUT = 30
_CHUNK_SIZE = 1 << 16


class _HttpsRedirectHandler(HTTPRedirectHandler):
    """Follow redirects, but only to ``https`` addresses."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise URLError(f"refused a redirect to {newurl}: it is not https")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = build_opener(_HttpsRedirectHandler)


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


def _file_sha256(path):
    """Return the SHA-256 digest of a file.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download_verified(url, path, *, sha256):
    """Download a file and keep it only when its SHA-256 digest is the expected one.

    The content goes to a unique temporary file in the folder of ``path``, then
    replaces ``path`` in one step, so that parallel downloads cannot corrupt it
    and a failed one leaves nothing behind.

    Parameters
    ----------
    url : str
        Address to download. A redirect to anything but ``https`` is refused.
    path : Path
        Destination file; its folder must exist.
    sha256 : str
        Expected hexadecimal SHA-256 digest of the content.

    Raises
    ------
    OSError
        If the download fails (``urllib.error.URLError`` is one), the content
        does not match ``sha256``, or the file cannot be written.
    """
    digest = hashlib.sha256()
    with (
        _OPENER.open(url, timeout=_DOWNLOAD_TIMEOUT) as response,
        tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".part",
            delete_on_close=False,
        ) as partial,
    ):
        while chunk := response.read(_CHUNK_SIZE):
            digest.update(chunk)
            partial.write(chunk)
        if digest.hexdigest() != sha256:
            raise OSError(
                f"{url} has SHA-256 {digest.hexdigest()}, expected {sha256}; "
                "the download is incomplete or the file was changed."
            )
        partial.close()
        try:
            Path(partial.name).replace(path)
        except PermissionError:
            # Windows refuses to replace a file that another process just
            # replaced or has open; the winner downloaded the same content.
            if not path.is_file() or _file_sha256(path) != sha256:
                raise


def fetch_asset(app, relative_path):
    """Return the cached asset of an application, downloading it when needed.

    The asset is downloaded from a fixed revision of the data repository and
    must match the SHA-256 digest listed in ``ASSET_SHA256``. A cached file
    that does not match, for instance after an interrupted download, is
    downloaded again.

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
        If ``app`` is empty or contains a path separator, if
        ``relative_path`` is empty, absolute, has a backslash or drive, or
        contains ``..``, or if the asset is not listed in ``ASSET_SHA256``.
    OSError
        If the asset must be downloaded and the download fails or does not
        match its digest (``urllib.error.URLError`` is one), or the asset
        cannot be written to the cache.
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
    sha256 = ASSET_SHA256.get(f"{app}/{relative_path}")
    if sha256 is None:
        raise ValueError(f"{app}/{relative_path} is not a known asset")

    path = asset_home(app) / "assets" / relative_path
    if path.is_file() and _file_sha256(path) == sha256:
        return path

    logger.info(f"Downloading {app} asset {relative_path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    download_verified(
        f"{DIPY_DATA_MIRROR}/{app}/assets/{relative_path}", path, sha256=sha256
    )
    return path
