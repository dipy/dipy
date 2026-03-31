import gzip
import importlib
import io
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import tempfile
from threading import Thread
from unittest.mock import MagicMock, call, patch
from urllib.error import HTTPError, URLError
import zipfile

import numpy.testing as npt
import pytest

from dipy.data import SPHERE_FILES
import dipy.data.fetcher as fetcher
from dipy.data.fetcher import (
    DIPY_MIRROR_URL,
    MIRRORABLE_HOSTS,
    FetcherError,
    _already_there_msg,
    _get_file_md5,
    _get_file_data,
    _get_mirror_url,
    _make_fetcher,
    check_md5,
    fetch_data,
    fetch_deepn4_test,
    fetch_evac_test,
    fetch_synthseg_torch_weights,
    fetch_synthseg_test,
    get_fnames,
)


# ===========================================================================
# Shared test helpers
# ===========================================================================

def _free_port_server(directory):
    """Start an HTTP server on an OS-assigned free port serving *directory*.

    Returns (server, base_url, original_cwd).
    Caller MUST call server.shutdown() and os.chdir(original_cwd) in finally.
    """
    original_cwd = os.getcwd()
    os.chdir(directory)
    server = HTTPServer(("127.0.0.1", 0), SimpleHTTPRequestHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    return server, f"http://127.0.0.1:{port}/", original_cwd


def _make_gz_file(tmp_path, inner_name, content=b"hello gz"):
    """Write a plain .gz file (not .tar.gz) and return its path."""
    gz_path = tmp_path / f"{inner_name}.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(content)
    return gz_path


def _make_zip_file(tmp_path, members):
    """Write a .zip with {filename: bytes} members and return its path."""
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return zip_path

# ===========================================================================
# Pre-existing test bug fixes
# ===========================================================================

class TestCheckMd5:

    def test_correct_md5_returns_none(self):
        fd, fname = tempfile.mkstemp()
        os.close(fd)                                 # This was missing, caused PermissionError on Windows
        try:
            stored = _get_file_md5(fname)
            assert check_md5(fname, stored_md5=stored) is None
        finally:
            os.unlink(fname)

    def test_none_stored_md5_skips_check(self):
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:
            assert check_md5(fname, stored_md5=None) is None
        finally:
            os.unlink(fname)

    def test_wrong_md5_raises_fetcher_error(self):
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:
            with pytest.raises(FetcherError):
                check_md5(fname, stored_md5="wrong")
        finally:
            os.unlink(fname)

    def test_error_message_contains_filename_and_both_checksums(self):
        content = b"dipy"
        fd, fname = tempfile.mkstemp()
        os.write(fd, content)
        os.close(fd)
        actual = _get_file_md5(fname)
        stored = "0" * 32
        try:
            with pytest.raises(FetcherError) as exc_info:
                check_md5(fname, stored_md5=stored)
            msg = str(exc_info.value)
            assert fname in msg
            assert stored in msg
            assert actual in msg
        finally:
            os.unlink(fname)


def test_make_fetcher_http(tmp_path):
    """Port hardcoding and chdir leak fixed."""
    symmetric362 = SPHERE_FILES["symmetric362"]
    symmetric642 = SPHERE_FILES["symmetric642"]
    stored_md5 = _get_file_md5(symmetric362)
    stored_md5_642 = _get_file_md5(symmetric642)

    server, base_url, original_cwd = _free_port_server(str(Path(symmetric362).parent))
    try:
        sf = _make_fetcher(
            "sphere_fetcher",
            str(tmp_path),
            base_url,
            [Path(symmetric362).name, Path(symmetric642).name],
            ["sphere_name", "sphere_name2"],
            md5_list=[stored_md5, stored_md5_642],
            optional_fnames=["sphere_name2"],
        )

        # No bare except — failures propagate to pytest.
        sf()
        assert (tmp_path / "sphere_name").is_file()
        assert not (tmp_path / "sphere_name2").is_file()
        assert _get_file_md5(tmp_path / "sphere_name") == stored_md5

        sf(include_optional=True)
        assert (tmp_path / "sphere_name2").is_file()
        assert _get_file_md5(tmp_path / "sphere_name2") == stored_md5_642
    finally:
        server.shutdown()
        os.chdir(original_cwd)  # FIX: always restored


def test_fetch_data_http(tmp_path):
    """Port hardcoding and chdir leak fixed."""
    symmetric362 = SPHERE_FILES["symmetric362"]
    md5 = _get_file_md5(symmetric362)
    bad_md5 = "8" * len(md5)
    name = Path(symmetric362).name
    newfile = tmp_path / "testfile.txt"

    server, base_url, original_cwd = _free_port_server(str(Path(symmetric362).parent))
    url = base_url + name
    try:
        # Normal download.
        fetch_data({"testfile.txt": (url, md5)}, str(tmp_path))
        assert newfile.exists()

        # Corrupted file is re-downloaded.
        newfile.write_bytes(newfile.read_bytes() + b"junk")
        fetch_data({"testfile.txt": (url, md5)}, str(tmp_path))
        assert _get_file_md5(newfile) == md5

        # Bad md5 must raise FetcherError.
        with pytest.raises(FetcherError):
            fetch_data({"testfile.txt": (url, bad_md5)}, str(tmp_path))
    finally:
        server.shutdown()
        os.chdir(original_cwd)

    def test_dipy_home():
        test_path = "TEST_PATH"
        if "DIPY_HOME" in os.environ:
            old_home = os.environ["DIPY_HOME"]
            del os.environ["DIPY_HOME"]
        else:
            old_home = None

        importlib.reload(fetcher)

        npt.assert_string_equal(
            fetcher.dipy_home, str(Path("~").expanduser() / ".dipy")
        )
        os.environ["DIPY_HOME"] = test_path
        importlib.reload(fetcher)
        npt.assert_string_equal(fetcher.dipy_home, test_path)

        # return to previous state
        if old_home:
            os.environ["DIPY_HOME"] = old_home
