from http.server import HTTPServer, SimpleHTTPRequestHandler
import importlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
from threading import Thread

import pytest

from dipy.data import SPHERE_FILES
import dipy.data.fetcher as fetcher
from dipy.data.fetcher import (
    DIPY_MIRROR_URL,
    MIRRORABLE_HOSTS,
    FetcherError,
    _already_there_msg,
    _get_file_md5,
    _get_mirror_url,
    _make_fetcher,
    check_md5,
    fetch_data,
)

_BAD_MD5 = "0" * 32


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


@pytest.fixture
def sphere_server():
    """HTTP server serving the sphere file directory; yields (base_url, name, md5)."""
    path = SPHERE_FILES["symmetric362"]
    server, base_url, original_cwd = _free_port_server(str(Path(path).parent))
    yield base_url, Path(path).name, _get_file_md5(path)
    server.shutdown()
    os.chdir(original_cwd)


@pytest.fixture
def dipy_log_propagate():
    """Enable propagation on the dipy logger so pytest caplog can capture records."""
    dipy_logger = logging.getLogger("dipy")
    old = dipy_logger.propagate
    dipy_logger.propagate = True
    yield
    dipy_logger.propagate = old


def test_check_md5_correct():
    fd, fname = tempfile.mkstemp()
    os.close(fd)
    try:
        stored = _get_file_md5(fname)
        assert check_md5(fname, stored_md5=stored) is None
    finally:
        os.unlink(fname)


def test_check_md5_none_skips():
    fd, fname = tempfile.mkstemp()
    os.close(fd)
    try:
        assert check_md5(fname, stored_md5=None) is None
    finally:
        os.unlink(fname)


def test_check_md5_wrong_raises():
    fd, fname = tempfile.mkstemp()
    os.close(fd)
    try:
        with pytest.raises(FetcherError):
            check_md5(fname, stored_md5="wrong")
    finally:
        os.unlink(fname)


def test_check_md5_error_message_contains_checksums():
    fd, fname = tempfile.mkstemp()
    os.write(fd, b"dipy")
    os.close(fd)
    actual = _get_file_md5(fname)
    stored = _BAD_MD5
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

        sf()
        assert (tmp_path / "sphere_name").is_file()
        assert not (tmp_path / "sphere_name2").is_file()
        assert _get_file_md5(tmp_path / "sphere_name") == stored_md5

        sf(include_optional=True)
        assert (tmp_path / "sphere_name2").is_file()
        assert _get_file_md5(tmp_path / "sphere_name2") == stored_md5_642
    finally:
        server.shutdown()
        os.chdir(original_cwd)


def test_fetch_data_http(tmp_path):
    symmetric362 = SPHERE_FILES["symmetric362"]
    md5 = _get_file_md5(symmetric362)
    bad_md5 = "8" * len(md5)
    name = Path(symmetric362).name
    newfile = tmp_path / "testfile.txt"

    server, base_url, original_cwd = _free_port_server(str(Path(symmetric362).parent))
    url = base_url + name
    try:
        fetch_data({"testfile.txt": (url, md5)}, str(tmp_path))
        assert newfile.exists()

        newfile.write_bytes(newfile.read_bytes() + b"junk")
        fetch_data({"testfile.txt": (url, md5)}, str(tmp_path))
        assert _get_file_md5(newfile) == md5

        with pytest.raises(FetcherError):
            fetch_data({"testfile.txt": (url, bad_md5)}, str(tmp_path))
    finally:
        server.shutdown()
        os.chdir(original_cwd)


def test_fetch_data_creates_missing_folder(tmp_path, sphere_server):
    base_url, name, md5 = sphere_server
    new = tmp_path / "new_dir"
    fetch_data({name: (base_url + name, md5)}, new)
    assert new.is_dir()
    assert (new / name).is_file()


def test_fetch_data_no_error_if_folder_exists(tmp_path, sphere_server):
    base_url, name, md5 = sphere_server
    fetch_data({name: (base_url + name, md5)}, tmp_path)
    fetch_data({name: (base_url + name, md5)}, tmp_path)


def test_fetch_data_skips_matching_md5(tmp_path, sphere_server):
    # Copy sphere file to tmp_path so the local md5 matches.
    # The URL points to the same file, but fetch_data must skip the download.
    # If it doesn't skip, it re-downloads and overwrites — content stays the
    # same so we verify by checking the file was not modified.
    base_url, name, md5 = sphere_server
    src = SPHERE_FILES["symmetric362"]
    dst = tmp_path / name
    shutil.copy(src, dst)
    mtime_before = dst.stat().st_mtime_ns

    fetch_data({name: (base_url + name, md5)}, str(tmp_path))

    assert dst.stat().st_mtime_ns == mtime_before


def test_fetch_data_redownloads_stale_file(tmp_path, sphere_server):
    base_url, name, md5 = sphere_server
    dst = tmp_path / name
    dst.write_bytes(b"stale content")

    fetch_data({name: (base_url + name, md5)}, str(tmp_path))

    assert _get_file_md5(dst) == md5


def test_fetch_data_raise_on_error_false_does_not_raise(tmp_path, sphere_server):
    # Bad md5 → _get_file_data retries without sleep, then raises FetcherError.
    # With raise_on_error=False, fetch_data must not propagate the error.
    base_url, name, _ = sphere_server
    fetch_data(
        {name: (base_url + name, _BAD_MD5)},
        tmp_path,
        raise_on_error=False,
    )


def test_fetch_data_raise_on_error_false_continues(tmp_path, sphere_server):
    # First file has wrong md5 (fails); second has correct md5 (succeeds).
    # fetch_data must process both when raise_on_error=False.
    base_url, name, md5 = sphere_server
    fetch_data(
        {
            "bad.gz": (base_url + name, _BAD_MD5),
            "good.gz": (base_url + name, md5),
        },
        tmp_path,
        raise_on_error=False,
    )
    assert (tmp_path / "good.gz").is_file()
    assert _get_file_md5(tmp_path / "good.gz") == md5


def test_fetch_data_raise_on_error_false_cleans_partial(tmp_path, sphere_server):
    # On the final retry _get_file_data leaves the file on disk before raising.
    # fetch_data's cleanup code must remove it.
    base_url, name, _ = sphere_server
    fetch_data(
        {"p.gz": (base_url + name, _BAD_MD5)},
        tmp_path,
        raise_on_error=False,
    )
    assert not (tmp_path / "p.gz").exists()


def test_fetch_data_data_size_logged(
    tmp_path, sphere_server, caplog, dipy_log_propagate
):
    base_url, name, md5 = sphere_server
    with caplog.at_level(logging.INFO, logger="dipy"):
        fetch_data({name: (base_url + name, md5)}, tmp_path, data_size="99 MB")
    assert "99 MB" in caplog.text


def test_fetch_data_all_skip_logs_already_there(
    tmp_path, sphere_server, caplog, dipy_log_propagate
):
    base_url, name, md5 = sphere_server
    shutil.copy(SPHERE_FILES["symmetric362"], tmp_path / name)
    with caplog.at_level(logging.INFO, logger="dipy"):
        fetch_data({name: (base_url + name, md5)}, str(tmp_path))
    assert "already" in caplog.text.lower()


def test_dipy_home():
    old_home = os.environ.pop("DIPY_HOME", None)
    try:
        importlib.reload(fetcher)
        expected = str(Path("~").expanduser() / ".dipy")
        assert str(fetcher.dipy_home) == expected

        with tempfile.TemporaryDirectory() as test_path:
            os.environ["DIPY_HOME"] = test_path
            importlib.reload(fetcher)
            assert str(fetcher.dipy_home) == test_path
    finally:
        if old_home is not None:
            os.environ["DIPY_HOME"] = old_home
        elif "DIPY_HOME" in os.environ:
            del os.environ["DIPY_HOME"]
        importlib.reload(fetcher)


@pytest.mark.parametrize("host", MIRRORABLE_HOSTS)
def test_get_mirror_url_mirrorable_host(host):
    result = _get_mirror_url(f"https://{host}/some/path/data.nii.gz")
    assert result is not None
    assert result.startswith(DIPY_MIRROR_URL)


@pytest.mark.parametrize("host", MIRRORABLE_HOSTS)
def test_get_mirror_url_path_preserved(host):
    suffix = "/bucket/handle/1773/file.nii.gz"
    result = _get_mirror_url(f"https://{host}{suffix}")
    assert suffix.lstrip("/") in result


def test_get_mirror_url_non_mirrorable_returns_none():
    assert _get_mirror_url("https://example.com/data.nii.gz") is None


def test_get_mirror_url_empty_string_returns_none():
    assert _get_mirror_url("") is None


def test_get_mirror_url_garbage_returns_none():
    assert _get_mirror_url("not-a-url") is None


def test_get_mirror_url_docstring_example():
    url = "https://stacks.stanford.edu/file/druid:yx282xq2090/dwi.nii.gz"
    expected = (
        "https://workshop.dipy.org/services/data/file/druid:yx282xq2090/dwi.nii.gz"
    )
    assert _get_mirror_url(url) == expected


def test_get_mirror_url_no_double_slash():
    result = _get_mirror_url("https://zenodo.org/record/999/files/f.nii.gz")
    path_part = result.split(DIPY_MIRROR_URL, 1)[-1]
    assert "//" not in path_part


def test_already_there_msg_str(tmp_path):
    _already_there_msg(str(tmp_path))


def test_already_there_msg_path(tmp_path):
    _already_there_msg(tmp_path)


def test_already_there_msg_logs_folder(tmp_path, caplog, dipy_log_propagate):
    with caplog.at_level(logging.INFO, logger="dipy"):
        _already_there_msg(str(tmp_path))
    assert str(tmp_path) in caplog.text
