from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import http.server
from pathlib import Path
from threading import Thread

import pytest

from dipy.utils import assets

BLOB = bytes(range(256)) * 4096
BLOB_SHA256 = hashlib.sha256(BLOB).hexdigest()


def test_asset_home_is_a_dot_folder_in_home():
    assert assets.asset_home("demo-app") == Path("~").expanduser() / ".demo-app"


@pytest.mark.parametrize("app", ["demo-app", "demo.app", "Demo App"])
def test_check_app_accepts_plain_names(app):
    assert assets.check_app(app) is None


@pytest.mark.parametrize("app", ["", "a/b", "a\\b"])
def test_check_app_rejects_bad_names(app):
    with pytest.raises(ValueError, match="app must be a non-empty name"):
        assets.check_app(app)


@pytest.mark.parametrize("app", ["", "a/b", "a\\b"])
def test_asset_functions_reject_bad_app_names(app):
    with pytest.raises(ValueError, match="app must be a non-empty name"):
        assets.asset_home(app)
    with pytest.raises(ValueError, match="app must be a non-empty name"):
        assets.fetch_asset(app, "x.png")


@pytest.mark.parametrize(
    "path",
    ["", "/etc/passwd", "../x.png", "icons/../../x.png", "..\\x.png", "C:x.png"],
)
def test_fetch_asset_rejects_paths_leaving_the_asset_folder(path):
    with pytest.raises(
        ValueError, match="relative_path must stay inside the asset folder"
    ):
        assets.fetch_asset("demo-app", path)


def test_fetch_asset_rejects_assets_without_a_known_digest():
    with pytest.raises(ValueError, match="not a known asset"):
        assets.fetch_asset("demo-app", "x.png")


class _AssetHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            port = self.server.server_address[1]
            self.send_header("Location", f"http://127.0.0.1:{port}/blob.bin")
            self.end_headers()
        else:
            super().do_GET()


@pytest.fixture
def served_blob(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    (served / "blob.bin").write_bytes(BLOB)
    handler = partial(_AssetHandler, directory=str(served))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_download_verified_keeps_one_intact_file_under_parallel_downloads(
    served_blob, tmp_path
):
    folder = tmp_path / "cache"
    folder.mkdir()
    target = folder / "blob.bin"

    with ThreadPoolExecutor(8) as pool:
        list(
            pool.map(
                lambda _: assets.download_verified(
                    f"{served_blob}/blob.bin", target, sha256=BLOB_SHA256
                ),
                range(8),
            )
        )

    assert target.read_bytes() == BLOB
    assert [path.name for path in folder.iterdir()] == ["blob.bin"]


def test_download_verified_rejects_content_with_another_digest(served_blob, tmp_path):
    target = tmp_path / "blob.bin"
    target.write_bytes(b"cached")

    with pytest.raises(OSError, match="expected"):
        assets.download_verified(f"{served_blob}/blob.bin", target, sha256="0" * 64)

    assert target.read_bytes() == b"cached"
    assert [path.name for path in tmp_path.iterdir() if path.is_file()] == ["blob.bin"]


def test_download_verified_refuses_a_redirect_to_plain_http(served_blob, tmp_path):
    target = tmp_path / "blob.bin"

    with pytest.raises(OSError, match="not https"):
        assets.download_verified(f"{served_blob}/redirect", target, sha256=BLOB_SHA256)

    assert not target.exists()
