from pathlib import Path

import pytest

from dipy.utils import assets


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
