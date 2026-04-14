import pytest

from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI import elements

_, has_imgui, _ = optional_package("imgui_bundle", min_version="1.92.600")


@pytest.mark.skipif(not has_imgui, reason="Requires imgui_bundle>=1.92.600")
def test_ensure_last_dir_creates_missing_directory(tmp_path, monkeypatch):
    missing_dir = tmp_path / "new" / ".dipy"
    monkeypatch.setattr(elements, "_LAST_DIR", missing_dir)

    resolved_dir = elements._ensure_last_dir()

    assert resolved_dir == missing_dir
    assert missing_dir.exists()
    assert missing_dir.is_dir()


@pytest.mark.skipif(not has_imgui, reason="Requires imgui_bundle>=1.92.600")
def test_ensure_last_dir_uses_parent_when_last_dir_is_file(tmp_path, monkeypatch):
    file_path = tmp_path / "last_location.txt"
    file_path.write_text("placeholder")
    monkeypatch.setattr(elements, "_LAST_DIR", file_path)

    resolved_dir = elements._ensure_last_dir()

    assert resolved_dir == file_path.parent
    assert resolved_dir.exists()
    assert resolved_dir.is_dir()
