import pytest


def _load_elements_or_skip():
    pytest.importorskip("PIL")
    pytest.importorskip("fury", minversion="2.0.0a6")
    from dipy.viz.skyline.UI import elements

    return elements


def test_ensure_last_dir_creates_missing_directory(tmp_path, monkeypatch):
    elements = _load_elements_or_skip()
    missing_dir = tmp_path / "new" / ".dipy"
    monkeypatch.setattr(elements, "_LAST_DIR", missing_dir)

    resolved_dir = elements._ensure_last_dir()

    assert resolved_dir == missing_dir
    assert missing_dir.exists()
    assert missing_dir.is_dir()


def test_ensure_last_dir_uses_parent_when_last_dir_is_file(tmp_path, monkeypatch):
    elements = _load_elements_or_skip()
    file_path = tmp_path / "last_location.txt"
    file_path.write_text("placeholder")
    monkeypatch.setattr(elements, "_LAST_DIR", file_path)

    resolved_dir = elements._ensure_last_dir()

    assert resolved_dir == file_path.parent
    assert resolved_dir.exists()
    assert resolved_dir.is_dir()
