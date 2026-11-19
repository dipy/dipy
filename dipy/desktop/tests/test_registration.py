import pytest

from dipy.desktop.registration import (
    _LINUX_ENTRY_MARKER,
    _can_write_shortcut,
    _desktop_entry_quote,
    _desktop_entry_unquote,
    _extensions,
    _linux_entry_is_ours,
    _linux_existing_mime_types,
    _restore_files,
    _snapshot_files,
)


def test_restore_files_undoes_overwrites_and_creations(tmp_path):
    kept = tmp_path / "kept.txt"
    kept.write_bytes(b"old")
    created = tmp_path / "created.txt"
    snapshot = _snapshot_files([kept, created])

    kept.write_bytes(b"new")
    created.write_bytes(b"new")
    _restore_files(snapshot)

    assert kept.read_bytes() == b"old"
    assert not created.exists()


def test_shortcut_guard_spares_foreign_entries(tmp_path):
    shortcut = tmp_path / "dipy-skyline.desktop"
    is_ours = _linux_entry_is_ours
    assert _can_write_shortcut(shortcut, is_ours=is_ours)

    shortcut.write_text("[Desktop Entry]\nName=Other\n", encoding="utf-8")
    assert not is_ours(shortcut)
    assert not _can_write_shortcut(shortcut, is_ours=is_ours)

    marker = _LINUX_ENTRY_MARKER
    shortcut.write_text(f"[Desktop Entry]\n{marker}\n", encoding="utf-8")
    assert is_ours(shortcut)
    assert _can_write_shortcut(shortcut, is_ours=is_ours)


def test_extensions_leave_compressed_formats_out():
    extensions = _extensions()

    assert "gz" not in extensions
    assert "nii" in extensions


def test_linux_existing_mime_types_reuses_types_of_other_applications(tmp_path):
    database = tmp_path / "mime"
    database.mkdir()
    (database / "globs2").write_text(
        "# c\n"
        "50:image/x-nifti:*.nii\n"
        "50:application/x-dipy-trk:*.trk\n"
        "50:application/gzip:*.gz\n",
        encoding="utf-8",
    )

    found = _linux_existing_mime_types(data_dirs=[tmp_path, tmp_path / "missing"])

    assert found == {".nii": ["image/x-nifti"]}


def test_desktop_entry_quote_round_trips_reserved_characters():
    argument = '/opt/my%env/b\\in/py"th$on'

    quoted = _desktop_entry_quote(argument)

    assert quoted.startswith('"') and quoted.endswith('"')
    assert _desktop_entry_unquote(quoted[1:-1]) == argument


def test_snapshot_files_leaves_dangling_links_alone(tmp_path):
    link = tmp_path / "dangling"
    try:
        link.symlink_to(tmp_path / "missing")
    except OSError:
        pytest.skip("symbolic links need a privilege on this system")
    absent = tmp_path / "absent"

    snapshot = _snapshot_files([link, absent])
    _restore_files(snapshot)

    assert snapshot == {absent: None}
    assert link.is_symlink()
