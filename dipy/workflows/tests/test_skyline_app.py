import os
import sys
import time
from urllib.parse import quote

import pytest

from dipy.workflows import skyline_app


def _touch(path):
    path.write_bytes(b"")
    return path


def test_skyline_url_round_trips_unusual_paths(tmp_path):
    name = "sub 01 #1 %20 a&b=c \u00e9 [x]" + ("" if os.name == "nt" else "?")
    folder = tmp_path / name
    folder.mkdir()
    volume = _touch(folder / "fa.nii.gz")

    url = skyline_app.skyline_url(path=volume)

    assert url.startswith("dipy-skyline://open?path=")
    assert skyline_app.skyline_target(url) == volume


def test_skyline_target_accepts_paths_file_urls_and_browser_slash(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")

    assert skyline_app.skyline_target(str(volume)) == volume
    assert skyline_app.skyline_target(volume.as_uri()) == volume

    query = skyline_app.skyline_url(path=volume).split("?", 1)[1]
    assert skyline_app.skyline_target(f"dipy-skyline://open/?{query}") == volume


@pytest.mark.parametrize("name", ["lh.pial", "pial.L.surf.gii", "mesh.gii.gz"])
def test_skyline_target_accepts_surface_files(tmp_path, name):
    surface = _touch(tmp_path / name)

    assert skyline_app.skyline_target(skyline_app.skyline_url(path=surface)) == surface


def test_skyline_target_rejects_bad_links(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    notes = tmp_path / "notes.txt"
    notes.write_text("not an image", encoding="utf-8")
    quoted = quote(str(volume), safe="")

    bad_arguments = [
        f"https://open?path={quoted}",
        f"dipy-skyline://evil?path={quoted}",
        f"dipy-skyline://open?path={quoted}&extra=1",
        f"dipy-skyline://open?path={quoted}&path={quoted}",
        "dipy-skyline://open",
        "dipy-skyline://open?path=fa.nii.gz",
        skyline_app.skyline_url(path=tmp_path / "missing.nii.gz"),
        skyline_app.skyline_url(path=tmp_path),
        skyline_app.skyline_url(path=notes),
    ]
    for argument in bad_arguments:
        with pytest.raises(ValueError, match="Skyline cannot open"):
            skyline_app.skyline_target(argument)


@pytest.mark.skipif(sys.platform != "win32", reason="UNC paths only exist on Windows")
@pytest.mark.parametrize(
    "path", ["//attacker.example/share/fa.nii", "\\\\attacker.example\\share\\fa.nii"]
)
def test_skyline_target_rejects_network_paths(path):
    link = f"dipy-skyline://open?path={quote(path, safe='')}"

    with pytest.raises(ValueError, match="network paths are not allowed"):
        skyline_app.skyline_target(link)


def test_skyline_target_rejects_file_urls_with_a_remote_host():
    with pytest.raises(ValueError, match="network paths are not allowed"):
        skyline_app.skyline_target("file://attacker.example/tmp/fa.nii.gz")


def test_restore_files_undoes_overwrites_and_creations(tmp_path):
    kept = tmp_path / "kept.txt"
    kept.write_bytes(b"old")
    created = tmp_path / "created.txt"
    snapshot = skyline_app._snapshot_files([kept, created])

    kept.write_bytes(b"new")
    created.write_bytes(b"new")
    skyline_app._restore_files(snapshot)

    assert kept.read_bytes() == b"old"
    assert not created.exists()


def test_shortcut_guard_spares_foreign_entries(tmp_path):
    shortcut = tmp_path / "dipy-skyline.desktop"
    is_ours = skyline_app._linux_entry_is_ours
    assert skyline_app._can_write_shortcut(shortcut, is_ours=is_ours)

    shortcut.write_text("[Desktop Entry]\nName=Other\n", encoding="utf-8")
    assert not is_ours(shortcut)
    assert not skyline_app._can_write_shortcut(shortcut, is_ours=is_ours)

    marker = skyline_app._LINUX_ENTRY_MARKER
    shortcut.write_text(f"[Desktop Entry]\n{marker}\n", encoding="utf-8")
    assert is_ours(shortcut)
    assert skyline_app._can_write_shortcut(shortcut, is_ours=is_ours)


def test_extensions_leave_compressed_formats_out():
    extensions = skyline_app._extensions()

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

    found = skyline_app._linux_existing_mime_types(
        data_dirs=[tmp_path, tmp_path / "missing"]
    )

    assert found == {".nii": ["image/x-nifti"]}


def test_desktop_entry_quote_round_trips_reserved_characters():
    argument = '/opt/my%env/b\\in/py"th$on'

    quoted = skyline_app._desktop_entry_quote(argument)

    assert quoted.startswith('"') and quoted.endswith('"')
    assert skyline_app._desktop_entry_unquote(quoted[1:-1]) == argument


def test_linux_dialog_text_never_shows_a_path_as_markup():
    link = '<a href="https://evil.example">x</a> & y\\n'

    assert skyline_app._linux_dialog_text("zenity", link) == link
    notify = skyline_app._linux_dialog_text("notify-send", link)
    assert "<" not in notify and "&lt;a" in notify and "&amp;" in notify
    kdialog = skyline_app._linux_dialog_text("kdialog", link)
    assert kdialog.startswith("<qt>") and kdialog.endswith("</qt>")
    assert "<a " not in kdialog and "&lt;a" in kdialog and "&amp;" in kdialog
    assert "\\\\n" in kdialog


def test_cannot_open_message_truncates_long_arguments():
    with pytest.raises(ValueError, match="Skyline cannot open") as err:
        skyline_app.skyline_target("http://" + "a" * 500)

    assert len(str(err.value)) < 300


def test_confirmation_slot_is_taken_once_until_released_or_stale(tmp_path):
    lock = tmp_path / "confirm.lock"
    assert skyline_app._acquire_confirmation_slot(lock=lock)
    assert not skyline_app._acquire_confirmation_slot(lock=lock)

    lock.unlink()
    assert skyline_app._acquire_confirmation_slot(lock=lock)

    stale = time.time() - skyline_app._CONFIRM_LOCK_SECONDS - 1
    os.utime(lock, (stale, stale))
    assert skyline_app._acquire_confirmation_slot(lock=lock)


def test_snapshot_files_leaves_dangling_links_alone(tmp_path):
    link = tmp_path / "dangling"
    try:
        link.symlink_to(tmp_path / "missing")
    except OSError:
        pytest.skip("symbolic links need a privilege on this system")
    absent = tmp_path / "absent"

    snapshot = skyline_app._snapshot_files([link, absent])
    skyline_app._restore_files(snapshot)

    assert snapshot == {absent: None}
    assert link.is_symlink()
