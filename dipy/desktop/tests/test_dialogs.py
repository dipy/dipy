import os
import time

from dipy.desktop.dialogs import (
    CONFIRM_LOCK_SECONDS,
    acquire_confirmation_slot,
    confirm_message,
    display_text,
    linux_dialog_text,
)


def test_linux_dialog_text_never_shows_a_path_as_markup():
    link = '<a href="https://evil.example">x</a> & y\\n'

    assert linux_dialog_text("zenity", link) == link
    notify = linux_dialog_text("notify-send", link)
    assert "<" not in notify and "&lt;a" in notify and "&amp;" in notify
    kdialog = linux_dialog_text("kdialog", link)
    assert kdialog.startswith("<qt>") and kdialog.endswith("</qt>")
    assert "<a " not in kdialog and "&lt;a" in kdialog and "&amp;" in kdialog
    assert "\\\\n" in kdialog


def test_confirmation_slot_is_taken_once_until_released_or_stale(tmp_path):
    lock = tmp_path / "confirm.lock"
    assert acquire_confirmation_slot(lock=lock)
    assert not acquire_confirmation_slot(lock=lock)

    lock.unlink()
    assert acquire_confirmation_slot(lock=lock)

    stale = time.time() - CONFIRM_LOCK_SECONDS - 1
    os.utime(lock, (stale, stale))
    assert acquire_confirmation_slot(lock=lock)


def test_confirm_message_names_every_file(tmp_path):
    volume = tmp_path / "fa.nii.gz"
    tractogram = tmp_path / "bundle.trk"
    for path in (volume, tractogram):
        path.write_bytes(b"")

    message = confirm_message([volume, tractogram], [volume, tractogram])

    assert str(volume) in message and str(tractogram) in message
    assert "Open these 2 files?" in message
    assert "Open this file?" in confirm_message([volume], [volume])


def test_confirm_message_shows_where_a_link_leads(tmp_path):
    link = tmp_path / "shortcut.nii.gz"
    target = tmp_path / "elsewhere" / "fa.nii.gz"

    message = confirm_message([link], [target])

    assert f"(a link to {target})" in message


def test_display_text_escapes_what_could_fake_or_hide_dialog_text():
    assert display_text("a\nb\u202ec") == "a\\u000ab\\u202ec"
    assert display_text("a\u00a0b\u3000c") == "a\\u00a0b\\u3000c"
    assert display_text("a\U000e0001b") == "a\\U000e0001b"
    assert display_text("fa \u00e9.nii.gz") == "fa \u00e9.nii.gz"
