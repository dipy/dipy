import os
import time

from dipy.desktop.dialogs import (
    CONFIRM_LOCK_SECONDS,
    acquire_confirmation_slot,
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
