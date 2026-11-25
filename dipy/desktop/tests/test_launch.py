from pathlib import PureWindowsPath

import pytest

from dipy.desktop.constants import MAX_FILES
from dipy.desktop.launch import _restore_drive, parse_arguments, resolve_targets
from dipy.desktop.links import skyline_url


def _touch(path):
    path.write_bytes(b"")
    return path


def test_a_link_must_come_alone(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")

    targets, errors, from_link = parse_arguments(
        [skyline_url(paths=[volume]), str(volume)]
    )

    assert targets == [] and from_link
    assert len(errors) == 1 and "must come alone" in errors[0]


def test_a_link_carries_all_its_files_and_asks_for_confirmation(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    tractogram = _touch(tmp_path / "bundle.trk")

    targets, errors, from_link = parse_arguments(
        [skyline_url(paths=[volume, tractogram])]
    )

    assert targets == [volume, tractogram] and not errors and from_link


def test_paths_are_deduplicated_in_order(tmp_path):
    first = _touch(tmp_path / "a.nii.gz")
    second = _touch(tmp_path / "b.trk")

    targets, errors, from_link = parse_arguments([str(first), str(second), str(first)])

    assert targets == [first, second] and not errors and not from_link


def test_a_malformed_link_is_reported_as_a_link_that_is_refused():
    targets, errors, from_link = parse_arguments(["dipy-skyline://["])

    assert targets == [] and from_link
    assert len(errors) == 1 and "Skyline cannot open" in errors[0]


def test_a_missing_path_does_not_stop_the_valid_ones(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")

    targets, errors, _ = parse_arguments(
        [str(tmp_path / "missing.nii.gz"), str(volume)]
    )

    assert targets == [volume]
    assert len(errors) == 1 and "no such file" in errors[0]


def test_too_many_arguments_are_refused(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")

    targets, errors, _ = parse_arguments([str(volume)] * (MAX_FILES + 1))

    assert targets == []
    assert len(errors) == 1 and str(MAX_FILES) in errors[0]


def test_resolve_targets_returns_the_files_themselves(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")

    shown, resolved, errors = resolve_targets([volume])

    assert shown == [volume] and resolved == [volume.resolve()] and not errors


def test_resolve_targets_follows_links_and_checks_the_destination(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    notes = tmp_path / "notes.txt"
    notes.write_text("not an image", encoding="utf-8")
    good_link = tmp_path / "good.nii.gz"
    bad_link = tmp_path / "bad.nii.gz"
    try:
        good_link.symlink_to(volume)
        bad_link.symlink_to(notes)
    except OSError:
        pytest.skip("symbolic links need a privilege on this system")

    shown, resolved, errors = resolve_targets([good_link, bad_link])

    assert shown == [good_link] and resolved == [volume.resolve()]
    assert len(errors) == 1 and "unsupported file type" in errors[0]


MAPPED_ROOT = PureWindowsPath("Z:/")
SHARE_ROOT = PureWindowsPath("//server/share/")


def test_a_file_on_a_mapped_drive_keeps_its_drive_letter():
    destination = SHARE_ROOT / "data" / "fa.nii.gz"

    restored = _restore_drive(destination, MAPPED_ROOT, SHARE_ROOT)

    assert restored == PureWindowsPath("Z:/data/fa.nii.gz")


def test_a_link_to_another_share_stays_a_network_path():
    destination = PureWindowsPath("//other/share/fa.nii.gz")

    assert _restore_drive(destination, MAPPED_ROOT, SHARE_ROOT) == destination


def test_a_local_drive_is_left_alone():
    local_root = PureWindowsPath("C:/")
    destination = PureWindowsPath("C:/data/fa.nii.gz")

    assert _restore_drive(destination, local_root, local_root) == destination
