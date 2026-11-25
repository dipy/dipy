import os
import sys
from urllib.parse import quote

import pytest

from dipy.desktop.constants import MAX_LINK_FILES
from dipy.desktop.links import (
    checked_file,
    is_skyline_link,
    skyline_targets,
    skyline_url,
)


def _touch(path):
    path.write_bytes(b"")
    return path


def test_skyline_url_round_trips_unusual_paths(tmp_path):
    name = "sub 01 #1 %20 a&b=c \u00e9 [x]" + ("" if os.name == "nt" else "?")
    folder = tmp_path / name
    folder.mkdir()
    volume = _touch(folder / "fa.nii.gz")

    url = skyline_url(paths=[volume])

    assert url.startswith("dipy-skyline://open?path=")
    assert skyline_targets(url) == [volume]


def test_skyline_url_round_trips_several_paths_in_order(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    tractogram = _touch(tmp_path / "bundle.trk")

    url = skyline_url(paths=[tractogram, volume])

    assert skyline_targets(url) == [tractogram, volume]


def test_skyline_targets_accepts_paths_file_urls_and_browser_slash(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")

    assert skyline_targets(str(volume)) == [volume]
    assert skyline_targets(volume.as_uri()) == [volume]

    query = skyline_url(paths=[volume]).split("?", 1)[1]
    assert skyline_targets(f"dipy-skyline://open/?{query}") == [volume]


def test_skyline_targets_drops_duplicates(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    quoted = quote(str(volume), safe="")

    assert skyline_targets(f"dipy-skyline://open?path={quoted}&path={quoted}") == [
        volume
    ]


@pytest.mark.parametrize("name", ["lh.pial", "pial.L.surf.gii", "mesh.gii.gz"])
def test_skyline_targets_accepts_surface_files(tmp_path, name):
    surface = _touch(tmp_path / name)

    assert skyline_targets(skyline_url(paths=[surface])) == [surface]


def test_skyline_targets_rejects_bad_links(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    notes = tmp_path / "notes.txt"
    notes.write_text("not an image", encoding="utf-8")
    quoted = quote(str(volume), safe="")

    bad_arguments = [
        f"https://open?path={quoted}",
        "dipy-skyline://[",
        "file://[",
        f"dipy-skyline://evil?path={quoted}",
        f"dipy-skyline://open?path={quoted}&extra=1",
        "dipy-skyline://open",
        "dipy-skyline://open?path=fa.nii.gz",
        skyline_url(paths=[tmp_path / "missing.nii.gz"]),
        skyline_url(paths=[tmp_path]),
        skyline_url(paths=[notes]),
    ]
    for argument in bad_arguments:
        with pytest.raises(ValueError, match="Skyline cannot open"):
            skyline_targets(argument)


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("dipy-skyline://open?path=x", True),
        ("DIPY-SKYLINE://open?path=x", True),
        ("dipy-skyline://[", True),
        ("  dipy-skyline://]", True),
        ("file:///tmp/fa.nii.gz", False),
        ("/tmp/fa.nii.gz", False),
        ("https://[", False),
    ],
)
def test_is_skyline_link_recognises_links_even_when_malformed(argument, expected):
    assert is_skyline_link(argument) is expected


def test_skyline_targets_refuses_a_link_with_one_missing_file(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    link = skyline_url(paths=[volume, tmp_path / "missing.nii.gz"])

    with pytest.raises(ValueError, match="no such file"):
        skyline_targets(link)


def test_skyline_targets_refuses_a_link_with_too_many_files(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")
    query = "&".join([f"path={quote(str(volume), safe='')}"] * (MAX_LINK_FILES + 1))

    with pytest.raises(ValueError, match="at most"):
        skyline_targets(f"dipy-skyline://open?{query}")


def test_skyline_url_rejects_empty_and_oversized_lists(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")

    with pytest.raises(ValueError, match="at least one file"):
        skyline_url(paths=[])
    with pytest.raises(ValueError, match="at most"):
        skyline_url(paths=[volume] * (MAX_LINK_FILES + 1))


def test_checked_file_requires_an_absolute_path():
    with pytest.raises(ValueError, match="must be absolute"):
        checked_file("relative.nii.gz")


@pytest.mark.parametrize("text", ["file:///tmp/fa.nii.gz", "FILE:/tmp/fa.nii.gz"])
def test_checked_file_refuses_urls(text):
    with pytest.raises(ValueError, match="not a file path"):
        checked_file(text)


@pytest.mark.skipif(sys.platform != "win32", reason="UNC paths only exist on Windows")
@pytest.mark.parametrize(
    "path", ["//attacker.example/share/fa.nii", "\\\\attacker.example\\share\\fa.nii"]
)
def test_skyline_targets_rejects_network_paths(path):
    link = f"dipy-skyline://open?path={quote(path, safe='')}"

    with pytest.raises(ValueError, match="network paths are not allowed"):
        skyline_targets(link)


def test_skyline_targets_rejects_file_urls_with_a_remote_host():
    with pytest.raises(ValueError, match="network paths are not allowed"):
        skyline_targets("file://attacker.example/tmp/fa.nii.gz")


def test_cannot_open_message_truncates_long_arguments():
    with pytest.raises(ValueError, match="Skyline cannot open") as err:
        skyline_targets("http://" + "a" * 500)

    assert len(str(err.value)) < 300
