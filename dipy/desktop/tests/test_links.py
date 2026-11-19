import os
import sys
from urllib.parse import quote

import pytest

from dipy.desktop.links import skyline_target, skyline_url


def _touch(path):
    path.write_bytes(b"")
    return path


def test_skyline_url_round_trips_unusual_paths(tmp_path):
    name = "sub 01 #1 %20 a&b=c \u00e9 [x]" + ("" if os.name == "nt" else "?")
    folder = tmp_path / name
    folder.mkdir()
    volume = _touch(folder / "fa.nii.gz")

    url = skyline_url(path=volume)

    assert url.startswith("dipy-skyline://open?path=")
    assert skyline_target(url) == volume


def test_skyline_target_accepts_paths_file_urls_and_browser_slash(tmp_path):
    volume = _touch(tmp_path / "fa.nii.gz")

    assert skyline_target(str(volume)) == volume
    assert skyline_target(volume.as_uri()) == volume

    query = skyline_url(path=volume).split("?", 1)[1]
    assert skyline_target(f"dipy-skyline://open/?{query}") == volume


@pytest.mark.parametrize("name", ["lh.pial", "pial.L.surf.gii", "mesh.gii.gz"])
def test_skyline_target_accepts_surface_files(tmp_path, name):
    surface = _touch(tmp_path / name)

    assert skyline_target(skyline_url(path=surface)) == surface


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
        skyline_url(path=tmp_path / "missing.nii.gz"),
        skyline_url(path=tmp_path),
        skyline_url(path=notes),
    ]
    for argument in bad_arguments:
        with pytest.raises(ValueError, match="Skyline cannot open"):
            skyline_target(argument)


@pytest.mark.skipif(sys.platform != "win32", reason="UNC paths only exist on Windows")
@pytest.mark.parametrize(
    "path", ["//attacker.example/share/fa.nii", "\\\\attacker.example\\share\\fa.nii"]
)
def test_skyline_target_rejects_network_paths(path):
    link = f"dipy-skyline://open?path={quote(path, safe='')}"

    with pytest.raises(ValueError, match="network paths are not allowed"):
        skyline_target(link)


def test_skyline_target_rejects_file_urls_with_a_remote_host():
    with pytest.raises(ValueError, match="network paths are not allowed"):
        skyline_target("file://attacker.example/tmp/fa.nii.gz")


def test_cannot_open_message_truncates_long_arguments():
    with pytest.raises(ValueError, match="Skyline cannot open") as err:
        skyline_target("http://" + "a" * 500)

    assert len(str(err.value)) < 300
