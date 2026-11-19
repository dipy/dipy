import pytest

from dipy.desktop.constants import MAX_FILES
from dipy.desktop.environment import open_in_skyline


def test_open_in_skyline_refuses_no_files():
    with pytest.raises(ValueError, match="1 to"):
        open_in_skyline([])


def test_open_in_skyline_refuses_too_many_files(tmp_path):
    paths = [tmp_path / f"{i}.nii.gz" for i in range(MAX_FILES + 1)]

    with pytest.raises(ValueError, match="1 to"):
        open_in_skyline(paths)
