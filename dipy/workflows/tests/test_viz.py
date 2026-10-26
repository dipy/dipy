"""End-to-end tests for the Skyline visualization workflow."""

import os

import numpy as np
import pytest

from dipy.io.image import save_nifti
from dipy.utils.optpkg import optional_package

_, has_fury_v2, _ = optional_package("fury", min_version="2.0.0")
if not has_fury_v2:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from PIL import Image

    from dipy.workflows.viz import SkylineFlow


@pytest.mark.parametrize("stealth, dipy_value", [(True, None), (False, "1")])
def test_skyline_flow_captures_offscreen(tmp_path, stealth, dipy_value):
    volume_path = tmp_path / "volume.nii.gz"
    save_nifti(
        str(volume_path),
        np.arange(64, dtype=np.float32).reshape(4, 4, 4),
        np.eye(4),
    )

    previous_dipy = os.environ.get("DIPY_OFFSCREEN")
    previous_fury = os.environ.get("FURY_OFFSCREEN")
    try:
        if dipy_value is None:
            os.environ.pop("DIPY_OFFSCREEN", None)
        else:
            os.environ["DIPY_OFFSCREEN"] = dipy_value
        os.environ["FURY_OFFSCREEN"] = "0"

        SkylineFlow().run(
            [str(volume_path)],
            stealth=stealth,
            out_dir=str(tmp_path),
            out_stealth_png="workflow.png",
        )

        assert os.environ.get("DIPY_OFFSCREEN") == dipy_value
        assert os.environ.get("FURY_OFFSCREEN") == "0"
        with Image.open(tmp_path / "workflow.png") as image:
            assert any(low != high for low, high in image.getextrema()[:3])
    finally:
        for key, value in (
            ("DIPY_OFFSCREEN", previous_dipy),
            ("FURY_OFFSCREEN", previous_fury),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
