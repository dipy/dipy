import os
from os.path import isfile, join
import tempfile
import zipfile

from numpy.testing import assert_equal
import pytest

from dipy.data import get_fnames
from dipy.utils.optpkg import optional_package
from dipy.workflows.atlasing import BundleAtlasFlow

_, has_fury, _ = optional_package("fury")


@pytest.mark.skipif(not has_fury, reason="Requires Fury")
def test_discrete_bundle_atlas():
    with tempfile.TemporaryDirectory() as in_dir:
        example_tracts = get_fnames(name="minimal_bundles")
        with zipfile.ZipFile(example_tracts, "r") as zip_ref:
            zip_ref.extractall(in_dir)

        flow = BundleAtlasFlow()

        # Check that final output was generated with default version
        with tempfile.TemporaryDirectory() as out_dir:
            flow.run(in_dir, out_dir=out_dir)

            assert_equal(isfile(join(out_dir, "AF_L.trk")), True)
            assert_equal(isfile(join(out_dir, "CST_R.trk")), True)
            assert_equal(isfile(join(out_dir, "CC_ForcepsMajor.trk")), True)

        # Check additional outputs (whole brain tractogram and temporary files)

        with tempfile.TemporaryDirectory() as out_dir:
            flow.run(in_dir, out_dir=out_dir, save_temp=True, merge_out=True)

            outputs = os.listdir(out_dir)
            temp_folder = [x for x in outputs if x.startswith("dipy_atlas_temp")][0]

            assert_equal(isfile(join(out_dir, "AF_L.trk")), True)
            assert_equal(isfile(join(out_dir, "CST_R.trk")), True)
            assert_equal(isfile(join(out_dir, "CC_ForcepsMajor.trk")), True)
            assert_equal(isfile(join(out_dir, "whole_brain.trk")), True)

            temp_files = os.listdir(join(out_dir, temp_folder, "AF_L", "step_0"))
            trk_files = [file for file in temp_files if file.endswith(".trk")]
            png_files = [file for file in temp_files if file.endswith(".png")]

            assert_equal(len(trk_files), 5)
            assert_equal(len(png_files), 15)
