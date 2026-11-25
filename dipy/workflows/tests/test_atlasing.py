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

            outputs = os.listdir(out_dir)
            out_dir_name = [x for x in outputs if x.startswith("bundle_atlasing")][0]
            out_dir_trk = os.path.join(out_dir, out_dir_name)

            assert_equal(isfile(join(out_dir_trk, "AF_L.trk")), True)
            assert_equal(isfile(join(out_dir_trk, "CST_R.trk")), True)
            assert_equal(isfile(join(out_dir_trk, "CC_ForcepsMajor.trk")), True)

        # Check additional outputs (whole brain tractogram and temporary files)

        with tempfile.TemporaryDirectory() as out_dir:
            flow.run(in_dir, out_dir=out_dir, save_temp=True, merge_out=True)

            outputs = os.listdir(out_dir)
            out_dir_name = [x for x in outputs if x.startswith("bundle_atlasing")][0]
            out_dir_trk = os.path.join(out_dir, out_dir_name)
            temp_dir = os.path.join(out_dir_trk, "temp")

            assert_equal(isfile(join(out_dir_trk, "AF_L.trk")), True)
            assert_equal(isfile(join(out_dir_trk, "CST_R.trk")), True)
            assert_equal(isfile(join(out_dir_trk, "CC_ForcepsMajor.trk")), True)
            assert_equal(isfile(join(out_dir_trk, "whole_brain.trk")), True)

            temp_files = os.listdir(join(temp_dir, "AF_L", "step_0"))
            trk_files = [file for file in temp_files if file.endswith(".trk")]
            png_files = [file for file in temp_files if file.endswith(".png")]

            assert_equal(len(trk_files), 5)
            assert_equal(len(png_files), 15)
