from os import listdir
from os.path import isfile, join
import zipfile

from nibabel.tmpdirs import TemporaryDirectory
from numpy.testing import assert_equal
import pytest

from dipy.data import get_fnames
from dipy.utils.optpkg import optional_package
from dipy.workflows.atlasing import BundleAtlasFlow

_, has_fury, _ = optional_package("fury")


@pytest.mark.skipif(not has_fury, reason="Requires Fury")
def test_discrete_bundle_atlas():
    with TemporaryDirectory() as in_dir:
        example_tracts = get_fnames("minimal_bundles")
        with zipfile.ZipFile(example_tracts, "r") as zip_ref:
            zip_ref.extractall(in_dir)

        flow = BundleAtlasFlow()

        # Check that final output was generated with default version
        with TemporaryDirectory() as out_dir:
            flow.run(in_dir, out_dir=out_dir)

            assert_equal(isfile(join(out_dir, "AF_L.trk")), True)
            assert_equal(isfile(join(out_dir, "CST_R.trk")), True)
            assert_equal(isfile(join(out_dir, "CC_ForcepsMajor.trk")), True)

        # Check additional outputs (whole brain tractogram and temporary files)
        with TemporaryDirectory() as out_dir:
            flow.run(in_dir, out_dir=out_dir, save_temp=True, merge_out=True)

            assert_equal(isfile(join(out_dir, "AF_L.trk")), True)
            assert_equal(isfile(join(out_dir, "CST_R.trk")), True)
            assert_equal(isfile(join(out_dir, "CC_ForcepsMajor.trk")), True)
            assert_equal(isfile(join(out_dir, "whole_brain.trk")), True)

            temp_files = listdir(join(out_dir, "temp", "AF_L", "step_0"))
            trk_files = [file for file in temp_files if file.endswith(".trk")]
            png_files = [file for file in temp_files if file.endswith(".png")]

            assert_equal(len(trk_files), 5)
            assert_equal(len(png_files), 5)
