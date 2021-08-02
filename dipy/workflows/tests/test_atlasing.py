import zipfile
from os import listdir
from os.path import join, isfile
from dipy.utils.optpkg import optional_package
import pytest
from numpy.testing import assert_equal
from nibabel.tmpdirs import TemporaryDirectory
# from dipy.testing import assert_true
from dipy.data import get_fnames
from dipy.workflows.atlasing import DiscreteBundleAtlasFlow

_, has_fury, _ = optional_package('fury')


@pytest.mark.skipif(not has_fury, reason="Requires Fury")
def test_discrete_bundle_atlas():

    with TemporaryDirectory() as in_dir:
        example_tracts = get_fnames('minimal_bundles')
        with zipfile.ZipFile(example_tracts, 'r') as zip_ref:
            zip_ref.extractall(in_dir)

        with TemporaryDirectory() as out_dir:
            flow = DiscreteBundleAtlasFlow()
            flow.run(in_dir, out_dir=out_dir, save_temp=True, merge_out=True)

            # Check that final output was generated
            assert_equal(isfile(join(out_dir, 'AF_L.trk')), True)
            assert_equal(isfile(join(out_dir, 'CST_R.trk')), True)
            assert_equal(isfile(join(out_dir, 'CC_ForcepsMajor.trk')), True)
            assert_equal(isfile(join(out_dir, 'whole_brain.trk')), True)

            # Check that temp output was generated
            temp_files = listdir(join(out_dir, 'temp', 'AF_L', 'step_0'))
            trk_files = [file for file in temp_files if file.endswith('.trk')]
            png_files = [file for file in temp_files if file.endswith('.png')]

            assert_equal(len(trk_files), 5)
            assert_equal(len(png_files), 5)
