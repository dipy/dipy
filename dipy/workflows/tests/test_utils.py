import numpy.testing as npt
from os.path import join
from dipy.workflows.utils import choose_create_out_dir
from nibabel.tmpdirs import InTemporaryDirectory

def test_choose_create_out_dir():
    with InTemporaryDirectory() as tmp_dir:
        root_path = join(tmp_dir, 'fake_file.nii')
        result_path = choose_create_out_dir('', root_path)
        npt.assert_equal(result_path, tmp_dir)

        rel_out_dir = 'test_dir'
        result_path = choose_create_out_dir(rel_out_dir, root_path)
        npt.assert_equal(result_path, join(tmp_dir, rel_out_dir))

        result_path = choose_create_out_dir(tmp_dir, '')
        npt.assert_equal(result_path, tmp_dir)
