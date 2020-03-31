import logging
import os
import numpy.testing as npt
from dipy.data import get_fnames
from dipy.io.image import load_nifti
from dipy.testing import assert_true
from dipy.data.fetcher import dipy_home
from dipy.workflows.io import IoInfoFlow, FetchFlow, SplitFlow
from nibabel.tmpdirs import TemporaryDirectory
from os.path import join as pjoin
from tempfile import mkstemp
fname_log = mkstemp()[1]

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(message)s',
                    filename=fname_log,
                    filemode='w')


def test_io_info():
    fimg, fbvals, fbvecs = get_fnames('small_101D')
    io_info_flow = IoInfoFlow()
    io_info_flow.run([fimg, fbvals, fbvecs])

    fimg, fbvals, fvecs = get_fnames('small_25')
    io_info_flow = IoInfoFlow()
    io_info_flow.run([fimg, fbvals, fvecs])

    io_info_flow = IoInfoFlow()
    io_info_flow.run([fimg, fbvals, fvecs], b0_threshold=20, bvecs_tol=0.001)

    file = open(fname_log, 'r')
    lines = file.readlines()
    try:
        npt.assert_equal(lines[-3], 'INFO Total number of unit bvectors 25\n')
    except IndexError:  # logging maybe disabled in IDE setting
        pass
    file.close()


def test_io_fetch():
    fetch_flow = FetchFlow()
    with TemporaryDirectory() as out_dir:

        fetch_flow.run(['bundle_fa_hcp'])
        npt.assert_equal(os.path.isdir(os.path.join(dipy_home,
                                                    'bundle_fa_hcp')),
                         True)

        fetch_flow.run(['bundle_fa_hcp'], out_dir=out_dir)
        npt.assert_equal(os.path.isdir(os.path.join(out_dir,
                                                    'bundle_fa_hcp')),
                         True)


def test_io_fetch_fetcher_datanames():
    available_data = FetchFlow.get_fetcher_datanames()

    dataset_names = ['bundle_atlas_hcp842', 'bundle_fa_hcp',
                     'bundles_2_subjects', 'cenir_multib', 'cfin_multib',
                     'file_formats', 'gold_standard_io', 'isbi2013_2shell',
                     'ivim', 'mni_template', 'qtdMRI_test_retest_2subjects',
                     'scil_b0', 'sherbrooke_3shell', 'stanford_hardi',
                     'stanford_labels', 'stanford_pve_maps', 'stanford_t1',
                     'syn_data', 'taiwan_ntu_dsi', 'target_tractogram_hcp',
                     'tissue_data']

    num_expected_fetch_methods = len(dataset_names)
    npt.assert_equal(len(available_data), num_expected_fetch_methods)
    npt.assert_equal(all(dataset_name in available_data.keys()
                         for dataset_name in dataset_names), True)


def test_split_flow():
    with TemporaryDirectory() as out_dir:
        split_flow = SplitFlow()
        data_path, _, _ = get_fnames()
        volume, affine = load_nifti(data_path)
        split_flow.run(data_path, out_dir=out_dir)
        assert_true(os.path.isfile(
         split_flow.last_generated_outputs['out_split']))
        split_flow._force_overwrite = True
        split_flow.run(data_path, vol_idx=0, out_dir=out_dir)
        split_path = split_flow.last_generated_outputs['out_split']
        assert_true(os.path.isfile(split_path))
        split_data, split_affine = load_nifti(split_path)
        npt.assert_equal(split_data.shape, volume[..., 0].shape)
        npt.assert_array_almost_equal(split_affine, affine)


if __name__ == '__main__':
    test_io_fetch()
    test_io_fetch_fetcher_datanames()
    test_io_info()
    test_split_flow()
