import logging
import os
from tempfile import mkstemp, TemporaryDirectory

import numpy.testing as npt
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, default_sphere
from dipy.direction.peaks import PeaksAndMetrics
from dipy.io.image import load_nifti, save_nifti
from dipy.io.peaks import save_pam, load_pam
from dipy.testing import assert_true
from dipy.data.fetcher import dipy_home
from dipy.reconst.dti import TensorModel
from dipy.workflows.io import (IoInfoFlow, FetchFlow, SplitFlow,
                               PamToNiftisFlow, NiftisToPamFlow,
                               TensorToPamFlow)

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
                     'file_formats', 'fury_surface',
                     'gold_standard_io', 'isbi2013_2shell',
                     'ivim', 'mni_template', 'qtdMRI_test_retest_2subjects',
                     'scil_b0', 'sherbrooke_3shell', 'stanford_hardi',
                     'stanford_labels', 'stanford_pve_maps', 'stanford_t1',
                     'syn_data', 'taiwan_ntu_dsi', 'target_tractogram_hcp',
                     'tissue_data', 'qte_lte_pte', 'resdnn_weights',
                     'DiB_217_lte_pte_ste', 'DiB_70_lte_pte_ste',
                     'synb0_weights', 'synb0_test', 'bundle_warp_dataset',
                     'evac_weights', 'evac_test', 'ptt_minimal_dataset',
                     'stanford_tracks']

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


def generate_random_pam():
    pam = PeaksAndMetrics()
    pam.affine = np.eye(4)
    pam.peak_dirs = np.random.rand(15, 15, 15, 5, 3)
    pam.peak_values = np.zeros((15, 15, 15, 5))
    pam.peak_indices = np.zeros((15, 15, 15, 5))
    pam.shm_coeff = np.zeros((15, 15, 15, 45))
    pam.sphere = default_sphere
    pam.B = np.zeros((45, default_sphere.vertices.shape[0]))
    pam.total_weight = 0.5
    pam.ang_thr = 60
    pam.gfa = np.zeros((10, 10, 10))
    pam.qa = np.zeros((10, 10, 10, 5))
    pam.odf = np.zeros((10, 10, 10, default_sphere.vertices.shape[0]))
    return pam


def test_niftis_to_pam_flow():
    pam = generate_random_pam()
    with TemporaryDirectory() as out_dir:
        fname = 'test.pam5'
        save_pam(fname, pam)

        args = [fname, out_dir]
        flow = PamToNiftisFlow()
        flow.run(*args)

        args = [flow.last_generated_outputs['out_peaks_dir'],
                flow.last_generated_outputs['out_peaks_values'],
                flow.last_generated_outputs['out_peaks_indices'],
                ]

        flow2 = NiftisToPamFlow()
        flow2.run(*args, out_dir=out_dir)
        pam_file = flow2.last_generated_outputs['out_pam']
        assert_true(os.path.isfile(pam_file))

        res_pam = load_pam(pam_file)
        npt.assert_array_equal(pam.affine, res_pam.affine)
        npt.assert_array_almost_equal(pam.peak_dirs, res_pam.peak_dirs)
        npt.assert_array_almost_equal(pam.peak_values, res_pam.peak_values)
        npt.assert_array_almost_equal(pam.peak_indices, res_pam.peak_indices)


def test_tensor_to_pam_flow():
    fdata, fbval, fbvec = get_fnames('small_25')
    gtab = gradient_table(fbval, fbvec)
    data, affine = load_nifti(fdata)
    dm = TensorModel(gtab)
    df = dm.fit(data)
    df.evals[0, 0, 0] = np.array([0, 0, 0])

    with TemporaryDirectory() as out_dir:
        f_mevals, f_mevecs = 'evals.nii.gz', 'evecs.nii.gz'
        save_nifti(f_mevals, df.evals, affine)
        save_nifti(f_mevecs, df.evecs, affine)

        args = [f_mevals, f_mevecs]
        flow = TensorToPamFlow()
        flow.run(*args, out_dir=out_dir)
        pam_file = flow.last_generated_outputs['out_pam']
        assert_true(os.path.isfile(pam_file))

        pam = load_pam(pam_file)
        npt.assert_array_equal(pam.affine, affine)
        npt.assert_array_almost_equal(pam.peak_dirs[..., :3, :], df.evecs)
        npt.assert_array_almost_equal(pam.peak_values[..., :3], df.evals)


def test_pam_to_niftis_flow():
    pam = generate_random_pam()

    with TemporaryDirectory():
        fname = 'test.pam5'
        save_pam(fname, pam)

        args = [fname, ]
        flow = PamToNiftisFlow()
        flow.run(*args)
        assert_true(
            os.path.isfile(flow.last_generated_outputs['out_peaks_dir']))
        assert_true(
            os.path.isfile(flow.last_generated_outputs['out_peaks_values']))
        assert_true(
            os.path.isfile(flow.last_generated_outputs['out_peaks_indices']))
        assert_true(
            os.path.isfile(flow.last_generated_outputs['out_shm']))
        assert_true(
            os.path.isfile(flow.last_generated_outputs['out_gfa']))
        assert_true(
            os.path.isfile(flow.last_generated_outputs['out_sphere']))
