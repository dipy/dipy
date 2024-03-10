import logging
import os
from tempfile import mkstemp, TemporaryDirectory

import numpy as np
import numpy.testing as npt

import dipy.core.gradients as grad
from dipy.data import get_fnames
from dipy.data.fetcher import dipy_home
from dipy.io.image import load_nifti, save_nifti
from dipy.io.streamline import load_tractogram
from dipy.io.utils import nifti1_symmat
from dipy.testing import assert_true
from dipy.reconst import dti, utils as reconst_utils
from dipy.reconst.shm import convert_sh_descoteaux_tournier
from dipy.workflows.io import (IoInfoFlow, FetchFlow, SplitFlow,
                               ConcatenateTractogramFlow, ConvertSHFlow,
                               ConvertTractogramFlow, ConvertTensorsFlow)

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

    filepath_dix, _, _ = get_fnames('gold_standard_tracks')
    io_info_flow = IoInfoFlow()
    io_info_flow.run([filepath_dix['gs.trx'], filepath_dix['gs.trk']])

    io_info_flow = IoInfoFlow()
    npt.assert_raises(TypeError, io_info_flow.run, filepath_dix['gs.tck'])

    io_info_flow = IoInfoFlow()
    io_info_flow.run(filepath_dix['gs.tck'], reference=filepath_dix['gs.nii'])

    with open(fname_log, 'r') as file:
        lines = file.readlines()
        try:
            npt.assert_equal(lines[-3],
                             'INFO Total number of unit bvectors 25\n')
        except IndexError:  # logging maybe disabled in IDE setting
            pass


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
                     'synb0_weights', 'synb0_test', 'deepn4_weights',
                     'deepn4_test', 'bundle_warp_dataset',
                     'evac_weights', 'evac_test', 'ptt_minimal_dataset',
                     'stanford_tracks', 'cti_rat1']

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


def test_concatenate_flow():
    with TemporaryDirectory() as out_dir:
        concatenate_flow = ConcatenateTractogramFlow()
        data_path, _, _ = get_fnames('gold_standard_tracks')
        input_files = [v for k, v in data_path.items()
                       if k in ['gs.trk', 'gs.tck', 'gs.trx', 'gs.fib']
                       ]
        concatenate_flow.run(*input_files, out_dir=out_dir)
        assert_true(
            concatenate_flow.last_generated_outputs['out_extension'].endswith(
                'trx'))
        assert_true(os.path.isfile(
            concatenate_flow.last_generated_outputs['out_tractogram'] +
            ".trx"))

        trk = load_tractogram(
            concatenate_flow.last_generated_outputs['out_tractogram'] + ".trx",
            'same')
        npt.assert_equal(len(trk), 13)


def test_convert_sh_flow():
    with TemporaryDirectory() as out_dir:
        filepath_in = os.path.join(out_dir, 'sh_coeff_img.nii.gz')
        filename_out = 'sh_coeff_img_converted.nii.gz'
        filepath_out = os.path.join(out_dir, filename_out)

        # Create an input image
        dim0, dim1, dim2 = 2, 3, 3  # spatial dimensions of array
        num_sh_coeffs = 15  # 15 sh coeffs means l_max is 4
        img_in = np.arange(
            dim0*dim1*dim2*num_sh_coeffs, dtype=float
        ).reshape(dim0, dim1, dim2, num_sh_coeffs)
        save_nifti(filepath_in, img_in, np.eye(4))

        # Compute expected result to compare against later
        expected_img_out = convert_sh_descoteaux_tournier(img_in)

        # Run the workflow and load the output
        workflow = ConvertSHFlow()
        workflow.run(
            filepath_in,
            out_dir=out_dir,
            out_file=filename_out,
        )
        img_out, _ = load_nifti(filepath_out)

        # Compare
        npt.assert_array_almost_equal(img_out, expected_img_out)


def test_convert_tractogram_flow():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames('gold_standard_tracks')
        input_files = [v for k, v in data_path.items()
                       if k in ['gs.tck', 'gs.trx']]

        convert_tractogram_flow = ConvertTractogramFlow(mix_names=True)
        convert_tractogram_flow.run(input_files,
                                    reference=data_path['gs.nii'],
                                    out_dir=out_dir)

        convert_tractogram_flow._force_overwrite = True
        npt.assert_raises(ValueError, convert_tractogram_flow.run,
                          input_files, out_dir=out_dir)

        npt.assert_warns(UserWarning, convert_tractogram_flow.run,
                         data_path['gs.trx'], out_dir=out_dir,
                         out_tractogram='gs_converted.trx')


def test_convert_tensors_flow():
    with TemporaryDirectory() as out_dir:
        filepath_in = os.path.join(out_dir, 'tensors_img.nii.gz')
        filename_out = 'tensors_converted.nii.gz'
        filepath_out = os.path.join(out_dir, filename_out)

        # Create an input image
        fdata, fbval, fbvec = get_fnames('small_25')
        data, affine = load_nifti(fdata)
        gtab = grad.gradient_table(fbval, fbvec)
        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(data)

        tensor_vals = dti.lower_triangular(tenfit.quadratic_form)
        ten_img = nifti1_symmat(tensor_vals, affine=affine)

        save_nifti(filepath_in, ten_img.get_fdata().squeeze(), affine)

        # Compute expected result to compare against later
        expected_img_out = reconst_utils.convert_tensors(
            ten_img.get_fdata(), 'dipy', 'mrtrix')

        # Run the workflow and load the output
        workflow = ConvertTensorsFlow()
        workflow.run(
            filepath_in,
            from_format='dipy',
            to_format='mrtrix',
            out_dir=out_dir,
            out_tensor=filename_out,
        )

        img_out, _ = load_nifti(filepath_out)
        npt.assert_array_almost_equal(img_out, expected_img_out)
