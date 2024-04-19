from os.path import join
from tempfile import TemporaryDirectory

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from dipy.data import get_fnames
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.io.peaks import load_peaks
from dipy.reconst.shm import sph_harm_ind_list
from dipy.workflows.reconst import ReconstDtiFlow


def test_reconst_dti_wls():
    reconst_flow_core(ReconstDtiFlow)


def test_reconst_dti_nlls():
    reconst_flow_core(ReconstDtiFlow, extra_args=[], extra_kwargs={})


def test_reconst_dti_alt_tensor():
    reconst_flow_core(ReconstDtiFlow, extra_args=[],
                      extra_kwargs={'nifti_tensor': False})


def reconst_flow_core(flow, extra_args=None, extra_kwargs=None):

    extra_args = extra_args or []
    extra_kwargs = extra_kwargs or {}

    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames('small_25')
        volume, affine = load_nifti(data_path)
        mask = np.ones_like(volume[:, :, :, 0], dtype=np.uint8)
        mask_path = join(out_dir, 'tmp_mask.nii.gz')
        save_nifti(mask_path, mask, affine)

        dti_flow = flow()

        args = [data_path, bval_path, bvec_path, mask_path]
        args.extend(extra_args)
        kwargs = dict(out_dir=out_dir, extract_pam_values=True)
        kwargs.update(extra_kwargs)

        dti_flow.run(*args, **kwargs)

        fa_path = dti_flow.last_generated_outputs['out_fa']
        fa_data = load_nifti_data(fa_path)
        assert_equal(fa_data.shape, volume.shape[:-1])

        tensor_path = dti_flow.last_generated_outputs['out_tensor']
        tensor_data = load_nifti_data(tensor_path)
        # Per default, tensor data is 5D, with six tensor elements on the last
        # dimension, except if nifti_tensor is set to False:
        if extra_kwargs.get('nifti_tensor', True):
            assert_equal(tensor_data.shape[-1], 6)
            assert_equal(tensor_data.shape[:-2], volume.shape[:-1])
        else:
            assert_equal(tensor_data.shape[-1], 6)
            assert_equal(tensor_data.shape[:-1], volume.shape[:-1])

        for out_name in ['out_ga', 'out_md', 'out_ad', 'out_rd', 'out_mode']:
            out_path = dti_flow.last_generated_outputs[out_name]
            out_data = load_nifti_data(out_path)
            assert_equal(out_data.shape, volume.shape[:-1])

        rgb_path = dti_flow.last_generated_outputs['out_rgb']
        rgb_data = load_nifti_data(rgb_path)
        assert_equal(rgb_data.shape[-1], 3)
        assert_equal(rgb_data.shape[:-1], volume.shape[:-1])

        evecs_path = dti_flow.last_generated_outputs['out_evec']
        evecs_data = load_nifti_data(evecs_path)
        assert_equal(evecs_data.shape[-2:], (3, 3))
        assert_equal(evecs_data.shape[:-2], volume.shape[:-1])

        evals_path = dti_flow.last_generated_outputs['out_eval']
        evals_data = load_nifti_data(evals_path)
        assert_equal(evals_data.shape[-1], 3)
        assert_equal(evals_data.shape[:-1], volume.shape[:-1])

        gfa_path = dti_flow.last_generated_outputs['out_gfa']
        gfa_data = load_nifti_data(gfa_path)
        assert_equal(gfa_data.shape, volume.shape[:-1])

        peaks_dir_path = dti_flow.last_generated_outputs['out_peaks_dir']
        peaks_dir_data = load_nifti_data(peaks_dir_path)
        assert_equal(peaks_dir_data.shape[-1], 15)
        assert_equal(peaks_dir_data.shape[:-1], volume.shape[:-1])

        peaks_idx_path = dti_flow.last_generated_outputs['out_peaks_indices']
        peaks_idx_data = load_nifti_data(peaks_idx_path)
        assert_equal(peaks_idx_data.shape[-1], 5)
        assert_equal(peaks_idx_data.shape[:-1], volume.shape[:-1])

        peaks_vals_path = dti_flow.last_generated_outputs['out_peaks_values']
        peaks_vals_data = load_nifti_data(peaks_vals_path)
        assert_equal(peaks_vals_data.shape[-1], 5)
        assert_equal(peaks_vals_data.shape[:-1], volume.shape[:-1])

        shm_path = dti_flow.last_generated_outputs['out_shm']
        shm_data = load_nifti_data(shm_path)
        # Test that the number of coefficients is what you would expect
        # given the order of the sh basis:
        sh_order = 8
        assert_equal(shm_data.shape[-1],
                     sph_harm_ind_list(sh_order)[0].shape[0])
        assert_equal(shm_data.shape[:-1], volume.shape[:-1])

        pam = load_peaks(dti_flow.last_generated_outputs['out_pam'])
        assert_allclose(pam.peak_dirs.reshape(peaks_dir_data.shape),
                        peaks_dir_data)
        assert_allclose(pam.peak_values, peaks_vals_data)
        assert_allclose(pam.peak_indices, peaks_idx_data)
        assert_allclose(pam.shm_coeff, shm_data)
        assert_allclose(pam.gfa, gfa_data)
