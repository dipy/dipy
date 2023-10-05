from os.path import join
from tempfile import TemporaryDirectory

import numpy as np
from numpy.testing import assert_equal

from dipy.data import get_fnames
from dipy.io.image import load_nifti_data, load_nifti, save_nifti
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
        kwargs = dict(out_dir=out_dir)
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
