from os.path import join

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

import numpy as np
from numpy.testing import assert_equal

from dipy.data import get_fnames
from dipy.workflows.reconst import ReconstDtiFlow


def test_reconst_dti_wls():
    reconst_flow_core(ReconstDtiFlow)


def test_reconst_dti_nlls():
    reconst_flow_core(ReconstDtiFlow, extra_args=[], extra_kwargs={})


def test_reconst_dti_alt_tensor():
    reconst_flow_core(ReconstDtiFlow, extra_args=[],
                      extra_kwargs={'nifti_tensor': False})


def reconst_flow_core(flow, extra_args=[], extra_kwargs={}):
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames('small_25')
        vol_img = nib.load(data_path)
        volume = vol_img.get_data()
        mask = np.ones_like(volume[:, :, :, 0])
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), vol_img.affine)
        mask_path = join(out_dir, 'tmp_mask.nii.gz')
        nib.save(mask_img, mask_path)

        dti_flow = flow()

        args = [data_path, bval_path, bvec_path, mask_path]
        args.extend(extra_args)
        kwargs = dict(out_dir=out_dir)
        kwargs.update(extra_kwargs)

        dti_flow.run(*args, **kwargs)

        fa_path = dti_flow.last_generated_outputs['out_fa']
        fa_data = nib.load(fa_path).get_data()
        assert_equal(fa_data.shape, volume.shape[:-1])

        tensor_path = dti_flow.last_generated_outputs['out_tensor']
        tensor_data = nib.load(tensor_path)
        # Per default, tensor data is 5D, with six tensor elements on the last
        # dimension, except if nifti_tensor is set to False:
        if extra_kwargs.get('nifti_tensor', True):
            assert_equal(tensor_data.shape[-1], 6)
            assert_equal(tensor_data.shape[:-2], volume.shape[:-1])
        else:
            assert_equal(tensor_data.shape[-1], 6)
            assert_equal(tensor_data.shape[:-1], volume.shape[:-1])

        ga_path = dti_flow.last_generated_outputs['out_ga']
        ga_data = nib.load(ga_path).get_data()
        assert_equal(ga_data.shape, volume.shape[:-1])

        rgb_path = dti_flow.last_generated_outputs['out_rgb']
        rgb_data = nib.load(rgb_path)
        assert_equal(rgb_data.shape[-1], 3)
        assert_equal(rgb_data.shape[:-1], volume.shape[:-1])

        md_path = dti_flow.last_generated_outputs['out_md']
        md_data = nib.load(md_path).get_data()
        assert_equal(md_data.shape, volume.shape[:-1])

        ad_path = dti_flow.last_generated_outputs['out_ad']
        ad_data = nib.load(ad_path).get_data()
        assert_equal(ad_data.shape, volume.shape[:-1])

        rd_path = dti_flow.last_generated_outputs['out_rd']
        rd_data = nib.load(rd_path).get_data()
        assert_equal(rd_data.shape, volume.shape[:-1])

        mode_path = dti_flow.last_generated_outputs['out_mode']
        mode_data = nib.load(mode_path).get_data()
        assert_equal(mode_data.shape, volume.shape[:-1])

        evecs_path = dti_flow.last_generated_outputs['out_evec']
        evecs_data = nib.load(evecs_path).get_data()
        assert_equal(evecs_data.shape[-2:], tuple((3, 3)))
        assert_equal(evecs_data.shape[:-2], volume.shape[:-1])

        evals_path = dti_flow.last_generated_outputs['out_eval']
        evals_data = nib.load(evals_path).get_data()
        assert_equal(evals_data.shape[-1], 3)
        assert_equal(evals_data.shape[:-1], volume.shape[:-1])



if __name__ == '__main__':
    test_reconst_dti_nlls()
