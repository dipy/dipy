from os.path import join as pjoin

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

import numpy as np

from nose.tools import assert_true, assert_equal
import numpy.testing as npt

from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import generate_bvecs
from dipy.workflows.reconst import ReconstDkiFlow


def test_reconst_dki():
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames('small_101D')
        vol_img = nib.load(data_path)
        volume = vol_img.get_data()
        mask = np.ones_like(volume[:, :, :, 0])
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), vol_img.affine)
        mask_path = pjoin(out_dir, 'tmp_mask.nii.gz')
        nib.save(mask_img, mask_path)

        dki_flow = ReconstDkiFlow()

        args = [data_path, bval_path, bvec_path, mask_path]

        dki_flow.run(*args, out_dir=out_dir)

        fa_path = dki_flow.last_generated_outputs['out_fa']
        fa_data = nib.load(fa_path).get_data()
        assert_equal(fa_data.shape, volume.shape[:-1])

        tensor_path = dki_flow.last_generated_outputs['out_dt_tensor']
        tensor_data = nib.load(tensor_path)
        assert_equal(tensor_data.shape[-1], 6)
        assert_equal(tensor_data.shape[:-1], volume.shape[:-1])

        ga_path = dki_flow.last_generated_outputs['out_ga']
        ga_data = nib.load(ga_path).get_data()
        assert_equal(ga_data.shape, volume.shape[:-1])

        rgb_path = dki_flow.last_generated_outputs['out_rgb']
        rgb_data = nib.load(rgb_path)
        assert_equal(rgb_data.shape[-1], 3)
        assert_equal(rgb_data.shape[:-1], volume.shape[:-1])

        md_path = dki_flow.last_generated_outputs['out_md']
        md_data = nib.load(md_path).get_data()
        assert_equal(md_data.shape, volume.shape[:-1])

        ad_path = dki_flow.last_generated_outputs['out_ad']
        ad_data = nib.load(ad_path).get_data()
        assert_equal(ad_data.shape, volume.shape[:-1])

        rd_path = dki_flow.last_generated_outputs['out_rd']
        rd_data = nib.load(rd_path).get_data()
        assert_equal(rd_data.shape, volume.shape[:-1])

        mk_path = dki_flow.last_generated_outputs['out_mk']
        mk_data = nib.load(mk_path).get_data()
        assert_equal(mk_data.shape, volume.shape[:-1])

        ak_path = dki_flow.last_generated_outputs['out_ak']
        ak_data = nib.load(ak_path).get_data()
        assert_equal(ak_data.shape, volume.shape[:-1])

        rk_path = dki_flow.last_generated_outputs['out_rk']
        rk_data = nib.load(rk_path).get_data()
        assert_equal(rk_data.shape, volume.shape[:-1])

        kt_path = dki_flow.last_generated_outputs['out_dk_tensor']
        kt_data = nib.load(kt_path)
        assert_equal(kt_data.shape[-1], 15)
        assert_equal(kt_data.shape[:-1], volume.shape[:-1])

        mode_path = dki_flow.last_generated_outputs['out_mode']
        mode_data = nib.load(mode_path).get_data()
        assert_equal(mode_data.shape, volume.shape[:-1])

        evecs_path = dki_flow.last_generated_outputs['out_evec']
        evecs_data = nib.load(evecs_path).get_data()
        assert_equal(evecs_data.shape[-2:], tuple((3, 3)))
        assert_equal(evecs_data.shape[:-2], volume.shape[:-1])

        evals_path = dki_flow.last_generated_outputs['out_eval']
        evals_data = nib.load(evals_path).get_data()
        assert_equal(evals_data.shape[-1], 3)
        assert_equal(evals_data.shape[:-1], volume.shape[:-1])

        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        bvals[0] = 5.
        bvecs = generate_bvecs(len(bvals))

        tmp_bval_path = pjoin(out_dir, "tmp.bval")
        tmp_bvec_path = pjoin(out_dir, "tmp.bvec")
        np.savetxt(tmp_bval_path, bvals)
        np.savetxt(tmp_bvec_path, bvecs.T)
        dki_flow._force_overwrite = True
        npt.assert_warns(UserWarning, dki_flow.run, data_path,
                         tmp_bval_path, tmp_bvec_path, mask_path,
                         out_dir=out_dir, b0_threshold=0)


if __name__ == '__main__':
    test_reconst_dki()
