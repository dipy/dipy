from os.path import join as pjoin

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

import numpy as np

from numpy.testing import assert_equal

from dipy.sims.voxel import multi_tensor
from dipy.core.gradients import generate_bvecs, gradient_table
from dipy.workflows.reconst import ReconstIvimFlow


def test_reconst_ivim():

    with TemporaryDirectory() as out_dir:
        bvals = np.array([0., 10., 20., 30., 40., 60., 80., 100.,
                          120., 140., 160., 180., 200., 300., 400.,
                          500., 600., 700., 800., 900., 1000.])
        N = len(bvals)
        bvecs = generate_bvecs(N)
        temp_bval_path = pjoin(out_dir, "temp.bval")
        np.savetxt(temp_bval_path, bvals)
        temp_bvec_path = pjoin(out_dir, "temp.bvec")
        np.savetxt(temp_bvec_path, bvecs)

        gtab = gradient_table(bvals, bvecs)

        S0, f, D_star, D = 1000.0, 0.132, 0.00885, 0.000921

        mevals = np.array(([D_star, D_star, D_star], [D, D, D]))
        # This gives an isotropic signal.
        data = multi_tensor(gtab, mevals, snr=None, S0=S0,
                            fractions=[f * 100, 100 * (1 - f)])
        # Single voxel data
        data_single = data[0]
        temp_affine = np.eye(4)

        data_multi = np.zeros((2, 2, 1, len(gtab.bvals)))
        data_multi[0, 0, 0] = data_multi[0, 1, 0] = data_multi[
            1, 0, 0] = data_multi[1, 1, 0] = data_single
        data_img = nib.Nifti1Image(data_multi.astype(int), temp_affine)
        data_path = pjoin(out_dir, 'tmp_data.nii.gz')
        nib.save(data_img, data_path)

        mask = np.ones_like(data_multi[..., 0])
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), data_img.affine)
        mask_path = pjoin(out_dir, 'tmp_mask.nii.gz')
        nib.save(mask_img, mask_path)

        ivim_flow = ReconstIvimFlow()

        args = [data_path, temp_bval_path, temp_bvec_path, mask_path]

        ivim_flow.run(*args, out_dir=out_dir)

        S0_path = ivim_flow.last_generated_outputs['out_S0_predicted']
        S0_data = nib.load(S0_path).get_data()
        assert_equal(S0_data.shape, data_img.shape[:-1])

        f_path = ivim_flow.last_generated_outputs['out_perfusion_fraction']
        f_data = nib.load(f_path).get_data()
        assert_equal(f_data.shape, data_img.shape[:-1])

        D_star_path = ivim_flow.last_generated_outputs['out_D_star']
        D_star_data = nib.load(D_star_path).get_data()
        assert_equal(D_star_data.shape, data_img.shape[:-1])

        D_path = ivim_flow.last_generated_outputs['out_D']
        D_data = nib.load(D_path).get_data()
        assert_equal(D_data.shape, data_img.shape[:-1])


if __name__ == '__main__':
    test_reconst_ivim()
