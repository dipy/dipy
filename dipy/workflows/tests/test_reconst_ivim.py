from os.path import join as pjoin

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

import numpy as np

import numpy.testing as npt
from numpy.testing import assert_equal

from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import generate_bvecs
from dipy.workflows.reconst import ReconstIvimFlow

def test_reconst_ivim():
    
    with TemporaryDirectory() as out_dir:        
        data_path, bval_path, bvec_path = get_fnames('ivim_small')
        vol_img = nib.load(data_path)
        volume = vol_img.get_data()
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        bvals[0] = 0
        tmp0_bval_path = pjoin(out_dir, "tmp0.bval")
        np.savetxt(tmp0_bval_path, bvals)
        
        ivim_flow = ReconstIvimFlow()

        args = [data_path, tmp0_bval_path, bvec_path]        

        ivim_flow.run(*args, out_dir=out_dir)

        S0_path = ivim_flow.last_generated_outputs['out_S0_est']
        S0_data = nib.load(S0_path).get_data()
        assert_equal(S0_data.shape, volume.shape[:-1])
        
        f_path = ivim_flow.last_generated_outputs['out_f_est']
        f_data = nib.load(f_path).get_data()
        assert_equal(f_data.shape, volume.shape[:-1])
  
        D_star_path = ivim_flow.last_generated_outputs['out_D_star_est']
        D_star_data = nib.load(D_star_path).get_data()
        assert_equal(D_star_data.shape, volume.shape[:-1])

        D_path = ivim_flow.last_generated_outputs['out_D_est']
        D_data = nib.load(D_path).get_data()
        assert_equal(D_data.shape, volume.shape[:-1])

        tmp_bval_path = pjoin(out_dir, "tmp.bval")
        tmp_bvec_path = pjoin(out_dir, "tmp.bvec")
        
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        bvals[0] = 0.
        bvecs = generate_bvecs(len(bvals))
        
        np.savetxt(tmp_bval_path, bvals)
        np.savetxt(tmp_bvec_path, bvecs.T)

        ivim_flow._force_overwrite = True
        npt.assert_warns(UserWarning, ivim_flow.run, data_path, tmp_bval_path,
                         tmp_bvec_path, out_dir=out_dir)


if __name__ == '__main__':
    test_reconst_ivim()

