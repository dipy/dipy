""" 
Test the registration workflows
""" 
import numpy as np
import numpy.testing as npt
import nibabel as nib


from dipy.align import api
import dipy.align.vector_fields as vfu
from dipy.align import floating
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.imwarp import get_synthetic_warped_circle

def test_syn_registration():
    """ 
    Test workflow for applying the SyN registration 
    """
    moving_data, static_data = get_synthetic_warped_circle(10)
    moving_affine = np.eye(4) 
    static_affine = np.eye(4)
    moving = nib.Nifti1Image(moving_data, moving_affine)
    static = nib.Nifti1Image(static_data, static_affine)
    new_image, forward, back = api.syn_registration(moving, static)
    d, dinv = vfu.create_harmonic_fields_2d(64, 64, 0.1, 4)
    npt.assert_array_almost_equal(forward.get_data()[:,:, 1], d)