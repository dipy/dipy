""" 
Test the registration workflows
""" 
import numpy as np
import nibabel as nib

from dipy.workflows import align
import dipy.align.vector_fields as vfu
from dipy.data import get_data
from dipy.align import floating
from dipy.align.imwarp import DiffeomorphicMap

def get_synthetic_warped_circle(nslices):
    #get a subsampled circle
    fname_cicle = get_data('reg_o')
    circle = np.load(fname_cicle)[::4,::4].astype(floating)

    #create a synthetic invertible map and warp the circle
    d, dinv = vfu.create_harmonic_fields_2d(64, 64, 0.1, 4)
    d = np.asarray(d, dtype=floating)
    dinv = np.asarray(dinv, dtype=floating)
    mapping = DiffeomorphicMap(2, (64, 64))
    mapping.forward, mapping.backward = d, dinv
    wcircle = mapping.transform(circle)

    if(nslices == 1):
        return circle, wcircle

    #normalize and form the 3d by piling slices
    circle = (circle-circle.min())/(circle.max() - circle.min())
    circle_3d = np.ndarray(circle.shape + (nslices,), dtype=floating)
    circle_3d[...] = circle[...,None]
    circle_3d[...,0] = 0
    circle_3d[...,-1] = 0

    #do the same with the warped circle
    wcircle = (wcircle-wcircle.min())/(wcircle.max() - wcircle.min())
    wcircle_3d = np.ndarray(wcircle.shape + (nslices,), dtype=floating)
    wcircle_3d[...] = wcircle[...,None]
    wcircle_3d[...,0] = 0
    wcircle_3d[...,-1] = 0
    return circle_3d, wcircle_3d


def test_syn_registration():
    """ 
    Test workflow for applying the SyN registration 
    """

    moving_data, static_data = get_synthetic_warped_circle(10)
    moving_affine = np.eye(4) 
    static_affine = np.eye(4)
    moving = nib.Nifti1Image(moving_data, moving_affine)
    static = nib.Nifti1Image(static_data, static_affine)
    align.syn_registration(moving, static)
