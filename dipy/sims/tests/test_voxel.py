import numpy as np
import numpy.testing.decorators as dec

import nose
import nibabel as nib
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal
from dipy.sims.voxel import (SingleTensor, multi_tensor_odf, all_tensor_evecs,
                             add_noise, single_tensor)
from dipy.core.geometry import vec2vec_rotmat
from dipy.data import get_data, get_sphere
from dipy.core.gradients import gradient_table

def diff2eigenvectors(dx,dy,dz):
    """ numerical derivatives 2 eigenvectors
    """
    u=np.array([dx,dy,dz])
    u=u/np.linalg.norm(u)
    R=vec2vec_rotmat(basis[:,0],u)
    eig0=u
    eig1=np.dot(R,basis[:,1])
    eig2=np.dot(R,basis[:,2])
    eigs=np.zeros((3,3))
    eigs[:,0]=eig0
    eigs[:,1]=eig1
    eigs[:,2]=eig2
    return eigs, R


def test_single_tensor():

    fimg,fbvals,fbvecs=get_data('small_64D')
    bvals=np.load(fbvals)
    bvecs=np.load(fbvecs)
    gtab = gradient_table(bvals, bvecs)
    #bvals=np.loadtxt(fbvals)
    #bvecs=np.loadtxt(fbvecs).T
    img=nib.load(fimg)
    data=img.get_data()

    evals=np.array([1.4,.35,.35])*10**(-3)
    evecs=np.eye(3)
    S=SingleTensor(gtab, 100,evals,evecs,snr=None)


def test_multi_tensor():
    sphere = get_sphere('symmetric724')
    vertices, faces = sphere.vertices, sphere.faces
    mevals=np.array(([0.0015, 0.0003, 0.0003],
                     [0.0015, 0.0003, 0.0003]))
    e0 = np.array([1, 0, 0.])
    e1 = np.array([0., 1, 0])
    mevecs=[all_tensor_evecs(e0), all_tensor_evecs(e1)]
    odf = multi_tensor_odf(vertices, [0.5,0.5], mevals, mevecs)

    assert odf.shape == (len(vertices),)
    assert np.all(odf <= 1) & np.all(odf >= 0)


@dec.slow
def test_snr():
    """
    Test the addition of noise with specific SNR
    """
    
    fimg,fbvals,fbvecs=get_data('small_64D')
    bvals=np.load(fbvals)
    bvecs=np.load(fbvecs)
    bvecs[np.isnan(bvecs)]=0
    gtab = gradient_table(bvals, bvecs)
    
    s1 = single_tensor(gtab)

    # For reasonably large SNR, var(signal) ~= sigma**2, where sigma = 1/SNR
    for snr in [5, 10, 20]:
        sigma = 1.0/snr
        s0 = []
        for j in range(1000):
            s2 = add_noise(s1, snr=snr, noise_type='rician')
            s0.append(s2[0])

        assert_array_almost_equal(np.var(s0), sigma**2, decimal=2)



if __name__ == "__main__":

    test_single_tensor()
