import numpy as np
import nose
import nibabel as nib
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal
from dipy.sims.voxel import SingleTensor, multi_tensor_odf, all_tensor_evecs
from dipy.core.geometry import vec2vec_rotmat
from dipy.data import get_data, get_sphere
from dipy.viz import fvtk

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
    #bvals=np.loadtxt(fbvals)
    #bvecs=np.loadtxt(fbvecs).T
    img=nib.load(fimg)
    data=img.get_data()

    evals=np.array([1.4,.35,.35])*10**(-3)
    evecs=np.eye(3)
    S=SingleTensor(bvals,bvecs,100,evals,evecs,snr=None)
    """
    colours=fvtk.colors(S,'jet')
    r=fvtk.ren()
    fvtk.add(r,fvtk.point(bvecs,colours))
    fvtk.show(r)
    """


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


if __name__ == "__main__":

    test_single_tensor()
