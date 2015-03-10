import numpy as np

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_almost_equal)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_)

from dipy.sims.voxel import (SingleTensor, MultiTensor, multi_tensor_odf, all_tensor_evecs,
                             add_noise, single_tensor, sticks_and_ball)
from dipy.core.geometry import vec2vec_rotmat
from dipy.data import get_data, get_sphere
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs


fimg, fbvals, fbvecs = get_data('small_64D')
bvals = np.load(fbvals)
bvecs = np.load(fbvecs)
gtab = gradient_table(bvals, bvecs)


def diff2eigenvectors(dx, dy, dz):
    """ numerical derivatives 2 eigenvectors
    """
    u = np.array([dx, dy, dz])
    u = u / np.linalg.norm(u)
    R = vec2vec_rotmat(basis[:, 0], u)
    eig0 = u
    eig1 = np.dot(R, basis[:, 1])
    eig2 = np.dot(R, basis[:, 2])
    eigs = np.zeros((3, 3))
    eigs[:, 0] = eig0
    eigs[:, 1] = eig1
    eigs[:, 2] = eig2
    return eigs, R


def test_sticks_and_ball():
    d = 0.0015
    S, sticks = sticks_and_ball(gtab, d=d, S0=1, angles=[(0, 0), ],
                                fractions=[100], snr=None)
    assert_array_equal(sticks, [[0, 0, 1]])
    S_st = SingleTensor(gtab, 1, evals=[d, 0, 0], evecs=[[0, 0, 0],
                                                         [0, 0, 0],
                                                         [1, 0, 0]])
    assert_array_almost_equal(S, S_st)


def test_single_tensor():
    evals = np.array([1.4, .35, .35]) * 10 ** (-3)
    evecs = np.eye(3)
    S = SingleTensor(gtab, 100, evals, evecs, snr=None)
    assert_array_almost_equal(S[gtab.b0s_mask], 100)
    assert_(np.mean(S[~gtab.b0s_mask]) < 100)

    from dipy.reconst.dti import TensorModel
    m = TensorModel(gtab)
    t = m.fit(S)

    assert_array_almost_equal(t.fa, 0.707, decimal=3)


def test_multi_tensor():
    sphere = get_sphere('symmetric724')
    vertices = sphere.vertices
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))
    e0 = np.array([np.sqrt(2) / 2., np.sqrt(2) / 2., 0])
    e1 = np.array([0, np.sqrt(2) / 2., np.sqrt(2) / 2.])
    mevecs = [all_tensor_evecs(e0), all_tensor_evecs(e1)]
    # odf = multi_tensor_odf(vertices, [0.5, 0.5], mevals, mevecs)

    # assert_(odf.shape == (len(vertices),))
    # assert_(np.all(odf <= 1) & np.all(odf >= 0))

    fimg, fbvals, fbvecs = get_data('small_101D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)

    s1 = single_tensor(gtab, 100, mevals[0], mevecs[0].T, snr=None)
    s2 = single_tensor(gtab, 100, mevals[1], mevecs[1].T, snr=None)

    Ssingle = 0.5 * s1 + 0.5 * s2

    S, sticks = MultiTensor(gtab, mevals, S0=100, angles=[(90, 45), (45, 90)],
                            fractions=[50, 50], snr=None)

    assert_array_almost_equal(S, Ssingle)


def test_snr():
    np.random.seed(1978)

    s = single_tensor(gtab)

    # For reasonably large SNR, var(signal) ~= sigma**2, where sigma = 1/SNR
    for snr in [5, 10, 20]:
        sigma = 1.0 / snr
        for j in range(1000):
            s_noise = add_noise(s, snr, 1, noise_type='rician')

        assert_array_almost_equal(np.var(s_noise - s), sigma ** 2, decimal=2)


def test_all_tensor_evecs():
    e0 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])

    desired = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0],
                        [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                        [0, 0, 1]])

    assert_array_almost_equal(all_tensor_evecs(e0), desired)


if __name__ == "__main__":

    test_multi_tensor()
