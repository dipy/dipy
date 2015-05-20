import numpy as np

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_almost_equal)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_)

from dipy.sims.voxel import (_check_directions, SingleTensor, MultiTensor,
                             multi_tensor_odf, all_tensor_evecs, add_noise,
                             single_tensor, sticks_and_ball, multi_tensor_dki,
                             dki_design_matrix, compute_Wijkl)
from dipy.core.geometry import (vec2vec_rotmat, sphere2cart)
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


def test_check_directions():
    # Testing spherical angles for two principal coordinate axis
    angles = [(0, 0)]  # axis z
    sticks = _check_directions(angles)
    assert_array_almost_equal(sticks, [[0, 0, 1]])
    angles = [(0, 90)]  # axis z again (phi can be anything it theta is zero)
    sticks = _check_directions(angles)
    assert_array_almost_equal(sticks, [[0, 0, 1]])
    angles = [(90, 0)]  # axis x
    sticks = _check_directions(angles)
    assert_array_almost_equal(sticks, [[1, 0, 0]])
    # Testing if directions are already given in cartesian coordinates
    angles = [(0, 0, 1)]
    sticks = _check_directions(angles)
    assert_array_almost_equal(sticks, [[0, 0, 1]])
    # Testing more than one direction simultaneously
    angles = np.array([[90, 0], [30, 0]])
    sticks = _check_directions(angles)
    ref_vec = [np.sin(np.pi*30/180), 0, np.cos(np.pi*30/180)]
    assert_array_almost_equal(sticks, [[1, 0, 0], ref_vec])
    # Testing directions not aligned to planes x = 0, y = 0, or z = 0
    the1 = 0
    phi1 = 90
    the2 = 30
    phi2 = 45
    angles = np.array([(the1, phi1), (the2, phi2)])
    sticks = _check_directions(angles)
    ref_vec1 = (np.sin(np.pi*the1/180) * np.cos(np.pi*phi1/180),
                np.sin(np.pi*the1/180) * np.sin(np.pi*phi1/180),
                np.cos(np.pi*the1/180))
    ref_vec2 = (np.sin(np.pi*the2/180) * np.cos(np.pi*phi2/180),
                np.sin(np.pi*the2/180) * np.sin(np.pi*phi2/180),
                np.cos(np.pi*the2/180))
    assert_array_almost_equal(sticks, [ref_vec1, ref_vec2])


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

    Ssingle = 0.5*s1 + 0.5*s2

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


def test_dki():
    x1, x2, x3 = sphere2cart(1, np.deg2rad(30), np.deg2rad(0))


def test_compute_Wijkl():
    # two fiber not aligned to planes x = 0, y = 0, or z = 0
    mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087],
                       [0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
    angles = [(90, 10), (90, 10), (20, 30), (20, 30)]
    fie = 0.49  # intra axonal water fraction
    frac = [fie*50, (1 - fie)*50, fie*50, (1 - fie)*50]
    sticks = _check_directions(angles)
    mD = np.zeros((len(frac), 3, 3))
    for i in range(len(frac)):
        R = all_tensor_evecs(sticks[i])
        mD[i] = np.dot(np.dot(R, np.diag(mevals[i])), R.T)
        
    # Test symmetry of all 81 elements


def test_DKI_aligned_fibers():
    """ 
    Testing DKI simulations rotating a single fiber.
    When rotation fiber direction ground truth, if biological parameters
    don't change, kt[0] of a fiber aligned to axis x has to be equal to kt[1]
    of a fiber aligned to the axis y and equal to kt[2] of a fiber aligned to 
    axis z. The same is applicable for dt
    """
    # Defining parameters based on Neto Henriques et al., 2015. NeuroImage 111
    mevals = np.array([[0.00099, 0, 0],               # Intra-cellular 
                       [0.00226, 0.00087, 0.00087]])  # Extra-cellular
    frac = [49, 51]
    ### axis x
    angles = [(90, 0), (90, 0)]
    signal_fx, dt_fx, kt_fx = multi_tensor_dki(gtab, mevals, angles=angles,
                                               fractions=frac)
    ### axis y
    angles = [(90, 90), (90, 90)]
    signal_fy, dt_fy, kt_fy = multi_tensor_dki(gtab, mevals, angles=angles,
                                               fractions=frac)
    ### axis z
    angles = [(0, 0), (0, 0)]
    signal_fz, dt_fz, kt_fz = multi_tensor_dki(gtab, mevals, angles=angles,
                                               fractions=frac)
    
    assert_array_equal([kt_fx[0], kt_fx[1], kt_fx[2]],
                       [kt_fy[1], kt_fy[0], kt_fy[2]])
    assert_array_equal([kt_fx[0], kt_fx[1], kt_fx[2]],
                       [kt_fz[2], kt_fz[0], kt_fz[1]])
                       
    assert_array_equal([dt_fx[0], dt_fx[1], dt_fx[2]],
                       [dt_fy[1], dt_fy[0], dt_fy[2]])
    assert_array_equal([dt_fx[0], dt_fx[1], dt_fx[2]],
                       [dt_fz[2], dt_fz[0], dt_fz[1]])

    
def test_DKI_crossing_fibers():
    """ Testing DKI simulations of a crossing fiber
    """
    # two fiber not aligned to planes x = 0, y = 0, or z = 0
    mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087],
                       [0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
    angles = [(90, 10), (90, 10), (20, 30), (20, 30)]
    fie = 0.49
    frac = [fie*50, (1 - fie)*50, fie*50, (1 - fie)*50]
    signal, dt, kt = multi_tensor_dki(gtab, mevals, angles=angles, 
                                      fractions=frac)
    # in this simulations dt and kt cannot have zero elements
    for i in range(len(dt)):
        assert dt[i] != 0
    for i in range(len(kt)):
        assert kt[i] != 0
        
    # testing the single_diffkurt_tensor
    """x1, x2, x3 = sphere2cart(1, np.deg2rad(30), np.deg2rad(0))
    x4, x5, x6 = sphere2cart(1, np.deg2rad(45), np.deg2rad(0))
    bvals = np.array([0., 0., 1000, 1000, 1000, 1000, 2000, 2000])
    bvecs = np.asarray([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [x1, x2, x3], [0, 0, 1], [x4, x5, x6]])
    gtab = gradient_table(bvals, bvecs)
    A = dki_design_matrix(gtab)
    dt = np.array([1638e-6, 444e-6, 444e-6, 0, 0, 0])
    kt = np.array([1.7068, 0.8010, 0.8010, 0, 0, 0, 0, 0, 0, 0.3897, 0.3897,
                   0.2670, 0, 0, 0])
    MD = sum(dt[0:3]) / 3
    S0 = 150
    X = np.concatenate((dt, kt*MD*MD, np.array([np.log(S0)])), axis=0)
    S = np.exp(np.dot(A, X))"""


if __name__ == "__main__":

    test_multi_tensor()
