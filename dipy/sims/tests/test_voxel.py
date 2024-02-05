import numpy as np

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_, assert_almost_equal)

from dipy.sims.voxel import (_check_directions, all_tensor_evecs, add_noise,
                             single_tensor, sticks_and_ball, multi_tensor_dki,
                             kurtosis_element, dki_signal, multi_tensor)
# from dipy.core.geometry import vec2vec_rotmat
from dipy.data import get_fnames
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.testing.decorators import set_random_number_generator


def setup_module():
    """Module-level setup"""
    global gtab, gtab_2s

    _, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)

    # 2 shells for techniques that requires multishell data
    bvals_2s = np.concatenate((bvals, bvals * 2), axis=0)
    bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
    gtab_2s = gradient_table(bvals_2s, bvecs_2s)


# Unused with missing references to basis
# def diff2eigenvectors(dx, dy, dz):
#     """ numerical derivatives 2 eigenvectors
#     """
#     u = np.array([dx, dy, dz])
#     u = u / np.linalg.norm(u)
#     R = vec2vec_rotmat(basis[:, 0], u)
#     eig0 = u
#     eig1 = np.dot(R, basis[:, 1])
#     eig2 = np.dot(R, basis[:, 2])
#     eigs = np.zeros((3, 3))
#     eigs[:, 0] = eig0
#     eigs[:, 1] = eig1
#     eigs[:, 2] = eig2
#     return eigs, R


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
    S_st = single_tensor(gtab, 1, evals=[d, 0, 0], evecs=[[0, 0, 0],
                                                          [0, 0, 0],
                                                          [1, 0, 0]])
    assert_array_almost_equal(S, S_st)


def test_single_tensor():
    evals = np.array([1.4, .35, .35]) * 10 ** (-3)
    evecs = np.eye(3)
    S = single_tensor(gtab, 100, evals, evecs, snr=None)
    assert_array_almost_equal(S[gtab.b0s_mask], 100)
    assert_(np.mean(S[~gtab.b0s_mask]) < 100)

    from dipy.reconst.dti import TensorModel
    m = TensorModel(gtab)
    t = m.fit(S)

    assert_array_almost_equal(t.fa, 0.707, decimal=3)


def test_multi_tensor():
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))
    e0 = np.array([np.sqrt(2) / 2., np.sqrt(2) / 2., 0])
    e1 = np.array([0, np.sqrt(2) / 2., np.sqrt(2) / 2.])
    mevecs = [all_tensor_evecs(e0), all_tensor_evecs(e1)]
    # odf = multi_tensor_odf(vertices, [0.5, 0.5], mevals, mevecs)
    # assert_(odf.shape == (len(vertices),))
    # assert_(np.all(odf <= 1) & np.all(odf >= 0))

    fimg, fbvals, fbvecs = get_fnames('small_101D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)

    s1 = single_tensor(gtab, 100, mevals[0], mevecs[0], snr=None)
    s2 = single_tensor(gtab, 100, mevals[1], mevecs[1], snr=None)

    Ssingle = 0.5*s1 + 0.5*s2

    S, _ = multi_tensor(gtab, mevals, S0=100,
                        angles=[(90, 45), (45, 90)],
                        fractions=[50, 50], snr=None)

    assert_array_almost_equal(S, Ssingle)


@set_random_number_generator(2000)
def test_snr(rng=None):
    s = single_tensor(gtab)

    # For reasonably large SNR, var(signal) ~= sigma**2, where sigma = 1/SNR
    for snr in [5, 10, 20]:
        sigma = 1.0 / snr
        for j in range(1000):
            s_noise = add_noise(s, snr, 1, noise_type='rician', rng=rng)

        assert_array_almost_equal(np.var(s_noise - s), sigma ** 2, decimal=2)


def test_all_tensor_evecs():
    e0 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])

    # Vectors are returned column-wise!
    desired = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0],
                        [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                        [0, 0, 1]]).T

    assert_array_almost_equal(all_tensor_evecs(e0), desired)


def test_kurtosis_elements():
    """ Testing symmetry of the elements of the KT

    As an 4th order tensor, KT has 81 elements. However, due to diffusion
    symmetry the KT is fully characterized by 15 independent elements. This
    test checks for this property.
    """
    # two fiber not aligned to planes x = 0, y = 0, or z = 0
    mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087],
                       [0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
    angles = [(80, 10), (80, 10), (20, 30), (20, 30)]
    fie = 0.49  # intra axonal water fraction
    frac = [fie * 50, (1-fie) * 50, fie * 50, (1-fie) * 50]
    sticks = _check_directions(angles)
    mD = np.zeros((len(frac), 3, 3))
    for i in range(len(frac)):
        R = all_tensor_evecs(sticks[i])
        mD[i] = np.dot(np.dot(R, np.diag(mevals[i])), R.T)

    # compute global DT
    D = np.zeros((3, 3))
    for i in range(len(frac)):
        D = D + frac[i]*mD[i]

    # compute voxel's MD
    MD = (D[0][0] + D[1][1] + D[2][2]) / 3

    # Reference dictionary with the 15 independent elements.
    # Note: The multiplication of the indexes (i+1) * (j+1) * (k+1) * (l+1)
    # for of an elements is only equal to this multiplication for another
    # element if an only if the element corresponds to an symmetry element.
    # Thus indexes multiplication is used as key of the reference dictionary
    kt_ref = {1: kurtosis_element(mD, frac, 0, 0, 0, 0),
              16: kurtosis_element(mD, frac, 1, 1, 1, 1),
              81: kurtosis_element(mD, frac, 2, 2, 2, 2),
              2: kurtosis_element(mD, frac, 0, 0, 0, 1),
              3: kurtosis_element(mD, frac, 0, 0, 0, 2),
              8: kurtosis_element(mD, frac, 0, 1, 1, 1),
              24: kurtosis_element(mD, frac, 1, 1, 1, 2),
              27: kurtosis_element(mD, frac, 0, 2, 2, 2),
              54: kurtosis_element(mD, frac, 1, 2, 2, 2),
              4: kurtosis_element(mD, frac, 0, 0, 1, 1),
              9: kurtosis_element(mD, frac, 0, 0, 2, 2),
              36: kurtosis_element(mD, frac, 1, 1, 2, 2),
              6: kurtosis_element(mD, frac, 0, 0, 1, 2),
              12: kurtosis_element(mD, frac, 0, 1, 1, 2),
              18: kurtosis_element(mD, frac, 0, 1, 2, 2)}

    # Testing all 81 possible elements
    xyz = [0, 1, 2]
    for i in xyz:
        for j in xyz:
            for k in xyz:
                for l in xyz:
                    key = (i+1) * (j+1) * (k+1) * (l+1)
                    assert_almost_equal(kurtosis_element(mD, frac, i, k, j, l),
                                        kt_ref[key])
                    # Testing optional function inputs
                    assert_almost_equal(kurtosis_element(mD, frac, i, k, j, l),
                                        kurtosis_element(mD, frac, i, k, j, l,
                                                         D, MD))


def test_DKI_simulations_aligned_fibers():
    """
    Testing DKI simulations when aligning the same fiber to different axis.

    If biological parameters don't change, kt[0] of a fiber aligned to axis x
    has to be equal to kt[1] of a fiber aligned to the axis y and equal to
    kt[2] of a fiber aligned to axis z. The same is applicable for dt
    """
    # Defining parameters based on Neto Henriques et al., 2015. NeuroImage 111
    mevals = np.array([[0.00099, 0, 0],               # Intra-cellular
                       [0.00226, 0.00087, 0.00087]])  # Extra-cellular
    frac = [49, 51]  # Compartment volume fraction
    # axis x
    angles = [(90, 0), (90, 0)]
    signal_fx, dt_fx, kt_fx = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                               fractions=frac)
    # axis y
    angles = [(90, 90), (90, 90)]
    signal_fy, dt_fy, kt_fy = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                               fractions=frac)
    # axis z
    angles = [(0, 0), (0, 0)]
    signal_fz, dt_fz, kt_fz = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                               fractions=frac)

    assert_array_equal([kt_fx[0], kt_fx[1], kt_fx[2]],
                       [kt_fy[1], kt_fy[0], kt_fy[2]])
    assert_array_equal([kt_fx[0], kt_fx[1], kt_fx[2]],
                       [kt_fz[2], kt_fz[0], kt_fz[1]])

    assert_array_equal([dt_fx[0], dt_fx[2], dt_fx[5]],
                       [dt_fy[2], dt_fy[0], dt_fy[5]])
    assert_array_equal([dt_fx[0], dt_fx[2], dt_fx[5]],
                       [dt_fz[5], dt_fz[0], dt_fz[2]])

    # testing S signal along axis x, y and z
    bvals = np.array([0, 0, 0, 1000, 1000, 1000, 2000, 2000, 2000])
    bvecs = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    gtab_axis = gradient_table(bvals, bvecs)
    # axis x
    S_fx = dki_signal(gtab_axis, dt_fx, kt_fx, S0=100)
    assert_array_almost_equal(S_fx[0:3], [100, 100, 100])  # test S f0r b=0
    # axis y
    S_fy = dki_signal(gtab_axis, dt_fy, kt_fy, S0=100)
    assert_array_almost_equal(S_fy[0:3], [100, 100, 100])  # test S f0r b=0
    # axis z
    S_fz = dki_signal(gtab_axis, dt_fz, kt_fz, S0=100)
    assert_array_almost_equal(S_fz[0:3], [100, 100, 100])  # test S f0r b=0

    # test S for b = 1000
    assert_array_almost_equal([S_fx[3], S_fx[4], S_fx[5]],
                              [S_fy[4], S_fy[3], S_fy[5]])
    assert_array_almost_equal([S_fx[3], S_fx[4], S_fx[5]],
                              [S_fz[5], S_fz[3], S_fz[4]])
    # test S for b = 2000
    assert_array_almost_equal([S_fx[6], S_fx[7], S_fx[8]],
                              [S_fy[7], S_fy[6], S_fy[8]])
    assert_array_almost_equal([S_fx[6], S_fx[7], S_fx[8]],
                              [S_fz[8], S_fz[6], S_fz[7]])


def test_DKI_crossing_fibers_simulations():
    """ Testing DKI simulations of a crossing fiber
    """
    # two fiber not aligned to planes x = 0, y = 0, or z = 0
    mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087],
                       [0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
    angles = [(80, 10), (80, 10), (20, 30), (20, 30)]
    fie = 0.49
    frac = [fie*50, (1 - fie)*50, fie*50, (1 - fie)*50]
    signal, dt, kt = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                      fractions=frac, snr=None)
    # in this simulations dt and kt cannot have zero elements
    for i in range(len(dt)):
        assert dt[i] != 0
    for i in range(len(kt)):
        assert kt[i] != 0

    # test S, dt and kt relative to the expected values computed from another
    # DKI package - UDKI (Neto Henriques et al., 2015)
    dt_ref = [1.0576161e-3, 0.1292542e-3, 0.4786179e-3,
              0.2667081e-3, 0.1136643e-3, 0.9888660e-3]
    kt_ref = [2.3529944, 0.8226448, 2.3011221, 0.2017312, -0.0437535,
              0.0404011, 0.0355281, 0.2449859, 0.2157668, 0.3495910,
              0.0413366, 0.3461519, -0.0537046, 0.0133414, -0.017441]
    assert_array_almost_equal(dt, dt_ref)
    assert_array_almost_equal(kt, kt_ref)
    assert_array_almost_equal(signal,
                              dki_signal(gtab_2s, dt_ref, kt_ref, S0=1.,
                                         snr=None),
                              decimal=5)


def test_single_tensor_btens():
    """ Testing single tensor simulations when a btensor is given
    """
    gtab_lte = gradient_table(gtab.bvals, gtab.bvecs, btens='LTE')
    gtab_ste = gradient_table(gtab.bvals, gtab.bvecs, btens='STE')

    # Check if Signals produced with LTE btensor gives same results as
    # previous simulations not specifying b-tensor
    evecs = np.eye(3)
    evals = np.array([1.4, .35, .35]) * 10 ** (-3)
    S_ref = single_tensor(gtab, 100, evals, evecs, snr=None)
    S_btens = single_tensor(gtab_lte, 100, evals, evecs, snr=None)
    assert_array_almost_equal(S_ref, S_btens)

    # Check if signals produced with STE btensor gives signals that matches
    # the signal decay for mean diffusivity
    md = np.sum(evals)/3
    S_ref = 100 * np.exp(-gtab.bvals * md)
    S_btens = single_tensor(gtab_ste, 100, evals, evecs, snr=None)
    assert_array_almost_equal(S_ref, S_btens)


def test_multi_tensor_btens():
    """ Testing multi tensor simulations when a btensor is given
    """
    mevals = np.array(([0.003, 0.0002, 0.0002],
                       [0.0015, 0.0003, 0.0003]))
    e0 = np.array([np.sqrt(2) / 2., np.sqrt(2) / 2., 0])
    e1 = np.array([0, np.sqrt(2) / 2., np.sqrt(2) / 2.])
    mevecs = [all_tensor_evecs(e0), all_tensor_evecs(e1)]

    gtab_ste = gradient_table(gtab.bvals, gtab.bvecs, btens='STE')

    s1 = single_tensor(gtab_ste, 100, mevals[0], mevecs[0], snr=None)
    s2 = single_tensor(gtab_ste, 100, mevals[1], mevecs[1], snr=None)

    Ssingle = 0.5*s1 + 0.5*s2

    S, _ = multi_tensor(gtab_ste, mevals, S0=100,
                        angles=[(90, 45), (45, 90)],
                        fractions=[50, 50], snr=None)

    assert_array_almost_equal(S, Ssingle)
