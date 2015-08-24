"""Test spherical harmonic models and the tools associated with those models"""
import numpy as np
import numpy.linalg as npl

from nose.tools import assert_equal, assert_raises, assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy.testing as npt
from scipy.special import sph_harm as sph_harm_sp

from dipy.core.sphere import hemi_icosahedron
from dipy.core.gradients import gradient_table
from dipy.sims.voxel import single_tensor
from dipy.reconst.peaks import peak_directions
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.reconst.interpolate import NearestNeighborInterpolator
from dipy.sims.voxel import multi_tensor_odf
from dipy.data import mrtrix_spherical_functions
from dipy.reconst import odf


from dipy.reconst.shm import (real_sph_harm, real_sym_sh_basis,
                              real_sym_sh_mrtrix, sph_harm_ind_list,
                              order_from_ncoef,
                              OpdtModel, normalize_data, hat, lcr_matrix,
                              smooth_pinv, bootstrap_data_array,
                              bootstrap_data_voxel, ResidualBootstrapWrapper,
                              CsaOdfModel, QballModel, SphHarmFit,
                              spherical_harmonics, anisotropic_power)

def test_order_from_ncoeff():
    """

    """
    # Just try some out:
    for sh_order in [2, 4, 6, 8, 12, 24]:
        m, n = sph_harm_ind_list(sh_order)
        n_coef = m.shape[0]
        npt.assert_equal(order_from_ncoef(n_coef), sh_order)


def test_sph_harm_ind_list():
    m_list, n_list = sph_harm_ind_list(8)
    assert_equal(m_list.shape, n_list.shape)
    assert_equal(m_list.shape, (45,))
    assert_true(np.all(np.abs(m_list) <= n_list))
    assert_array_equal(n_list % 2, 0)
    assert_raises(ValueError, sph_harm_ind_list, 1)


def test_real_sph_harm():
    # Tests derived from tables in
    # http://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    # where real spherical harmonic $Y^m_n$ is defined to be:
    #    Real($Y^m_n$) * sqrt(2) if m > 0
    #    $Y^m_n$                 if m == 0
    #    Imag($Y^m_n$) * sqrt(2) if m < 0

    rsh = real_sph_harm
    pi = np.pi
    exp = np.exp
    sqrt = np.sqrt
    sin = np.sin
    cos = np.cos
    assert_array_almost_equal(rsh(0, 0, 0, 0),
                              0.5 / sqrt(pi))
    assert_array_almost_equal(rsh(-2, 2, pi / 5, pi / 3),
                              0.25 * sqrt(15. / (2. * pi)) *
                             (sin(pi / 5.)) ** 2. * cos(0 + 2. * pi / 3) *
                              sqrt(2))
    assert_array_almost_equal(rsh(2, 2, pi / 5, pi / 3),
                              -1 * 0.25 * sqrt(15. / (2. * pi)) *
                              (sin(pi / 5.)) ** 2. * sin(0 - 2. * pi / 3) *
                              sqrt(2))
    assert_array_almost_equal(rsh(-2, 2, pi / 2, pi),
                              0.25 * sqrt(15 / (2. * pi)) *
                              cos(2. * pi) * sin(pi / 2.) ** 2. * sqrt(2))
    assert_array_almost_equal(rsh(2, 4, pi / 3., pi / 4.),
                              -1 * (3. / 8.) * sqrt(5. / (2. * pi)) *
                              sin(0 - 2. * pi / 4.) *
                              sin(pi / 3.) ** 2. *
                              (7. * cos(pi / 3.) ** 2. - 1) * sqrt(2))
    assert_array_almost_equal(rsh(-4, 4, pi / 6., pi / 8.),
                              (3. / 16.) * sqrt(35. / (2. * pi)) *
                              cos(0 + 4. * pi / 8.) * sin(pi / 6.) ** 4. *
                              sqrt(2))
    assert_array_almost_equal(rsh(4, 4, pi / 6., pi / 8.),
                              -1 * (3. / 16.) * sqrt(35. / (2. * pi)) *
                              sin(0 - 4. * pi / 8.) * sin(pi / 6.) ** 4. *
                              sqrt(2))

    aa = np.ones((3, 1, 1, 1))
    bb = np.ones((1, 4, 1, 1))
    cc = np.ones((1, 1, 5, 1))
    dd = np.ones((1, 1, 1, 6))
    assert_equal(rsh(aa, bb, cc, dd).shape, (3, 4, 5, 6))


def test_real_sym_sh_mrtrix():
    coef, expected, sphere = mrtrix_spherical_functions()
    basis, m, n = real_sym_sh_mrtrix(8, sphere.theta, sphere.phi)
    func = np.dot(coef, basis.T)
    assert_array_almost_equal(func, expected, 4)


def test_real_sym_sh_basis():
    # This test should do for now
    # The mrtrix basis should be the same as re-ordering and re-scaling the
    # fibernav basis
    new_order = [0, 5, 4, 3, 2, 1, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    sphere = hemi_icosahedron.subdivide(2)
    basis, m, n = real_sym_sh_mrtrix(4, sphere.theta, sphere.phi)
    expected = basis[:, new_order]
    expected *= np.where(m == 0, 1., np.sqrt(2))

    fibernav_basis, m, n = real_sym_sh_basis(4, sphere.theta, sphere.phi)
    assert_array_almost_equal(fibernav_basis, expected)


def test_smooth_pinv():
    hemi = hemi_icosahedron.subdivide(2)
    m, n = sph_harm_ind_list(4)
    B = real_sph_harm(m, n, hemi.theta[:, None], hemi.phi[:, None])

    L = np.zeros(len(m))
    C = smooth_pinv(B, L)
    D = np.dot(npl.inv(np.dot(B.T, B)), B.T)
    assert_array_almost_equal(C, D)

    L = n * (n + 1) * .05
    C = smooth_pinv(B, L)
    L = np.diag(L)
    D = np.dot(npl.inv(np.dot(B.T, B) + L * L), B.T)

    assert_array_almost_equal(C, D)

    L = np.arange(len(n)) * .05
    C = smooth_pinv(B, L)
    L = np.diag(L)
    D = np.dot(npl.inv(np.dot(B.T, B) + L * L), B.T)
    assert_array_almost_equal(C, D)


def test_normalize_data():

    sig = np.arange(1, 66)[::-1]

    where_b0 = np.zeros(65, 'bool')
    where_b0[0] = True
    d = normalize_data(sig, where_b0, 1)
    assert_raises(ValueError, normalize_data, sig, where_b0, out=sig)

    norm_sig = normalize_data(sig, where_b0, min_signal=1)
    assert_array_almost_equal(norm_sig, sig / 65.)
    norm_sig = normalize_data(sig, where_b0, min_signal=5)
    assert_array_almost_equal(norm_sig[-5:], 5 / 65.)

    where_b0[[0, 1]] = [True, True]
    norm_sig = normalize_data(sig, where_b0, min_signal=1)
    assert_array_almost_equal(norm_sig, sig / 64.5)
    norm_sig = normalize_data(sig, where_b0, min_signal=5)
    assert_array_almost_equal(norm_sig[-5:], 5 / 64.5)

    sig = sig * np.ones((2, 3, 1))

    where_b0[[0, 1]] = [True, False]
    norm_sig = normalize_data(sig, where_b0, min_signal=1)
    assert_array_almost_equal(norm_sig, sig / 65.)
    norm_sig = normalize_data(sig, where_b0, min_signal=5)
    assert_array_almost_equal(norm_sig[..., -5:], 5 / 65.)

    where_b0[[0, 1]] = [True, True]
    norm_sig = normalize_data(sig, where_b0, min_signal=1)
    assert_array_almost_equal(norm_sig, sig / 64.5)
    norm_sig = normalize_data(sig, where_b0, min_signal=5)
    assert_array_almost_equal(norm_sig[..., -5:], 5 / 64.5)


def make_fake_signal():
    hemisphere = hemi_icosahedron.subdivide(2)
    bvecs = np.concatenate(([[0, 0, 0]], hemisphere.vertices))
    bvals = np.zeros(len(bvecs)) + 2000
    bvals[0] = 0
    gtab = gradient_table(bvals, bvecs)

    evals = np.array([[2.1, .2, .2], [.2, 2.1, .2]]) * 10 ** -3
    evecs0 = np.eye(3)
    sq3 = np.sqrt(3) / 2.
    evecs1 = np.array([[sq3, .5, 0],
                       [.5, sq3, 0],
                       [0, 0, 1.]])
    evecs1 = evecs0
    a = evecs0[0]
    b = evecs1[1]
    S1 = single_tensor(gtab, .55, evals[0], evecs0)
    S2 = single_tensor(gtab, .45, evals[1], evecs1)
    return S1 + S2, gtab, np.vstack([a, b])


class TestQballModel(object):

    model = QballModel

    def test_single_voxel_fit(self):
        signal, gtab, expected = make_fake_signal()
        sphere = hemi_icosahedron.subdivide(4)

        model = self.model(gtab, sh_order=4, min_signal=1e-5,
                           assume_normed=True)
        fit = model.fit(signal)
        odf = fit.odf(sphere)
        assert_equal(odf.shape, sphere.phi.shape)
        directions, _, _ = peak_directions(odf, sphere)
        # Check the same number of directions
        n = len(expected)
        assert_equal(len(directions), n)
        # Check directions are unit vectors
        cos_similarity = (directions * directions).sum(-1)
        assert_array_almost_equal(cos_similarity, np.ones(n))
        # Check the directions == expected or -expected
        cos_similarity = (directions * expected).sum(-1)
        assert_array_almost_equal(abs(cos_similarity), np.ones(n))

        # Test normalize data
        model = self.model(gtab, sh_order=4, min_signal=1e-5,
                           assume_normed=False)
        fit = model.fit(signal * 5)
        odf_with_norm = fit.odf(sphere)
        assert_array_almost_equal(odf, odf_with_norm)

    def test_mulit_voxel_fit(self):
        signal, gtab, expected = make_fake_signal()
        sphere = hemi_icosahedron
        nd_signal = np.vstack([signal, signal])

        model = self.model(gtab, sh_order=4, min_signal=1e-5,
                           assume_normed=True)
        fit = model.fit(nd_signal)
        odf = fit.odf(sphere)
        assert_equal(odf.shape, (2,) + sphere.phi.shape)

        # Test fitting with mask, where mask is False odf should be 0
        fit = model.fit(nd_signal, mask=[False, True])
        odf = fit.odf(sphere)
        assert_array_equal(odf[0], 0.)

    def test_sh_order(self):
        signal, gtab, expected = make_fake_signal()
        model = self.model(gtab, sh_order=4, min_signal=1e-5)
        assert_equal(model.B.shape[1], 15)
        assert_equal(max(model.n), 4)
        model = self.model(gtab, sh_order=6, min_signal=1e-5)
        assert_equal(model.B.shape[1], 28)
        assert_equal(max(model.n), 6)

    def test_gfa(self):
        signal, gtab, expected = make_fake_signal()
        signal = np.ones((2, 3, 4, 1)) * signal
        sphere = hemi_icosahedron.subdivide(3)
        model = self.model(gtab, 6, min_signal=1e-5)
        fit = model.fit(signal)
        gfa_shm = fit.gfa
        gfa_odf = odf.gfa(fit.odf(sphere))
        assert_array_almost_equal(gfa_shm, gfa_odf, 3)

        # gfa should be 0 if all coefficients are 0 (masked areas)
        mask = np.zeros(signal.shape[:-1])
        fit = model.fit(signal, mask)
        assert_array_equal(fit.gfa, 0)


def test_SphHarmFit():
    coef = np.zeros((3, 4, 5, 45))
    mask = np.zeros((3, 4, 5), dtype=bool)

    fit = SphHarmFit(None, coef, mask)
    item = fit[0, 0, 0]
    assert_equal(item.shape, ())
    slice = fit[0]
    assert_equal(slice.shape, (4, 5))
    slice = fit[..., 0]
    assert_equal(slice.shape, (3, 4))


class TestOpdtModel(TestQballModel):
    model = OpdtModel


class TestCsaOdfModel(TestQballModel):
    model = CsaOdfModel


def test_hat_and_lcr():
    hemi = hemi_icosahedron.subdivide(3)
    m, n = sph_harm_ind_list(8)
    B = real_sph_harm(m, n, hemi.theta[:, None], hemi.phi[:, None])
    H = hat(B)
    B_hat = np.dot(H, B)
    assert_array_almost_equal(B, B_hat)

    R = lcr_matrix(H)
    d = np.arange(len(hemi.theta))
    r = d - np.dot(H, d)
    lev = np.sqrt(1 - H.diagonal())
    r /= lev
    r -= r.mean()

    r2 = np.dot(R, d)
    assert_array_almost_equal(r, r2)

    r3 = np.dot(d, R.T)
    assert_array_almost_equal(r, r3)


def test_bootstrap_array():
    B = np.array([[4, 5, 7, 4, 2.],
                  [4, 6, 2, 3, 6.]])
    H = hat(B.T)

    R = np.zeros((5, 5))
    d = np.arange(1, 6)
    dhat = np.dot(H, d)

    assert_array_almost_equal(bootstrap_data_voxel(dhat, H, R), dhat)
    assert_array_almost_equal(bootstrap_data_array(dhat, H, R), dhat)

    H = np.zeros((5, 5))


def test_ResidualBootstrapWrapper():
    B = np.array([[4, 5, 7, 4, 2.],
                  [4, 6, 2, 3, 6.]])
    B = B.T
    H = hat(B)
    d = np.arange(10) / 8.
    d.shape = (2, 5)
    dhat = np.dot(d, H)
    signal_object = NearestNeighborInterpolator(dhat, (1,))
    ms = .2
    where_dwi = np.ones(len(H), dtype=bool)

    boot_obj = ResidualBootstrapWrapper(signal_object, B, where_dwi, ms)
    assert_array_almost_equal(boot_obj[0], dhat[0].clip(ms, 1))
    assert_array_almost_equal(boot_obj[1], dhat[1].clip(ms, 1))

    dhat = np.column_stack([[.6, .7], dhat])
    signal_object = NearestNeighborInterpolator(dhat, (1,))
    where_dwi = np.concatenate([[False], where_dwi])
    boot_obj = ResidualBootstrapWrapper(signal_object, B, where_dwi, ms)
    assert_array_almost_equal(boot_obj[0], dhat[0].clip(ms, 1))
    assert_array_almost_equal(boot_obj[1], dhat[1].clip(ms, 1))


def test_sf_to_sh():
    # Subdividing a hemi_icosahedron twice produces 81 unique points, which
    # is more than enough to fit a order 8 (45 coefficients) spherical harmonic
    sphere = hemi_icosahedron.subdivide(2)

    mevals = np.array(([0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]))
    angles = [(0, 0), (90, 0)]

    odf = multi_tensor_odf(sphere.vertices, mevals, angles, [50, 50])

    # 1D case with the 3 bases functions
    odf_sh = sf_to_sh(odf, sphere, 8)
    odf2 = sh_to_sf(odf_sh, sphere, 8)
    assert_array_almost_equal(odf, odf2, 2)

    odf_sh = sf_to_sh(odf, sphere, 8, "mrtrix")
    odf2 = sh_to_sf(odf_sh, sphere, 8, "mrtrix")
    assert_array_almost_equal(odf, odf2, 2)

    odf_sh = sf_to_sh(odf, sphere, 8, "fibernav")
    odf2 = sh_to_sf(odf_sh, sphere, 8, "fibernav")
    assert_array_almost_equal(odf, odf2, 2)

    # 2D case
    odf2d = np.vstack((odf2, odf))
    odf2d_sh = sf_to_sh(odf2d, sphere, 8)
    odf2d_sf = sh_to_sf(odf2d_sh, sphere, 8)
    assert_array_almost_equal(odf2d, odf2d_sf, 2)


def test_faster_sph_harm():

    sh_order = 8

    m, n = sph_harm_ind_list(sh_order)

    theta = np.array([1.61491146,  0.76661665,  0.11976141,  1.20198246,  1.74066314,
                      1.5925956 ,  2.13022055,  0.50332859,  1.19868988,  0.78440679,
                      0.50686938,  0.51739718,  1.80342999,  0.73778957,  2.28559395,
                      1.29569064,  1.86877091,  0.39239191,  0.54043037,  1.61263047,
                      0.72695314,  1.90527318,  1.58186125,  0.23130073,  2.51695237,
                      0.99835604,  1.2883426 ,  0.48114057,  1.50079318,  1.07978624,
                      1.9798903 ,  2.36616966,  2.49233299,  2.13116602,  1.36801518,
                      1.32932608,  0.95926683,  1.070349  ,  0.76355762,  2.07148422,
                      1.50113501,  1.49823314,  0.89248164,  0.22187079,  1.53805373,
                      1.9765295 ,  1.13361568,  1.04908355,  1.68737368,  1.91732452,
                      1.01937457,  1.45839   ,  0.49641525,  0.29087155,  0.52824641,
                      1.29875871,  1.81023541,  1.17030475,  2.24953206,  1.20280498,
                      0.76399964,  2.16109722,  0.79780421,  0.87154509])

    phi = np.array([-1.5889514 , -3.11092733, -0.61328674, -2.4485381 ,  2.88058822,
                    2.02165946, -1.99783366,  2.71235211,  1.41577992, -2.29413676,
                    -2.24565773, -1.55548635,  2.59318232, -1.84672472, -2.33710739,
                    2.12111948,  1.87523722, -1.05206575, -2.85381987, -2.22808984,
                    2.3202034 , -2.19004474, -1.90358372,  2.14818373,  3.1030696 ,
                    -2.86620183, -2.19860123, -0.45468447, -3.0034923 ,  1.73345011,
                    -2.51716288,  2.49961525, -2.68782986,  2.69699056,  1.78566133,
                    -1.59119705, -2.53378963, -2.02476738,  1.36924987,  2.17600517,
                    2.38117241,  2.99021511, -1.4218007 , -2.44016802, -2.52868164,
                    3.01531658,  2.50093627, -1.70745826, -2.7863931 , -2.97359741,
                    2.17039906,  2.68424643,  1.77896086,  0.45476215,  0.99734418,
                    -2.73107896,  2.28815009,  2.86276506,  3.09450274, -3.09857384,
                    -1.06955885, -2.83826831,  1.81932195,  2.81296654])

    sh = spherical_harmonics(m, n, theta[:, None], phi[:, None])
    sh2 = sph_harm_sp(m, n, theta[:, None], phi[:, None])

    assert_array_almost_equal(sh, sh2, 8)

def test_anisotropic_power():
    testset = np.array([[  2.52783730e-01,  -8.63827673e-03,  -1.76620393e-03,
         -4.34251390e-03,  -5.06066428e-03,   3.81412854e-03,
          5.47631094e-03,   1.39282880e-03,  -1.97130701e-03,
         -2.24682506e-03,  -1.53527315e-04,  -2.10712616e-03,
         -4.25627588e-03,  -1.17619725e-03,   4.01009840e-03,
         -6.47920358e-04,   8.22481678e-04,   8.23377092e-04,
         -7.11836682e-04,   2.82638114e-04,   1.46455538e-03,
         -4.38317576e-04,  -2.61096300e-03,  -1.00295214e-03,
          7.23340554e-04,  -5.83654797e-04,  -3.58804282e-04,
         -3.94789973e-04,  -2.07047830e-04,   5.64046556e-05,
          2.41292157e-04,  -2.90961130e-04,   2.97358074e-04,
          1.15252750e-04,  -1.28836328e-04,  -3.46523499e-04,
          6.22056608e-04,   6.52724346e-05,   3.37867643e-04,
         -1.78158269e-04,  -8.06460277e-04,  -1.61496289e-04,
          1.03801834e-04,  -4.04971608e-04,   4.06671521e-04],
       [  6.54906618e-01,   6.01605429e-02,  -1.29095193e-03,
         -5.98209383e-02,  -1.68493669e-02,   7.15610018e-03,
          5.75249662e-03,  -1.35418171e-03,  -1.00052805e-02,
         -4.87658886e-03,   1.17126517e-02,   2.43934596e-03,
         -3.81930503e-03,   2.04635341e-03,   2.15191399e-03,
          1.45180291e-03,  -2.14420615e-03,  -8.67168497e-04,
         -3.98567829e-04,   2.71038437e-03,   1.43256098e-03,
         -2.55735319e-03,  -1.22762066e-03,  -2.94671979e-04,
         -8.24539983e-04,  -1.91217995e-03,   2.92358998e-03,
          3.84784047e-04,   5.65364655e-04,  -1.52953294e-04,
         -7.69113245e-04,   7.86232204e-05,   4.28770877e-04,
         -3.84858162e-04,   3.16806826e-04,   3.69560911e-05,
         -3.77435401e-04,  -9.92549109e-04,   9.46511835e-05,
         -4.20370516e-05,  -3.49006474e-04,   7.59378701e-04,
         -1.29880688e-04,   4.27303338e-04,  -6.21976709e-04],
       [  1.89360240e+00,   1.87110413e-01,   3.58681307e-02,
         -9.84681046e-02,  -6.31403011e-03,  -3.29867297e-02,
          1.14994091e-02,  -3.92568546e-03,  -6.85004580e-03,
         -5.52995838e-03,   2.08796166e-03,  -7.30827851e-03,
          2.28580857e-03,  -7.06024059e-03,   3.55504738e-03,
          2.63670661e-03,   7.45926612e-04,  -3.49921955e-03,
          2.64668186e-03,  -2.11425877e-03,  -3.21175666e-03,
          2.15584339e-03,   1.52908205e-03,   6.50053432e-03,
          4.84225433e-03,   1.04335885e-03,   2.03526656e-03,
          3.24381403e-03,  -8.39235407e-04,   1.03430794e-03,
         -1.24813979e-03,  -8.71406267e-04,  -4.90323771e-04,
          5.53422314e-04,  -1.03307848e-03,  -8.26799601e-04,
         -1.04932393e-03,   1.02847598e-03,   1.72438048e-03,
          6.09002332e-05,  -2.35245883e-04,   1.33029383e-03,
          9.53928674e-04,   4.53461286e-04,  -8.50660289e-04]])

    answers = [0.0, 3.4198238120739317, 5.2417375088492255]

    apvals = anisotropic_power(testset)   
    assert_array_almost_equal(apvals, answers)

if __name__ == "__main__":
    import nose
    nose.runmodule()
