"""Test spherical harmonic models and the tools associated with those models"""
import numpy as np
import numpy.linalg as npl

from nose.tools import assert_equal, assert_raises, assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal

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
                              OpdtModel, normalize_data, hat, lcr_matrix,
                              smooth_pinv, bootstrap_data_array,
                              bootstrap_data_voxel, ResidualBootstrapWrapper,
                              CsaOdfModel, QballModel, SphHarmFit)


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


if __name__ == "__main__":
    import nose
    nose.runmodule()
