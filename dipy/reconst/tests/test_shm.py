"""Test spherical harmonic models and the tools associated with those models.
"""
import warnings
import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

from dipy.testing import assert_true
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal, assert_raises,)
from scipy.special import sph_harm as sph_harm_sp

from dipy.core.sphere import hemi_icosahedron, Sphere
from dipy.core.gradients import gradient_table
from dipy.core.interpolation import NearestNeighborInterpolator
from dipy.sims.voxel import single_tensor
from dipy.direction.peaks import peak_directions
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.sims.voxel import multi_tensor_odf
from dipy.data import mrtrix_spherical_functions
from dipy.reconst import odf


from dipy.reconst.shm import (real_sh_descoteaux_from_index, real_sym_sh_basis,
                              real_sym_sh_mrtrix, real_sh_descoteaux,
                              real_sh_tournier, sph_harm_ind_list,
                              order_from_ncoef, OpdtModel,
                              normalize_data, hat, lcr_matrix,
                              smooth_pinv, bootstrap_data_array,
                              bootstrap_data_voxel, ResidualBootstrapWrapper,
                              CsaOdfModel, QballModel, SphHarmFit,
                              spherical_harmonics, anisotropic_power,
                              calculate_max_order, sh_to_sf_matrix, gen_dirac,
                              convert_sh_to_full_basis, convert_sh_from_legacy,
                              convert_sh_to_legacy,
                              convert_sh_descoteaux_tournier,
                              descoteaux07_legacy_msg, tournier07_legacy_msg)


def test_order_from_ncoeff():
    # Just try some out:
    for sh_order_max in [2, 4, 6, 8, 12, 24]:
        m_values, l_values = sph_harm_ind_list(sh_order_max)
        n_coef = m_values.shape[0]
        assert_equal(order_from_ncoef(n_coef), sh_order_max)

        # Try out full basis
        m_full, l_full = sph_harm_ind_list(sh_order_max, True)
        n_coef_full = m_full.shape[0]
        assert_equal(order_from_ncoef(n_coef_full, True), sh_order_max)


def test_sph_harm_ind_list():
    m_list, l_list = sph_harm_ind_list(8)
    assert_equal(m_list.shape, l_list.shape)
    assert_equal(m_list.shape, (45,))
    assert_true(np.all(np.abs(m_list) <= l_list))
    assert_array_equal(l_list % 2, 0)
    assert_raises(ValueError, sph_harm_ind_list, 1)

    # Test for a full basis
    m_list, l_list = sph_harm_ind_list(8, True)
    assert_equal(m_list.shape, l_list.shape)
    # There are (sh_order + 1) * (sh_order + 1) coefficients
    assert_equal(m_list.shape, (81,))
    assert_true(np.all(np.abs(m_list) <= l_list))


def test_real_sh_descoteaux_from_index():
    # Tests derived from tables in
    # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    # where real spherical harmonic $Y^m_l$ is defined to be:
    #    Real($Y^m_l$) * sqrt(2) if m > 0
    #    $Y^m_l$                 if m == 0
    #    Imag($Y^m_l$) * sqrt(2) if m < 0

    rsh = real_sh_descoteaux_from_index
    pi = np.pi
    sqrt = np.sqrt
    sin = np.sin
    cos = np.cos

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

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

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

        assert_equal(rsh(aa, bb, cc, dd).shape, (3, 4, 5, 6))


def test_gen_dirac():

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

        sh = gen_dirac(
            np.array([0]), np.array([0]), np.array([0]), np.array([0]))

    assert_true(np.abs(sh[0] - 1.0/np.sqrt(4.0 * np.pi)) < 0.0001)


def test_real_sym_sh_mrtrix():
    coef, expected, sphere = mrtrix_spherical_functions()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        basis, m_values, l_values = real_sym_sh_mrtrix(8, sphere.theta,
                                                       sphere.phi)

    npt.assert_equal(len(w), 2)
    npt.assert_(issubclass(w[0].category, DeprecationWarning))
    npt.assert_(
        "dipy.reconst.shm.real_sym_sh_mrtrix is deprecated, Please use "
        "dipy.reconst.shm.real_sh_tournier instead" in str(w[0].message))
    npt.assert_(issubclass(w[1].category, PendingDeprecationWarning))
    npt.assert_(tournier07_legacy_msg in str(w[1].message))

    func = np.dot(coef, basis.T)
    assert_array_almost_equal(func, expected, 4)


def test_real_sym_sh_basis():

    new_order = [0, 5, 4, 3, 2, 1, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    sphere = hemi_icosahedron.subdivide(2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        basis, m_values, l_values = real_sym_sh_mrtrix(4, sphere.theta,
                                                       sphere.phi)

    expected = basis[:, new_order]
    expected *= np.where(m_values == 0, 1., np.sqrt(2))

    with warnings.catch_warnings(record=True) as w:
        descoteaux07_basis, m_values, l_values = real_sym_sh_basis(
            4, sphere.theta, sphere.phi)

    npt.assert_equal(len(w), 2)
    npt.assert_(issubclass(w[0].category, DeprecationWarning))
    npt.assert_(
        "dipy.reconst.shm.real_sym_sh_basis is deprecated, Please use "
        "dipy.reconst.shm.real_sh_descoteaux instead" in str(w[0].message))
    npt.assert_(issubclass(w[1].category, PendingDeprecationWarning))
    npt.assert_(descoteaux07_legacy_msg in str(w[1].message))

    assert_array_almost_equal(descoteaux07_basis, expected)


def test_real_sh_descoteaux1():
    # This test should do for now
    # The tournier07 basis should be the same as re-ordering and re-scaling the
    # descoteaux07 basis
    new_order = [0, 5, 4, 3, 2, 1, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    sphere = hemi_icosahedron.subdivide(2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        basis, m_values, l_values = real_sym_sh_mrtrix(4, sphere.theta,
                                                       sphere.phi)

    expected = basis[:, new_order]
    expected *= np.where(m_values == 0, 1., np.sqrt(2))

    with warnings.catch_warnings(record=True) as w:
        descoteaux07_basis, m_values, l_values = real_sh_descoteaux(
            4, sphere.theta, sphere.phi)

    npt.assert_equal(len(w), 1)
    npt.assert_(issubclass(w[0].category, PendingDeprecationWarning))
    npt.assert_(descoteaux07_legacy_msg in str(w[0].message))

    assert_array_almost_equal(descoteaux07_basis, expected)


def test_real_sh_tournier():
    vertices = hemi_icosahedron.subdivide(2).vertices
    mevals = np.array([[0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (60, 0)]
    odf = multi_tensor_odf(vertices, mevals, angles, [50, 50])

    mevals = np.array([[0.0015, 0.0003, 0.0003]])
    angles = [(0, 0)]
    odf2 = multi_tensor_odf(-vertices, mevals, angles, [100])

    sphere = Sphere(xyz=np.vstack((vertices, -vertices)))
    # Asymmetric spherical function with 162 coefficients
    sf = np.append(odf, odf2)

    # In order for our approximation to be precise enough, we
    # will use a SH basis of orders up to 10 (121 coefficients)

    with warnings.catch_warnings(record=True) as w:
        B, m_values, l_values = real_sh_tournier(
            10, sphere.theta, sphere.phi, full_basis=True)

    npt.assert_equal(len(w), 1)
    npt.assert_(issubclass(w[0].category, PendingDeprecationWarning))
    npt.assert_(tournier07_legacy_msg in str(w[0].message))

    invB = smooth_pinv(B, L=np.zeros_like(l_values))
    sh_coefs = np.dot(invB, sf)
    sf_approx = np.dot(B, sh_coefs)

    assert_array_almost_equal(sf_approx, sf, 2)


def test_real_sh_descoteaux2():
    vertices = hemi_icosahedron.subdivide(2).vertices
    mevals = np.array([[0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (60, 0)]
    odf = multi_tensor_odf(vertices, mevals, angles, [50, 50])

    mevals = np.array([[0.0015, 0.0003, 0.0003]])
    angles = [(0, 0)]
    odf2 = multi_tensor_odf(-vertices, mevals, angles, [100])

    sphere = Sphere(xyz=np.vstack((vertices, -vertices)))
    # Asymmetric spherical function with 162 coefficients
    sf = np.append(odf, odf2)

    # In order for our approximation to be precise enough, we
    # will use a SH basis of orders up to 10 (121 coefficients)

    with warnings.catch_warnings(record=True) as w:
        B, m_values, l_values = real_sh_descoteaux(10, sphere.theta, sphere.phi,
                                     full_basis=True)

    npt.assert_equal(len(w), 1)
    npt.assert_(issubclass(w[0].category, PendingDeprecationWarning))
    npt.assert_(descoteaux07_legacy_msg in str(w[0].message))

    invB = smooth_pinv(B, L=np.zeros_like(l_values))
    sh_coefs = np.dot(invB, sf)
    sf_approx = np.dot(B, sh_coefs)

    assert_array_almost_equal(sf_approx, sf, 2)


def test_sh_to_sf_matrix():
    sphere = Sphere(xyz=hemi_icosahedron.vertices)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

        B1, invB1 = sh_to_sf_matrix(sphere)

        B2, m_values, l_values = real_sh_descoteaux(4, sphere.theta, sphere.phi)

    invB2 = smooth_pinv(B2, L=np.zeros_like(l_values))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

        B3 = sh_to_sf_matrix(sphere, return_inv=False)

    assert_array_almost_equal(B1, B2.T)
    assert_array_almost_equal(invB1, invB2.T)
    assert_array_almost_equal(B3, B1)
    assert_raises(ValueError, sh_to_sf_matrix, sphere, basis_type="")


def test_smooth_pinv():
    hemi = hemi_icosahedron.subdivide(2)
    m_values, l_values = sph_harm_ind_list(4)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

        B = real_sh_descoteaux_from_index(
            m_values, l_values, hemi.theta[:, None], hemi.phi[:, None])

    L = np.zeros(len(m_values))
    C = smooth_pinv(B, L)
    D = np.dot(npl.inv(np.dot(B.T, B)), B.T)
    assert_array_almost_equal(C, D)

    L = l_values * (l_values + 1) * .05
    C = smooth_pinv(B, L)
    L = np.diag(L)
    D = np.dot(npl.inv(np.dot(B.T, B) + L * L), B.T)

    assert_array_almost_equal(C, D)

    L = np.arange(len(l_values)) * .05
    C = smooth_pinv(B, L)
    L = np.diag(L)
    D = np.dot(npl.inv(np.dot(B.T, B) + L * L), B.T)
    assert_array_almost_equal(C, D)


def test_normalize_data():

    sig = np.arange(1, 66)[::-1]

    where_b0 = np.zeros(65, 'bool')
    where_b0[0] = True
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
    evecs1 = evecs0
    a = evecs0[0]
    b = evecs1[1]
    S1 = single_tensor(gtab, .55, evals[0], evecs0)
    S2 = single_tensor(gtab, .45, evals[1], evecs1)
    return S1 + S2, gtab, np.vstack([a, b])


class TestQballModel:

    model = QballModel

    def test_single_voxel_fit(self):
        signal, gtab, expected = make_fake_signal()
        sphere = hemi_icosahedron.subdivide(4)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)

            model = self.model(gtab, sh_order_max=4, min_signal=1e-5,
                               assume_normed=True)

        fit = model.fit(signal)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)

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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)

            model = self.model(gtab, sh_order_max=4, min_signal=1e-5,
                               assume_normed=False)

        fit = model.fit(signal * 5)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)

            odf_with_norm = fit.odf(sphere)

        assert_array_almost_equal(odf, odf_with_norm)

    def test_mulit_voxel_fit(self):
        signal, gtab, expected = make_fake_signal()
        sphere = hemi_icosahedron
        nd_signal = np.vstack([signal, signal])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",  message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)

            model = self.model(gtab, sh_order_max=4, min_signal=1e-5,
                               assume_normed=True)

        fit = model.fit(nd_signal)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)

            odf = fit.odf(sphere)

        assert_equal(odf.shape, (2,) + sphere.phi.shape)

        # Test fitting with mask, where mask is False odf should be 0
        fit = model.fit(nd_signal, mask=[False, True])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)

            odf = fit.odf(sphere)

        assert_array_equal(odf[0], 0.)

    def test_sh_order(self):
        signal, gtab, expected = make_fake_signal()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)

            model = self.model(gtab, sh_order_max=4, min_signal=1e-5)

        assert_equal(model.B.shape[1], 15)
        assert_equal(max(model.l_values), 4)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)

            model = self.model(gtab, sh_order_max=6, min_signal=1e-5)

        assert_equal(model.B.shape[1], 28)
        assert_equal(max(model.l_values), 6)

    def test_gfa(self):
        signal, gtab, expected = make_fake_signal()
        signal = np.ones((2, 3, 4, 1)) * signal
        sphere = hemi_icosahedron.subdivide(3)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)

            model = self.model(gtab, 6, min_signal=1e-5)

        fit = model.fit(signal)
        gfa_shm = fit.gfa

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)

            gfa_odf = odf.gfa(fit.odf(sphere))

        assert_array_almost_equal(gfa_shm, gfa_odf, 3)

        # gfa should be 0 if all coefficients are 0 (masked areas)
        mask = np.zeros(signal.shape[:-1])
        fit = model.fit(signal, mask)
        assert_array_equal(fit.gfa, 0)

    def test_min_signal_default(self):
        signal, gtab, expected = make_fake_signal()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)

            model_default = self.model(gtab, 4)

        shm_default = model_default.fit(signal).shm_coeff

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)

            model_correct = self.model(gtab, 4, min_signal=1e-5)

        shm_correct = model_correct.fit(signal).shm_coeff
        assert_equal(shm_default, shm_correct)


def test_SphHarmFit():
    coef = np.zeros((3, 4, 5, 45))
    mask = np.zeros((3, 4, 5), dtype=bool)

    fit = SphHarmFit(None, coef, mask)
    item = fit[0, 0, 0]
    assert_equal(item.shape, ())
    data = fit[0]
    assert_equal(data.shape, (4, 5))
    data = fit[:, :, 0]
    assert_equal(data.shape, (3, 4))


class TestOpdtModel(TestQballModel):
    model = OpdtModel


class TestCsaOdfModel(TestQballModel):
    model = CsaOdfModel


def test_hat_and_lcr():
    hemi = hemi_icosahedron.subdivide(3)
    m_values, l_values = sph_harm_ind_list(8)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

        B = real_sh_descoteaux_from_index(
            m_values, l_values, hemi.theta[:, None], hemi.phi[:, None])

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
    hemisphere = hemi_icosahedron.subdivide(2)
    mevals = np.array([[0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (60, 0)]
    odf = multi_tensor_odf(hemisphere.vertices, mevals, angles, [50, 50])

    # 1D case with the 2 symmetric bases functions
    # Tournier basis
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=tournier07_legacy_msg,
            category=PendingDeprecationWarning)

        odf_sh = sf_to_sh(odf, hemisphere, 8, "tournier07")
        odf_reconst = sh_to_sf(odf_sh, hemisphere, 8, "tournier07")

    assert_array_almost_equal(odf, odf_reconst, 2)

    # Legacy definition
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=tournier07_legacy_msg,
            category=PendingDeprecationWarning)

        odf_sh = sf_to_sh(odf, hemisphere, 8, "tournier07", legacy=True)
        odf_reconst = sh_to_sf(
            odf_sh, hemisphere, 8, "tournier07", legacy=True)

    assert_array_almost_equal(odf, odf_reconst, 2)

    # Descoteaux basis
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

        odf_sh = sf_to_sh(odf, hemisphere, 8, "descoteaux07")
        odf_reconst = sh_to_sf(odf_sh, hemisphere, 8, "descoteaux07")

    assert_array_almost_equal(odf, odf_reconst, 2)

    # Legacy definition
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

        odf_sh = sf_to_sh(odf, hemisphere, 8, "descoteaux07", legacy=True)
        odf_reconst = sh_to_sf(
            odf_sh, hemisphere, 8, "descoteaux07", legacy=True)

    assert_array_almost_equal(odf, odf_reconst, 2)

    # We now create an asymmetric signal
    # to try out our full SH basis
    mevals = np.array([[0.0015, 0.0003, 0.0003]])
    angles = [(0, 0)]
    odf2 = multi_tensor_odf(hemisphere.vertices, mevals, angles, [100])

    # We simulate our asymmetric signal by using a different ODF
    # per hemisphere. The sphere used is a concatenation of the
    # vertices of our hemisphere, for a total of 162 vertices.
    sphere = Sphere(xyz=np.vstack((hemisphere.vertices, -hemisphere.vertices)))
    asym_odf = np.append(odf, odf2)

    # Try out full bases with order 10 (121 coefficients)
    # Tournier basis
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=tournier07_legacy_msg,
            category=PendingDeprecationWarning)

        odf_sh = sf_to_sh(asym_odf, sphere, 10, 'tournier07', full_basis=True)
        odf_reconst = sh_to_sf(
            odf_sh, sphere, 10, 'tournier07', full_basis=True)

    assert_array_almost_equal(odf_reconst, asym_odf, 2)

    # Legacy definition
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=tournier07_legacy_msg,
            category=PendingDeprecationWarning)

        odf_sh = sf_to_sh(asym_odf, sphere, 10, 'tournier07',
                          full_basis=True, legacy=True)
        odf_reconst = sh_to_sf(odf_sh, sphere, 10, 'tournier07',
                               full_basis=True, legacy=True)

    assert_array_almost_equal(odf_reconst, asym_odf, 2)

    # Descoteaux basis
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

        odf_sh = sf_to_sh(
            asym_odf, sphere, 10, 'descoteaux07', full_basis=True)
        odf_reconst = sh_to_sf(
            odf_sh, sphere, 10, 'descoteaux07', full_basis=True)

    assert_array_almost_equal(odf_reconst, asym_odf, 2)

    # Legacy definition
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

        odf_sh = sf_to_sh(asym_odf, sphere, 10, 'descoteaux07',
                          full_basis=True, legacy=True)
        odf_reconst = sh_to_sf(odf_sh, sphere, 10, 'descoteaux07',
                               full_basis=True, legacy=True)

    assert_array_almost_equal(odf_reconst, asym_odf, 2)

    # An invalid basis name should raise an error
    assert_raises(ValueError, sh_to_sf, odf, hemisphere, basis_type="")
    assert_raises(ValueError, sf_to_sh, odf_sh, hemisphere, basis_type="")

    # 2D case
    odf2d = np.vstack((odf, odf))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

        odf2d_sh = sf_to_sh(odf2d, hemisphere, 8)
        odf2d_sf = sh_to_sf(odf2d_sh, hemisphere, 8)

    assert_array_almost_equal(odf2d, odf2d_sf, 2)


def test_faster_sph_harm():

    sh_order_max = 8
    m_values, l_values = sph_harm_ind_list(sh_order_max)
    theta = np.array([1.61491146,  0.76661665,  0.11976141,  1.20198246,
                      1.74066314, 1.5925956,  2.13022055,  0.50332859,
                      1.19868988,  0.78440679, 0.50686938,  0.51739718,
                      1.80342999,  0.73778957,  2.28559395, 1.29569064,
                      1.86877091,  0.39239191,  0.54043037,  1.61263047,
                      0.72695314,  1.90527318,  1.58186125,  0.23130073,
                      2.51695237, 0.99835604,  1.2883426,  0.48114057,
                      1.50079318,  1.07978624, 1.9798903,  2.36616966,
                      2.49233299,  2.13116602,  1.36801518, 1.32932608,
                      0.95926683,  1.070349,  0.76355762, 2.07148422,
                      1.50113501,  1.49823314,  0.89248164,  0.22187079,
                      1.53805373, 1.9765295,  1.13361568,  1.04908355,
                      1.68737368,  1.91732452, 1.01937457,  1.45839,
                      0.49641525,  0.29087155,  0.52824641, 1.29875871,
                      1.81023541,  1.17030475,  2.24953206,  1.20280498,
                      0.76399964,  2.16109722,  0.79780421,  0.87154509])

    phi = np.array([-1.5889514, -3.11092733, -0.61328674, -2.4485381,
                    2.88058822, 2.02165946, -1.99783366,  2.71235211,
                    1.41577992, -2.29413676, -2.24565773, -1.55548635,
                    2.59318232, -1.84672472, -2.33710739, 2.12111948,
                    1.87523722, -1.05206575, -2.85381987,
                    -2.22808984, 2.3202034, -2.19004474, -1.90358372,
                    2.14818373,  3.1030696, -2.86620183, -2.19860123,
                    -0.45468447, -3.0034923,  1.73345011, -2.51716288,
                    2.49961525, -2.68782986,  2.69699056,  1.78566133,
                    -1.59119705, -2.53378963, -2.02476738,  1.36924987,
                    2.17600517, 2.38117241,  2.99021511, -1.4218007,
                    -2.44016802, -2.52868164, 3.01531658,  2.50093627,
                    -1.70745826, -2.7863931, -2.97359741, 2.17039906,
                    2.68424643,  1.77896086,  0.45476215,  0.99734418,
                    -2.73107896,  2.28815009,  2.86276506,  3.09450274,
                    -3.09857384, -1.06955885, -2.83826831,  1.81932195,
                    2.81296654])

    sh = spherical_harmonics(m_values, l_values, theta[:, None], phi[:, None])
    sh2 = sph_harm_sp(m_values, l_values, theta[:, None], phi[:, None])

    assert_array_almost_equal(sh, sh2, 8)
    sh = spherical_harmonics(m_values, l_values, theta[:, None], phi[:, None],
                             use_scipy=False)
    assert_array_almost_equal(sh, sh2, 8)


def test_anisotropic_power():
    for n_coeffs in [6, 15, 28, 45, 66, 91]:
        for norm_factor in [0.0005, 0.00001]:

            # Create some really simple cases:
            coeffs = np.ones((3, n_coeffs))
            max_order = calculate_max_order(coeffs.shape[-1])
            # For the case where all coeffs == 1, the ap is simply log of the
            # number of even orders up to the maximal order:
            analytic = (np.log(len(range(2, max_order + 2, 2))) -
                        np.log(norm_factor))

            answers = [analytic] * 3
            apvals = anisotropic_power(coeffs, norm_factor=norm_factor)
            assert_array_almost_equal(apvals, answers)
            # Test that this works for single voxel arrays as well:
            assert_array_almost_equal(
                anisotropic_power(coeffs[1],
                                  norm_factor=norm_factor),
                answers[1])

    # Test that even when we look at an all-zeros voxel, this
    # avoids a log-of-zero warning:
    with warnings.catch_warnings(record=True) as w:
        assert_equal(anisotropic_power(np.zeros(6)), 0)
        assert len(w) == 0


def test_calculate_max_order():
    """Based on the table in:
    https://jdtournier.github.io/mrtrix-0.2/tractography/preprocess.html
    """
    orders = [2, 4, 6, 8, 10, 12]
    n_coeffs_sym = [6, 15, 28, 45, 66, 91]
    # n = (R + 1)^2 for a full basis
    n_coeffs_full = [9, 25, 49, 81, 121, 169]
    for o, n_sym, n_full in zip(orders, n_coeffs_sym, n_coeffs_full):
        assert_equal(calculate_max_order(n_sym), o)
        assert_equal(calculate_max_order(n_full, True), o)

    assert_raises(ValueError, calculate_max_order, 29)
    assert_raises(ValueError, calculate_max_order, 29, True)


def test_convert_sh_to_full_basis():
    hemisphere = hemi_icosahedron.subdivide(2)
    mevals = np.array([[0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (60, 0)]
    odf = multi_tensor_odf(hemisphere.vertices, mevals, angles, [50, 50])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        sh_coeffs = sf_to_sh(odf, hemisphere, 8)

    full_sh_coeffs = convert_sh_to_full_basis(sh_coeffs)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

        odf_reconst = sh_to_sf(full_sh_coeffs, hemisphere, 8, full_basis=True)

    assert_array_almost_equal(odf, odf_reconst, 2)


def test_convert_sh_from_legacy():
    hemisphere = hemi_icosahedron.subdivide(2)
    mevals = np.array([[0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (60, 0)]
    odf = multi_tensor_odf(hemisphere.vertices, mevals, angles, [50, 50])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        sh_coeffs = sf_to_sh(odf, hemisphere, 8, legacy=True)

    converted_coeffs = convert_sh_from_legacy(sh_coeffs, 'descoteaux07')
    expected_coeffs = sf_to_sh(odf, hemisphere, 8, legacy=False)

    assert_array_almost_equal(converted_coeffs, expected_coeffs, 2)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=tournier07_legacy_msg,
            category=PendingDeprecationWarning)

        sh_coeffs = sf_to_sh(odf, hemisphere, 8, basis_type='tournier07',
                             legacy=True)
    converted_coeffs = convert_sh_from_legacy(sh_coeffs, 'tournier07')
    expected_coeffs = sf_to_sh(odf, hemisphere, 8,
                               basis_type='tournier07', legacy=False)

    assert_array_almost_equal(converted_coeffs, expected_coeffs, 2)

    # 2D case
    odfs = np.array([odf, odf])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=tournier07_legacy_msg,
            category=PendingDeprecationWarning)

        sh_coeffs = sf_to_sh(odfs, hemisphere, 8, basis_type='tournier07',
                             legacy=True, full_basis=True)

    converted_coeffs = convert_sh_from_legacy(sh_coeffs, 'tournier07',
                                              full_basis=True)
    expected_coeffs = sf_to_sh(odfs, hemisphere, 8, basis_type='tournier07',
                               full_basis=True, legacy=False)

    assert_array_almost_equal(converted_coeffs, expected_coeffs, 2)
    assert_raises(ValueError, convert_sh_from_legacy, sh_coeffs, '', True)


def test_convert_sh_to_legacy():
    hemisphere = hemi_icosahedron.subdivide(2)
    mevals = np.array([[0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (60, 0)]
    odf = multi_tensor_odf(hemisphere.vertices, mevals, angles, [50, 50])

    sh_coeffs = sf_to_sh(odf, hemisphere, 8, legacy=False)
    converted_coeffs = convert_sh_to_legacy(sh_coeffs, 'descoteaux07')

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)

        expected_coeffs = sf_to_sh(odf, hemisphere, 8, legacy=True)

    assert_array_almost_equal(converted_coeffs, expected_coeffs, 2)

    sh_coeffs = sf_to_sh(odf, hemisphere, 8, basis_type='tournier07',
                         legacy=False)
    converted_coeffs = convert_sh_to_legacy(sh_coeffs, 'tournier07')

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=tournier07_legacy_msg,
            category=PendingDeprecationWarning)

        expected_coeffs = sf_to_sh(odf, hemisphere, 8, basis_type='tournier07',
                                   legacy=True)

    assert_array_almost_equal(converted_coeffs, expected_coeffs, 2)

    # 2D case
    odfs = np.array([odf, odf])
    sh_coeffs = sf_to_sh(odfs, hemisphere, 8, basis_type='tournier07',
                         full_basis=True, legacy=False)
    converted_coeffs = convert_sh_to_legacy(sh_coeffs, 'tournier07',
                                            full_basis=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=tournier07_legacy_msg,
            category=PendingDeprecationWarning)

        expected_coeffs = sf_to_sh(
            odfs, hemisphere, 8, basis_type='tournier07', legacy=True,
            full_basis=True)

    assert_array_almost_equal(converted_coeffs, expected_coeffs, 2)
    assert_raises(ValueError, convert_sh_to_legacy, sh_coeffs, '', True)


def test_convert_sh_descoteaux_tournier():

    # case: max degree zero
    sh_coeffs = np.array([1.54])  # there is only an l=0,m=0 coefficient
    converted_coeffs = convert_sh_descoteaux_tournier(sh_coeffs)
    assert_array_equal(converted_coeffs, sh_coeffs)

    # case: max degree 4
    sh_coeffs = np.array([
        1,
        2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12, 13, 14, 15,
    ], dtype=float)
    # expected result is that there is a swap m <--> -m
    expected_coeffs = np.array([
        1,
        6, 5, 4, 3, 2,
        15, 14, 13, 12, 11, 10, 9, 8, 7,
    ], dtype=float)
    converted_coeffs = convert_sh_descoteaux_tournier(sh_coeffs)
    assert_array_equal(converted_coeffs, expected_coeffs)

    # case: max degree 4 but with more axes
    dim0 = 2
    dim1 = 3
    sh_coeffs_grid = np.array(
        [np.linspace(10*i, 10*i+1, 6) for i in range(6)]
    ).reshape(dim0, dim1, 6)
    converted_coeffs_grid = convert_sh_descoteaux_tournier(sh_coeffs_grid)
    assert_equal(sh_coeffs_grid.shape, converted_coeffs_grid.shape)
    for i0 in range(dim0):
        for i1 in range(dim1):
            shc = sh_coeffs_grid[i0, i1]  # shc is short for "sh_coeffs"
            converted_coeffs = converted_coeffs_grid[i0, i1]
            expected_coeffs = np.array([
                shc[0],
                shc[5], shc[4], shc[3], shc[2], shc[1],
            ])
            assert_array_equal(converted_coeffs, expected_coeffs)
