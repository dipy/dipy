import numpy as np
from dipy.data import get_gtab_taiwan_dsi
from numpy.testing import (assert_almost_equal,
                           assert_array_almost_equal,
                           assert_equal,
                           run_module_suite)
from dipy.reconst.mapmri import MapmriModel, mapmri_index_matrix, mapmri_EAP
from dipy.reconst import dti, mapmri
from dipy.sims.voxel import (MultiTensor, all_tensor_evecs,  multi_tensor_pdf)
from scipy.special import gamma
from scipy.misc import factorial
from dipy.data import get_sphere


def int_func(n):
    f = np.sqrt(2) * factorial(n) / float(((gamma(1 + n / 2.0)) *
                                          np.sqrt(2**(n + 1) * factorial(n))))
    return f


def generate_signal_crossing(gtab, lambda1, lambda2, lambda3, angle2=60):
    mevals = np.array(([lambda1, lambda2, lambda3],
                       [lambda1, lambda2, lambda3]))
    angl = [(0, 0), (angle2, 0)]
    S, sticks = MultiTensor(gtab, mevals, S0=100.0, angles=angl,
                            fractions=[50, 50], snr=None)
    return S


def test_mapmri_number_of_coefficients(radial_order=6):
    indices = mapmri_index_matrix(radial_order)
    n_c = indices.shape[0]
    F = radial_order / 2
    n_gt = np.round(1 / 6.0 * (F + 1) * (F + 2) * (4 * F + 3))
    assert_equal(n_c, n_gt)


def test_mapmri_signal_fitting(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab, l1, l2, l3)

    mapm = MapmriModel(gtab, radial_order=radial_order,
                       laplacian_weighting=0.02)
    mapfit = mapm.fit(S)
    S_reconst = mapfit.predict(gtab, 1.0)

    # test the signal reconstruction
    S = S / S[0]
    nmse_signal = np.sqrt(np.sum((S - S_reconst) ** 2)) / (S.sum())
    assert_almost_equal(nmse_signal, 0.0, 3)


def test_mapmri_pdf_integral_unity(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab, l1, l2, l3)

    # test MAPMRI fitting

    mapm = MapmriModel(gtab, radial_order=radial_order,
                       laplacian_weighting=0.02)
    mapfit = mapm.fit(S)
    c_map = mapfit.mapmri_coeff

    R = mapfit.mapmri_R
    mu = mapfit.mapmri_mu

    # test if the analytical integral of the pdf is equal to one
    indices = mapmri_index_matrix(radial_order)
    integral = 0
    for i in range(indices.shape[0]):
        n1, n2, n3 = indices[i]
        integral += c_map[i] * int_func(n1) * int_func(n2) * int_func(n3)

    assert_almost_equal(integral, 1.0, 3)


def test_mapmri_pdf_correctness(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab, l1, l2, l3)

    # test MAPMRI fitting
    mapm = MapmriModel(gtab, radial_order=radial_order,
                       laplacian_weighting=0.02)
    mapfit = mapm.fit(S)
    c_map = mapfit.mapmri_coeff

    R = mapfit.mapmri_R
    mu = mapfit.mapmri_mu

    # compare the mapmri pdf with the ground truth multi_tensor pdf

    mevals = np.array(([l1, l2, l3],
                       [l1, l2, l3]))
    angl = [(0, 0), (60, 0)]
    sphere = get_sphere('symmetric724')
    v = sphere.vertices
    radius = 10e-3
    r_points = v * radius
    pdf_mt = multi_tensor_pdf(r_points, mevals=mevals,
                              angles=angl, fractions=[50, 50])
    pdf_map = mapmri_EAP(r_points, radial_order, c_map, mu, R)

    nmse_pdf = np.sqrt(np.sum((pdf_mt - pdf_map) ** 2)) / (pdf_mt.sum())
    assert_almost_equal(nmse_pdf, 0.0, 2)


def test_mapmri_metrics_anisotropic(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab, l1, l2, l3, angle2=0)

    # test MAPMRI q-space indices

    mapm = MapmriModel(gtab, radial_order=radial_order,
                       laplacian_regularization=False)
    mapfit = mapm.fit(S)

    tau = 1 / (4 * np.pi ** 2)

    # ground truth indices estimated from the DTI tensor
    rtpp_gt = 1. / (2 * np.sqrt(np.pi * l1 * tau))
    rtap_gt = (
        1. / (2 * np.sqrt(np.pi * l2 * tau)) * 1. /
        (2 * np.sqrt(np.pi * l3 * tau))
    )
    rtop_gt = rtpp_gt * rtap_gt
    msd_gt = 2 * (l1 + l2 + l3) * tau
    qiv_gt = (
        (64 * np.pi ** (7 / 2.) * (l1 * l2 * l3 * tau ** 3) ** (3 / 2.)) /
        ((l2 * l3 + l1 * (l2 + l3)) * tau ** 2)
    )

    assert_almost_equal(mapfit.rtap(), rtap_gt, 5)
    assert_almost_equal(mapfit.rtpp(), rtpp_gt, 5)
    assert_almost_equal(mapfit.rtop(), rtop_gt, 5)
    assert_almost_equal(mapfit.ng(), 0., 5)
    assert_almost_equal(mapfit.ng_parallel(), 0., 5)
    assert_almost_equal(mapfit.ng_perpendicular(), 0., 5)
    assert_almost_equal(mapfit.msd(), msd_gt, 5)
    assert_almost_equal(mapfit.qiv(), qiv_gt, 5)

def test_mapmri_laplacian_anisotropic(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab, l1, l2, l3, angle2=0)

    mapm = MapmriModel(gtab, radial_order=radial_order,
                    laplacian_regularization=False)
    mapfit = mapm.fit(S)

    tau = 1 / (4 * np.pi ** 2)

    # ground truth norm of laplacian of tensor
    norm_of_laplacian_gt = (
    (3 * (l1 ** 2 + l2 ** 2 + l3 ** 2) + 2 * l2 * l3 + 2 * l1 * (l2 + l3)) *
    (np.pi ** (5 / 2.) * tau) /
    (np.sqrt(2 * l1 * l2 * l3 * tau))
    )

    # check if estimated laplacian corresponds with ground truth
    laplacian_matrix = mapmri.mapmri_laplacian_reg_matrix(
                mapm.ind_mat, mapfit.mu, mapm.R_mat,
                mapm.L_mat, mapm.S_mat)

    coef = mapfit._mapmri_coef
    norm_of_laplacian = np.dot(np.dot(coef, laplacian_matrix), coef)

    assert_almost_equal(norm_of_laplacian, norm_of_laplacian_gt)

def test_mapmri_laplacian_isotropic(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0003, 0.0003, 0.0003] # isotropic diffusivities
    S = generate_signal_crossing(gtab, l1, l2, l3, angle2=0)

    mapm = MapmriModel(gtab, radial_order=radial_order,
                    laplacian_regularization=False,
                    anisotropic_scaling=False)
    mapfit = mapm.fit(S)

    tau = 1 / (4 * np.pi ** 2)

    # ground truth norm of laplacian of tensor
    norm_of_laplacian_gt = (
    (3 * (l1 ** 2 + l2 ** 2 + l3 ** 2) + 2 * l2 * l3 + 2 * l1 * (l2 + l3)) *
    (np.pi ** (5 / 2.) * tau) /
    (np.sqrt(2 * l1 * l2 * l3 * tau))
    )

    # check if estimated laplacian corresponds with ground truth
    laplacian_matrix = mapmri.mapmri_isotropic_laplacian_reg_matrix(
                                                    radial_order, mapfit.mu[0])

    coef = mapfit._mapmri_coef
    norm_of_laplacian = np.dot(np.dot(coef, laplacian_matrix), coef)

    assert_almost_equal(norm_of_laplacian, norm_of_laplacian_gt)

def test_signal_fitting_equality_anisotropic_isotropic(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab, l1, l2, l3, angle2=60)

    tenmodel = dti.TensorModel(gtab)
    evals = tenmodel.fit(S).evals
    tau = 1 / (4 * np.pi ** 2)
    mumean = np.sqrt(evals.mean() * 2 * tau)
    mu = np.array([mumean, mumean, mumean])

    qvals = np.sqrt(gtab.bvals / tau) / (2 * np.pi)
    q = gtab.bvecs * qvals[:, None]

    M_aniso = mapmri.mapmri_phi_matrix(radial_order, mu, q.T)
    M_iso = mapmri.mapmri_isotropic_phi_matrix(radial_order, mumean, q)

    # test if anisotropic and isotropic implementation produce equal results
    # if the same isotropic scale factors are used
    s_fitted_aniso = np.dot(M_aniso,
        np.dot(np.dot(np.linalg.inv(np.dot(M_aniso.T, M_aniso)), M_aniso.T), S)
        )
    s_fitted_iso = np.dot(M_iso,
        np.dot(np.dot(np.linalg.inv(np.dot(M_iso.T, M_iso)), M_iso.T), S)
        )

    assert_array_almost_equal(s_fitted_aniso, s_fitted_iso)

    # test if the implemented version also produces the same result
    mapm = MapmriModel(gtab, radial_order=radial_order,
                       laplacian_regularization=False,
                       anisotropic_scaling=False)
    s_fitted_implemented_isotropic = mapm.fit(S).fitted_signal()

    # normalize non-implemented fitted signal with b0 value
    s_fitted_aniso_norm = s_fitted_aniso / s_fitted_aniso.max()

    assert_array_almost_equal(s_fitted_aniso_norm,
                              s_fitted_implemented_isotropic)

def test_mapmri_isotropic_design_matrix_separability(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    tau = 1 / (4 * np.pi ** 2)
    qvals = np.sqrt(gtab.bvals / tau) / (2 * np.pi)
    q = gtab.bvecs * qvals[:, None]
    mu = 0.0003 #random value

    M = mapmri.mapmri_isotropic_phi_matrix(radial_order, mu, q)
    M_independent = mapmri.mapmri_isotropic_M_mu_independent(radial_order, q)
    M_dependent = mapmri.mapmri_isotropic_M_mu_dependent(radial_order, mu, qvals)

    M_reconstructed = M_independent * M_dependent

    assert_array_almost_equal(M, M_reconstructed)


def test_mapmri_metrics_isotropic(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0003, 0.0003, 0.0003] # isotropic diffusivities
    S = generate_signal_crossing(gtab, l1, l2, l3, angle2=0)

    # test MAPMRI q-space indices

    mapm = MapmriModel(gtab, radial_order=radial_order,
                       laplacian_regularization=False,
                       anisotropic_scaling=False)
    mapfit = mapm.fit(S)

    tau = 1 / (4 * np.pi ** 2)

    # ground truth indices estimated from the DTI tensor
    rtpp_gt = 1. / (2 * np.sqrt(np.pi * l1 * tau))
    rtap_gt = (
        1. / (2 * np.sqrt(np.pi * l2 * tau)) * 1. /
        (2 * np.sqrt(np.pi * l3 * tau))
    )
    rtop_gt = rtpp_gt * rtap_gt
    msd_gt = 2 * (l1 + l2 + l3) * tau
    qiv_gt = (
        (64 * np.pi ** (7 / 2.) * (l1 * l2 * l3 * tau ** 3) ** (3 / 2.)) /
        ((l2 * l3 + l1 * (l2 + l3)) * tau ** 2)
    )

    assert_almost_equal(mapfit.rtap(), rtap_gt, 5)
    assert_almost_equal(mapfit.rtpp(), rtpp_gt, 5)
    assert_almost_equal(mapfit.rtop(), rtop_gt, 4)
    assert_almost_equal(mapfit.msd(), msd_gt, 5)
    assert_almost_equal(mapfit.qiv(), qiv_gt, 5)

if __name__ == '__main__':
    run_module_suite()
