import platform
import time
from math import factorial

from scipy.special import gamma
import scipy.integrate as integrate
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal,
                           assert_array_almost_equal,
                           assert_equal, assert_,
                           assert_raises)
import pytest
from dipy.core.sphere_stats import angular_similarity
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.data import get_gtab_taiwan_dsi, default_sphere
from dipy.direction.peaks import peak_directions
from dipy.reconst.mapmri import MapmriModel, mapmri_index_matrix
from dipy.reconst import dti, mapmri
from dipy.reconst.odf import gfa
from dipy.reconst.tests.test_dsi import sticks_and_ball_dummies
from dipy.reconst.shm import sh_to_sf, descoteaux07_legacy_msg
from dipy.sims.voxel import (multi_tensor, multi_tensor_pdf, add_noise,
                             single_tensor, cylinders_and_ball_soderman)
from dipy.testing.decorators import set_random_number_generator


def int_func(n):
    f = np.sqrt(2) * factorial(n) / float(((gamma(1 + n / 2.0)) *
                                          np.sqrt(2**(n + 1) * factorial(n))))
    return f


def generate_signal_crossing(gtab, lambda1, lambda2, lambda3, angle2=60):
    mevals = np.array(([lambda1, lambda2, lambda3],
                       [lambda1, lambda2, lambda3]))
    angl = [(0, 0), (angle2, 0)]
    S, sticks = multi_tensor(gtab, mevals, S0=100.0, angles=angl,
                             fractions=[50, 50], snr=None)
    return S, sticks


def test_orthogonality_basis_functions():
    # numerical integration parameters
    diffusivity = 0.0015
    qmin = 0
    qmax = 1000

    int1 = integrate.quad(lambda x:
                          np.real(mapmri.mapmri_phi_1d(0, x, diffusivity)) *
                          np.real(mapmri.mapmri_phi_1d(2, x, diffusivity)),
                          qmin, qmax)[0]
    int2 = integrate.quad(lambda x:
                          np.real(mapmri.mapmri_phi_1d(2, x, diffusivity)) *
                          np.real(mapmri.mapmri_phi_1d(4, x, diffusivity)),
                          qmin, qmax)[0]
    int3 = integrate.quad(lambda x:
                          np.real(mapmri.mapmri_phi_1d(4, x, diffusivity)) *
                          np.real(mapmri.mapmri_phi_1d(6, x, diffusivity)),
                          qmin, qmax)[0]
    int4 = integrate.quad(lambda x:
                          np.real(mapmri.mapmri_phi_1d(6, x, diffusivity)) *
                          np.real(mapmri.mapmri_phi_1d(8, x, diffusivity)),
                          qmin, qmax)[0]

    # checking for first 5 basis functions if they are indeed orthogonal
    assert_almost_equal(int1, 0.)
    assert_almost_equal(int2, 0.)
    assert_almost_equal(int3, 0.)
    assert_almost_equal(int4, 0.)

    # do the same for the isotropic mapmri basis functions
    # we already know the spherical harmonics are orthonormal
    # only check j>0, l=0 basis functions

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                category=integrate.IntegrationWarning)
        int1 = integrate.quad(
            lambda q: mapmri.mapmri_isotropic_radial_signal_basis(
                1, 0, diffusivity, q) *
            mapmri.mapmri_isotropic_radial_signal_basis(
                2, 0, diffusivity, q) * q ** 2, qmin, qmax)[0]
        int2 = integrate.quad(
            lambda q: mapmri.mapmri_isotropic_radial_signal_basis(
                2, 0, diffusivity, q) *
            mapmri.mapmri_isotropic_radial_signal_basis(
                3, 0, diffusivity, q) * q ** 2, qmin, qmax)[0]
        int3 = integrate.quad(
            lambda q: mapmri.mapmri_isotropic_radial_signal_basis(
                3, 0, diffusivity, q) *
            mapmri.mapmri_isotropic_radial_signal_basis(
                4, 0, diffusivity, q) * q ** 2, qmin, qmax)[0]
        int4 = integrate.quad(
            lambda q: mapmri.mapmri_isotropic_radial_signal_basis(
                4, 0, diffusivity, q) *
            mapmri.mapmri_isotropic_radial_signal_basis(
                5, 0, diffusivity, q) * q ** 2, qmin, qmax)[0]

    # checking for first 5 basis functions if they are indeed orthogonal
    assert_almost_equal(int1, 0.)
    assert_almost_equal(int2, 0.)
    assert_almost_equal(int3, 0.)
    assert_almost_equal(int4, 0.)


def test_mapmri_number_of_coefficients(radial_order=6):
    indices = mapmri_index_matrix(radial_order)
    n_c = indices.shape[0]
    F = radial_order / 2
    n_gt = np.round(1 / 6.0 * (F + 1) * (F + 2) * (4 * F + 3))
    assert_equal(n_c, n_gt)


def test_mapmri_initialize_radial_error():
    """
    Test initialization conditions
    """
    gtab = get_gtab_taiwan_dsi()
    # No negative radial_order allowed
    assert_raises(ValueError, MapmriModel, gtab, radial_order=-1)
    # No odd radial order allowed:
    assert_raises(ValueError, MapmriModel, gtab, radial_order=3)


def test_mapmri_initialize_gcv():
    """
    Test initialization conditions
    """
    gtab = get_gtab_taiwan_dsi()
    # When string is provided it has to be "GCV"
    assert_raises(ValueError, MapmriModel, gtab, laplacian_weighting="notGCV")


def test_mapmri_initialize_pos_radius():
    """
    Test initialization conditions
    """
    gtab = get_gtab_taiwan_dsi()
    # When string is provided it has to be "adaptive"
    ErrorType = ImportError if not mapmri.have_cvxpy else ValueError
    assert_raises(ErrorType, MapmriModel, gtab, positivity_constraint=True,
                  pos_radius="notadaptive")
    # When a number is provided it has to be positive
    assert_raises(ErrorType, MapmriModel, gtab, positivity_constraint=True,
                  pos_radius=-1)


def test_mapmri_signal_fitting(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S, _ = generate_signal_crossing(gtab, l1, l2, l3)

    mapm = MapmriModel(gtab, radial_order=radial_order,
                       laplacian_weighting=0.02)
    mapfit = mapm.fit(S)
    S_reconst = mapfit.predict(gtab, 1.0)

    # test the signal reconstruction
    S = S / S[0]
    nmse_signal = np.sqrt(np.sum((S - S_reconst) ** 2)) / (S.sum())
    assert_almost_equal(nmse_signal, 0.0, 3)

    # Test with multidimensional signals:
    mapm = MapmriModel(gtab, radial_order=radial_order,
                       laplacian_weighting=0.02)
    # Each voxel is identical:
    mapfit = mapm.fit(S[:, None, None].T * np.ones((3, 3, 3, S.shape[0])))

    # Predict back with an array of ones or a single value:
    for S0 in [S[0], np.ones((3, 3, 3, 203))]:
        S_reconst = mapfit.predict(gtab, S0=S0)
        # test the signal reconstruction for one voxel:
        nmse_signal = (np.sqrt(np.sum((S - S_reconst[0, 0, 0]) ** 2)) /
                       (S.sum()))
        assert_almost_equal(nmse_signal, 0.0, 3)

    # do the same for isotropic implementation
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        mapm = MapmriModel(gtab, radial_order=radial_order,
                           laplacian_weighting=0.0001,
                           anisotropic_scaling=False)
    mapfit = mapm.fit(S)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        S_reconst = mapfit.predict(gtab, 1.0)

    # test the signal reconstruction
    S = S / S[0]
    nmse_signal = np.sqrt(np.sum((S - S_reconst) ** 2)) / (S.sum())
    assert_almost_equal(nmse_signal, 0.0, 3)

    # do the same without the positivity constraint:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        mapm = MapmriModel(gtab, radial_order=radial_order,
                           laplacian_weighting=0.0001,
                           positivity_constraint=False,
                           anisotropic_scaling=False)

    mapfit = mapm.fit(S)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        S_reconst = mapfit.predict(gtab, 1.0)

    # test the signal reconstruction
    S = S / S[0]
    nmse_signal = np.sqrt(np.sum((S - S_reconst) ** 2)) / (S.sum())
    assert_almost_equal(nmse_signal, 0.0, 3)

    # Repeat with a gtab with big_delta and small_delta:
    gtab.big_delta = 5
    gtab.small_delta = 3
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        mapm = MapmriModel(gtab, radial_order=radial_order,
                           laplacian_weighting=0.0001,
                           positivity_constraint=False,
                           anisotropic_scaling=False)

    mapfit = mapm.fit(S)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        S_reconst = mapfit.predict(gtab, 1.0)

    # test the signal reconstruction
    S = S / S[0]
    nmse_signal = np.sqrt(np.sum((S - S_reconst) ** 2)) / (S.sum())
    assert_almost_equal(nmse_signal, 0.0, 3)

    if mapmri.have_cvxpy:
        # Positivity constraint and anisotropic scaling:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            mapm = MapmriModel(gtab, radial_order=radial_order,
                               laplacian_weighting=0.0001,
                               positivity_constraint=True,
                               anisotropic_scaling=False,
                               pos_radius=2)

        mapfit = mapm.fit(S)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            S_reconst = mapfit.predict(gtab, 1.0)

        # test the signal reconstruction
        S = S / S[0]
        nmse_signal = np.sqrt(np.sum((S - S_reconst) ** 2)) / (S.sum())
        assert_almost_equal(nmse_signal, 0.0, 3)

        # Positivity constraint and anisotropic scaling:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            mapm = MapmriModel(gtab, radial_order=radial_order,
                               laplacian_weighting=None,
                               positivity_constraint=True,
                               anisotropic_scaling=False,
                               pos_radius=2)

        mapfit = mapm.fit(S)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            S_reconst = mapfit.predict(gtab, 1.0)

        # test the signal reconstruction
        S = S / S[0]
        nmse_signal = np.sqrt(np.sum((S - S_reconst) ** 2)) / (S.sum())
        assert_almost_equal(nmse_signal, 0.0, 2)


@set_random_number_generator(1234)
def test_mapmri_isotropic_static_scale_factor(radial_order=6, rng=None):
    gtab = get_gtab_taiwan_dsi()
    D = 0.7e-3
    tau = 1 / (4 * np.pi ** 2)
    mu = np.sqrt(D * 2 * tau)

    l1, l2, l3 = [D, D, D]
    S = single_tensor(gtab, evals=np.r_[l1, l2, l3], rng=rng)
    S_array = np.tile(S, (5, 1))

    stat_weight = 0.1
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        mapm_scale_stat_reg_stat = MapmriModel(gtab,
                                               radial_order=radial_order,
                                               anisotropic_scaling=False,
                                               dti_scale_estimation=False,
                                               static_diffusivity=D,
                                               laplacian_regularization=True,
                                               laplacian_weighting=stat_weight)
        mapm_scale_adapt_reg_stat = MapmriModel(
            gtab,
            radial_order=radial_order,
            anisotropic_scaling=False,
            dti_scale_estimation=True,
            laplacian_regularization=True,
            laplacian_weighting=stat_weight)

    start = time.time()
    mapf_scale_stat_reg_stat = mapm_scale_stat_reg_stat.fit(S_array)
    time_scale_stat_reg_stat = time.time() - start

    start = time.time()
    mapf_scale_adapt_reg_stat = mapm_scale_adapt_reg_stat.fit(S_array)
    time_scale_adapt_reg_stat = time.time() - start

    # test if indeed the scale factor is fixed now
    assert_equal(np.all(mapf_scale_stat_reg_stat.mu == mu),
                 True)

    # test if computation time is shorter (except on Windows):
    if not platform.system() == "Windows":
        assert_equal(time_scale_stat_reg_stat < time_scale_adapt_reg_stat,
                     True,
                     "mapf_scale_stat_reg_stat ({0}s) slower than "
                     "mapf_scale_adapt_reg_stat ({1}s). It should be the"
                     " opposite.".format(time_scale_stat_reg_stat,
                                         time_scale_adapt_reg_stat))

    # check if the fitted signal is the same
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        assert_almost_equal(mapf_scale_stat_reg_stat.fitted_signal(),
                            mapf_scale_adapt_reg_stat.fitted_signal())


def test_mapmri_signal_fitting_over_radial_order(order_max=8):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0012, 0.0003, 0.0003]
    S, _ = generate_signal_crossing(gtab, l1, l2, l3, angle2=60)

    # take radial order 0, 4 and 8
    orders = [0, 4, 8]
    error_array = np.zeros(len(orders))

    for i, order in enumerate(orders):
        mapm = MapmriModel(gtab, radial_order=order,
                           laplacian_regularization=False)
        mapfit = mapm.fit(S)
        S_reconst = mapfit.predict(gtab, 100.0)
        error_array[i] = np.mean((S - S_reconst) ** 2)
    # check if the fitting error decreases as radial order increases
    assert_equal(np.diff(error_array) < 0., True)


def test_mapmri_pdf_integral_unity(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S, _ = generate_signal_crossing(gtab, l1, l2, l3)
    sphere = default_sphere
    # test MAPMRI fitting

    mapm = MapmriModel(gtab, radial_order=radial_order,
                       laplacian_weighting=0.02)
    mapfit = mapm.fit(S)
    c_map = mapfit.mapmri_coeff

    # test if the analytical integral of the pdf is equal to one
    indices = mapmri_index_matrix(radial_order)
    integral = 0
    for i in range(indices.shape[0]):
        n1, n2, n3 = indices[i]
        integral += c_map[i] * int_func(n1) * int_func(n2) * int_func(n3)

    assert_almost_equal(integral, 1.0, 3)

    # test if numerical integral of odf is equal to one
    odf = mapfit.odf(sphere, s=0)
    odf_sum = odf.sum() / sphere.vertices.shape[0] * (4 * np.pi)
    assert_almost_equal(odf_sum, 1.0, 2)

    # do the same for isotropic implementation
    radius_max = 0.04  # 40 microns
    gridsize = 17
    r_points = mapmri.create_rspace(gridsize, radius_max)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        mapm = MapmriModel(gtab, radial_order=radial_order,
                           laplacian_weighting=0.02,
                           anisotropic_scaling=False)
    mapfit = mapm.fit(S)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        pdf = mapfit.pdf(r_points)
    pdf[r_points[:, 2] == 0.] /= 2  # for antipodal symmetry on z-plane

    point_volume = (radius_max / (gridsize // 2)) ** 3
    integral = pdf.sum() * point_volume * 2
    assert_almost_equal(integral, 1.0, 3)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        odf = mapfit.odf(sphere, s=0)
    odf_sum = odf.sum() / sphere.vertices.shape[0] * (4 * np.pi)
    assert_almost_equal(odf_sum, 1.0, 2)


def test_mapmri_compare_fitted_pdf_with_multi_tensor(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S, _ = generate_signal_crossing(gtab, l1, l2, l3)

    radius_max = 0.02  # 40 microns
    gridsize = 10
    r_points = mapmri.create_rspace(gridsize, radius_max)

    # test MAPMRI fitting
    mapm = MapmriModel(gtab, radial_order=radial_order,
                       laplacian_weighting=0.0001)
    mapfit = mapm.fit(S)

    # compare the mapmri pdf with the ground truth multi_tensor pdf

    mevals = np.array(([l1, l2, l3],
                       [l1, l2, l3]))
    angl = [(0, 0), (60, 0)]
    pdf_mt = multi_tensor_pdf(r_points, mevals=mevals,
                              angles=angl, fractions=[50, 50])
    pdf_map = mapfit.pdf(r_points)

    nmse_pdf = np.sqrt(np.sum((pdf_mt - pdf_map) ** 2)) / (pdf_mt.sum())
    assert_almost_equal(nmse_pdf, 0.0, 2)


def test_mapmri_metrics_anisotropic(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S, _ = generate_signal_crossing(gtab, l1, l2, l3, angle2=0)

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
    with warnings.catch_warnings(record=True) as w:
        ng = mapfit.ng()
        ng_parallel = mapfit.ng_parallel()
        ng_perpendicular = mapfit.ng_perpendicular()
        assert_equal(len(w), 3)
        for l_w in w:
            assert_(issubclass(l_w.category, UserWarning))
            assert_("model bval_threshold must be lower than 2000".lower()
                    in str(l_w.message).lower())

    assert_almost_equal(ng, 0., 5)
    assert_almost_equal(ng_parallel, 0., 5)
    assert_almost_equal(ng_perpendicular, 0., 5)
    assert_almost_equal(mapfit.msd(), msd_gt, 5)
    assert_almost_equal(mapfit.qiv(), qiv_gt, 5)


def test_mapmri_metrics_isotropic(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0003, 0.0003, 0.0003]  # isotropic diffusivities
    S = single_tensor(gtab, evals=np.r_[l1, l2, l3])

    # test MAPMRI q-space indices

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
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

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        assert_almost_equal(mapfit.rtap(), rtap_gt, 5)
        assert_almost_equal(mapfit.rtpp(), rtpp_gt, 5)
    assert_almost_equal(mapfit.rtop(), rtop_gt, 4)
    assert_almost_equal(mapfit.msd(), msd_gt, 5)
    assert_almost_equal(mapfit.qiv(), qiv_gt, 5)


def test_mapmri_laplacian_anisotropic(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = single_tensor(gtab, evals=np.r_[l1, l2, l3])

    mapm = MapmriModel(gtab, radial_order=radial_order,
                       laplacian_regularization=False)
    mapfit = mapm.fit(S)
    tau = 1 / (4 * np.pi ** 2)

    # ground truth norm of laplacian of tensor
    norm_of_laplacian_gt = (
        (3 * (l1 ** 2 + l2 ** 2 + l3 ** 2) +
         2 * l2 * l3 + 2 * l1 * (l2 + l3)) * (np.pi ** (5 / 2.) * tau) /
        (np.sqrt(2 * l1 * l2 * l3 * tau))
        )

    # check if estimated laplacian corresponds with ground truth
    laplacian_matrix = mapmri.mapmri_laplacian_reg_matrix(
        mapm.ind_mat, mapfit.mu, mapm.S_mat,
        mapm.T_mat, mapm.U_mat)

    coef = mapfit._mapmri_coef
    norm_of_laplacian = np.dot(np.dot(coef, laplacian_matrix), coef)

    assert_almost_equal(norm_of_laplacian, norm_of_laplacian_gt)


def test_mapmri_laplacian_isotropic(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0003, 0.0003, 0.0003]  # isotropic diffusivities
    S = single_tensor(gtab, evals=np.r_[l1, l2, l3])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        mapm = MapmriModel(gtab, radial_order=radial_order,
                           laplacian_regularization=False,
                           anisotropic_scaling=False)
    mapfit = mapm.fit(S)
    tau = 1 / (4 * np.pi ** 2)

    # ground truth norm of laplacian of tensor
    norm_of_laplacian_gt = (
        (3 * (l1 ** 2 + l2 ** 2 + l3 ** 2) +
         2 * l2 * l3 + 2 * l1 * (l2 + l3)) * (np.pi ** (5 / 2.) * tau) /
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
    S, _ = generate_signal_crossing(gtab, l1, l2, l3, angle2=60)
    gridsize = 17
    radius_max = 0.02
    r_points = mapmri.create_rspace(gridsize, radius_max)

    tenmodel = dti.TensorModel(gtab)
    evals = tenmodel.fit(S).evals
    tau = 1 / (4 * np.pi ** 2)

    # estimate isotropic scale factor
    u0 = mapmri.isotropic_scale_factor(evals * 2 * tau)
    mu = np.array([u0, u0, u0])

    qvals = np.sqrt(gtab.bvals / tau) / (2 * np.pi)
    q = gtab.bvecs * qvals[:, None]

    M_aniso = mapmri.mapmri_phi_matrix(radial_order, mu, q)
    K_aniso = mapmri.mapmri_psi_matrix(radial_order, mu, r_points)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        M_iso = mapmri.mapmri_isotropic_phi_matrix(radial_order, u0, q)
        K_iso = mapmri.mapmri_isotropic_psi_matrix(radial_order, u0, r_points)

    coef_aniso = np.dot(np.linalg.pinv(M_aniso), S)
    coef_iso = np.dot(np.linalg.pinv(M_iso), S)
    # test if anisotropic and isotropic implementation produce equal results
    # if the same isotropic scale factors are used
    s_fitted_aniso = np.dot(M_aniso, coef_aniso)
    s_fitted_iso = np.dot(M_iso, coef_iso)
    assert_array_almost_equal(s_fitted_aniso, s_fitted_iso)

    # the same test for the PDF
    pdf_fitted_aniso = np.dot(K_aniso, coef_aniso)
    pdf_fitted_iso = np.dot(K_iso, coef_iso)

    assert_array_almost_equal(pdf_fitted_aniso / pdf_fitted_iso,
                              np.ones_like(pdf_fitted_aniso), 3)

    # test if the implemented version also produces the same result
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        mapm = MapmriModel(gtab, radial_order=radial_order,
                           laplacian_regularization=False,
                           anisotropic_scaling=False)
        s_fitted_implemented_isotropic = mapm.fit(S).fitted_signal()

    # normalize non-implemented fitted signal with b0 value
    s_fitted_aniso_norm = s_fitted_aniso / s_fitted_aniso.max()

    assert_array_almost_equal(s_fitted_aniso_norm,
                              s_fitted_implemented_isotropic)

    # test if norm of signal laplacians are the same
    laplacian_matrix_iso = mapmri.mapmri_isotropic_laplacian_reg_matrix(
                           radial_order, mu[0])
    ind_mat = mapmri.mapmri_index_matrix(radial_order)
    S_mat, T_mat, U_mat = mapmri.mapmri_STU_reg_matrices(radial_order)
    laplacian_matrix_aniso = mapmri.mapmri_laplacian_reg_matrix(
        ind_mat, mu, S_mat, T_mat, U_mat)

    norm_aniso = np.dot(coef_aniso, np.dot(coef_aniso, laplacian_matrix_aniso))
    norm_iso = np.dot(coef_iso, np.dot(coef_iso, laplacian_matrix_iso))
    assert_almost_equal(norm_iso, norm_aniso)


def test_mapmri_isotropic_design_matrix_separability(radial_order=6):
    gtab = get_gtab_taiwan_dsi()
    tau = 1 / (4 * np.pi ** 2)
    qvals = np.sqrt(gtab.bvals / tau) / (2 * np.pi)
    q = gtab.bvecs * qvals[:, None]
    mu = 0.0003  # random value

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        M = mapmri.mapmri_isotropic_phi_matrix(radial_order, mu, q)
        M_independent = mapmri.mapmri_isotropic_M_mu_independent(
            radial_order, q)
    M_dependent = mapmri.mapmri_isotropic_M_mu_dependent(radial_order, mu,
                                                         qvals)
    M_reconstructed = M_independent * M_dependent

    assert_array_almost_equal(M, M_reconstructed)


def test_estimate_radius_with_rtap(radius_gt=5e-3):
    gtab = get_gtab_taiwan_dsi()
    tau = 1 / (4 * np.pi ** 2)
    # we estimate the infinite diffusion time case for a perfectly reflecting
    # cylinder using the Callaghan model
    E = cylinders_and_ball_soderman(gtab, tau, radii=[radius_gt], snr=None,
                                    angles=[(0, 90)], fractions=[100])[0]

    # estimate radius using anisotropic MAP-MRI.
    mapmod = mapmri.MapmriModel(gtab, radial_order=6,
                                laplacian_regularization=True,
                                laplacian_weighting=0.01)
    mapfit = mapmod.fit(E)
    radius_estimated = np.sqrt(1 / (np.pi * mapfit.rtap()))
    assert_almost_equal(radius_estimated, radius_gt, 5)

    # estimate radius using isotropic MAP-MRI.
    # note that the radial order is higher and the precision is lower due to
    # less accurate signal extrapolation.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        mapmod = mapmri.MapmriModel(gtab, radial_order=8,
                                    laplacian_regularization=True,
                                    laplacian_weighting=0.01,
                                    anisotropic_scaling=False)
    mapfit = mapmod.fit(E)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        radius_estimated = np.sqrt(1 / (np.pi * mapfit.rtap()))
    assert_almost_equal(radius_estimated, radius_gt, 4)


@pytest.mark.skipif(not mapmri.have_cvxpy, reason="Requires CVXPY")
@set_random_number_generator(1234)
def test_positivity_constraint(radial_order=6, rng=None):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S, _ = generate_signal_crossing(gtab, l1, l2, l3, angle2=60)
    S_noise = add_noise(S, snr=20, S0=100., rng=rng)

    gridsize = 20
    max_radius = 15e-3  # 20 microns maximum radius
    r_grad = mapmri.create_rspace(gridsize, max_radius)

    # The positivity constraint does not make the pdf completely positive
    # but greatly decreases the amount of negativity in the constrained points.
    # We test if the amount of negative pdf has decreased more than 90%

    mapmod_no_constraint = MapmriModel(gtab, radial_order=radial_order,
                                       laplacian_regularization=False,
                                       positivity_constraint=False)
    mapfit_no_constraint = mapmod_no_constraint.fit(S_noise)
    pdf = mapfit_no_constraint.pdf(r_grad)
    pdf_negative_no_constraint = pdf[pdf < 0].sum()

    mapmod_constraint = MapmriModel(gtab, radial_order=radial_order,
                                    laplacian_regularization=False,
                                    positivity_constraint=True,
                                    pos_grid=gridsize,
                                    pos_radius='adaptive')
    mapfit_constraint = mapmod_constraint.fit(S_noise)
    pdf = mapfit_constraint.pdf(r_grad)
    pdf_negative_constraint = pdf[pdf < 0].sum()

    assert_equal((pdf_negative_constraint / pdf_negative_no_constraint) < 0.1,
                 True)

    # the same for isotropic scaling
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        mapmod_no_constraint = MapmriModel(gtab, radial_order=radial_order,
                                           laplacian_regularization=False,
                                           positivity_constraint=False,
                                           anisotropic_scaling=False)
    mapfit_no_constraint = mapmod_no_constraint.fit(S_noise)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        pdf = mapfit_no_constraint.pdf(r_grad)
    pdf_negative_no_constraint = pdf[pdf < 0].sum()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        mapmod_constraint = MapmriModel(gtab, radial_order=radial_order,
                                        laplacian_regularization=False,
                                        positivity_constraint=True,
                                        anisotropic_scaling=False,
                                        pos_grid=gridsize,
                                        pos_radius='adaptive')
        mapfit_constraint = mapmod_constraint.fit(S_noise)
        pdf = mapfit_constraint.pdf(r_grad)
    pdf_negative_constraint = pdf[pdf < 0].sum()

    assert_equal((pdf_negative_constraint / pdf_negative_no_constraint) < 0.1,
                 True)


@pytest.mark.skipif(not mapmri.have_cvxpy, reason="Requires CVXPY")
@set_random_number_generator(1234)
def test_plus_constraint(radial_order=6, rng=None):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S, _ = generate_signal_crossing(gtab, l1, l2, l3, angle2=60)
    S_noise = add_noise(S, snr=20, S0=100., rng=rng)

    gridsize = 50
    max_radius = 25e-3  # 25 microns maximum radius
    r_grad = mapmri.create_rspace(gridsize, max_radius)

    # The positivity constraint should make the pdf positive everywhere
    mapmod_constraint = MapmriModel(gtab, radial_order=radial_order,
                                    laplacian_regularization=False,
                                    positivity_constraint=True,
                                    global_constraints=True)
    mapfit_constraint = mapmod_constraint.fit(S_noise)
    pdf = mapfit_constraint.pdf(r_grad)
    pdf_negative_constraint = pdf[pdf < 0].sum()

    assert_equal(pdf_negative_constraint == 0.0, True)


@set_random_number_generator(1234)
def test_laplacian_regularization(radial_order=6, rng=None):
    gtab = get_gtab_taiwan_dsi()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S, _ = generate_signal_crossing(gtab, l1, l2, l3, angle2=60)
    S_noise = add_noise(S, snr=20, S0=100., rng=rng)

    weight_array = np.linspace(0, .3, 301)
    mapmod_unreg = MapmriModel(gtab, radial_order=radial_order,
                               laplacian_regularization=False,
                               laplacian_weighting=weight_array)
    mapmod_laplacian_array = MapmriModel(gtab, radial_order=radial_order,
                                         laplacian_regularization=True,
                                         laplacian_weighting=weight_array)
    mapmod_laplacian_gcv = MapmriModel(gtab, radial_order=radial_order,
                                       laplacian_regularization=True,
                                       laplacian_weighting="GCV")

    # test the Generalized Cross Validation
    # test if GCV gives very low if there is no noise
    mapfit_laplacian_array = mapmod_laplacian_array.fit(S)
    assert_equal(mapfit_laplacian_array.lopt < 0.01, True)

    # test if GCV gives higher values if there is noise
    mapfit_laplacian_array = mapmod_laplacian_array.fit(S_noise)
    lopt_array = mapfit_laplacian_array.lopt
    assert_equal(lopt_array > 0.01, True)

    # test if continuous GCV gives the same the one based on an array
    mapfit_laplacian_gcv = mapmod_laplacian_gcv.fit(S_noise)
    lopt_gcv = mapfit_laplacian_gcv.lopt
    assert_almost_equal(lopt_array, lopt_gcv, 2)

    # test if laplacian reduced the norm of the laplacian in the reconstruction
    mu = mapfit_laplacian_gcv.mu
    laplacian_matrix = mapmri.mapmri_laplacian_reg_matrix(
        mapmod_laplacian_gcv.ind_mat, mu, mapmod_laplacian_gcv.S_mat,
        mapmod_laplacian_gcv.T_mat, mapmod_laplacian_gcv.U_mat)

    coef_unreg = mapmod_unreg.fit(S_noise)._mapmri_coef
    coef_laplacian = mapfit_laplacian_gcv._mapmri_coef

    laplacian_norm_unreg = np.dot(
        coef_unreg, np.dot(coef_unreg, laplacian_matrix))
    laplacian_norm_laplacian = np.dot(
        coef_laplacian, np.dot(coef_laplacian, laplacian_matrix))

    assert_equal(laplacian_norm_laplacian < laplacian_norm_unreg, True)

    # the same for isotropic scaling
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        mapmod_unreg = MapmriModel(gtab, radial_order=radial_order,
                                   laplacian_regularization=False,
                                   laplacian_weighting=weight_array,
                                   anisotropic_scaling=False)
        mapmod_laplacian_array = MapmriModel(gtab, radial_order=radial_order,
                                             laplacian_regularization=True,
                                             laplacian_weighting=weight_array,
                                             anisotropic_scaling=False)
        mapmod_laplacian_gcv = MapmriModel(gtab, radial_order=radial_order,
                                           laplacian_regularization=True,
                                           laplacian_weighting="GCV",
                                           anisotropic_scaling=False)

    # test the Generalized Cross Validation
    # test if GCV gives zero if there is no noise
    mapfit_laplacian_array = mapmod_laplacian_array.fit(S)
    assert_equal(mapfit_laplacian_array.lopt < 0.01, True)

    # test if GCV gives higher values if there is noise
    mapfit_laplacian_array = mapmod_laplacian_array.fit(S_noise)
    lopt_array = mapfit_laplacian_array.lopt
    assert_equal(lopt_array > 0.01, True)

    # test if continuous GCV gives the same the one based on an array
    mapfit_laplacian_gcv = mapmod_laplacian_gcv.fit(S_noise)
    lopt_gcv = mapfit_laplacian_gcv.lopt
    assert_almost_equal(lopt_array, lopt_gcv, 2)

    # test if laplacian reduced the norm of the laplacian in the reconstruction
    mu = mapfit_laplacian_gcv.mu
    laplacian_matrix = mapmri.mapmri_isotropic_laplacian_reg_matrix(
        radial_order, mu[0])

    coef_unreg = mapmod_unreg.fit(S_noise)._mapmri_coef
    coef_laplacian = mapfit_laplacian_gcv._mapmri_coef

    laplacian_norm_unreg = np.dot(
        coef_unreg, np.dot(coef_unreg, laplacian_matrix))
    laplacian_norm_laplacian = np.dot(
        coef_laplacian, np.dot(coef_laplacian, laplacian_matrix))

    assert_equal(laplacian_norm_laplacian < laplacian_norm_unreg, True)


def test_mapmri_odf(radial_order=6):
    gtab = get_gtab_taiwan_dsi()

    # load repulsion 724 sphere
    sphere = default_sphere

    # load icosahedron sphere
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    data, golden_directions = generate_signal_crossing(gtab, l1, l2, l3,
                                                       angle2=90)
    mapmod = MapmriModel(gtab, radial_order=radial_order,
                         laplacian_regularization=True,
                         laplacian_weighting=0.01)
    # repulsion724
    sphere2 = create_unit_sphere(5)
    mapfit = mapmod.fit(data)
    odf = mapfit.odf(sphere)

    directions, _, _ = peak_directions(odf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, golden_directions), 2, 1)

    # 5 subdivisions
    odf = mapfit.odf(sphere2)
    directions, _, _ = peak_directions(odf, sphere2, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, golden_directions), 2, 1)

    sb_dummies = sticks_and_ball_dummies(gtab)
    for sbd in sb_dummies:
        data, golden_directions = sb_dummies[sbd]
        asmfit = mapmod.fit(data)
        odf = asmfit.odf(sphere2)
        directions, _, _ = peak_directions(odf, sphere2, .35, 25)
        if len(directions) <= 3:
            assert_equal(len(directions), len(golden_directions))
        if len(directions) > 3:
            assert_equal(gfa(odf) < 0.1, True)

    # for the isotropic implementation check if the odf spherical harmonics
    # actually represent the discrete sphere function.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        mapmod = MapmriModel(gtab, radial_order=radial_order,
                             laplacian_regularization=True,
                             laplacian_weighting=0.01,
                             anisotropic_scaling=False)
    mapfit = mapmod.fit(data)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        odf = mapfit.odf(sphere)
    odf_sh = mapfit.odf_sh()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        odf_from_sh = sh_to_sf(odf_sh, sphere, radial_order, basis_type=None,
                               legacy=True)
    assert_almost_equal(odf, odf_from_sh, 10)
