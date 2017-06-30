import numpy as np
from dipy.data import get_gtab_taiwan_dsi
from numpy.testing import (assert_almost_equal,
                           assert_array_almost_equal,
                           assert_equal,
                           run_module_suite)
from dipy.reconst import qtdmri, mapmri
from dipy.sims.voxel import MultiTensor
from dipy.data import get_sphere
from dipy.sims.voxel import add_noise
import scipy.integrate as integrate
from dipy.core.gradients import gradient_table_from_qvals_bvecs


def generate_gtab4D(number_of_tau_shells=4, delta=0.01):
    """Generates testing gradient table for 4D qt-dMRI scheme"""
    gtab = get_gtab_taiwan_dsi()
    qvals = np.tile(gtab.bvals / 100., number_of_tau_shells)
    bvecs = np.tile(gtab.bvecs, (number_of_tau_shells, 1))
    pulse_separation = []
    for ps in np.linspace(0.02, 0.05, number_of_tau_shells):
        pulse_separation = np.append(pulse_separation,
                                     np.tile(ps, gtab.bvals.shape[0]))
    pulse_duration = np.tile(delta, qvals.shape[0])
    gtab_4d = gradient_table_from_qvals_bvecs(qvals=qvals, bvecs=bvecs,
                                              big_delta=pulse_separation,
                                              small_delta=pulse_duration)
    return gtab_4d


def generate_signal_crossing(gtab, lambda1, lambda2, lambda3, angle2=60):
    mevals = np.array(([lambda1, lambda2, lambda3],
                       [lambda1, lambda2, lambda3]))
    angl = [(0, 0), (angle2, 0)]
    S, sticks = MultiTensor(gtab, mevals, S0=1.0, angles=angl,
                            fractions=[50, 50], snr=None)
    return S


def test_input_parameters():
    gtab_4d = generate_gtab4D()
    try:
        qtdmri.QtdmriModel(gtab_4d, radial_order=3)
        assert_equal(True, False)
    except ValueError:
        print ('uneven radial_order is caught')

    try:
        qtdmri.QtdmriModel(gtab_4d, radial_order=-1)
        assert_equal(True, False)
    except ValueError:
        print ('negative radial_order is caught')

    try:
        qtdmri.QtdmriModel(gtab_4d, time_order=-1)
        assert_equal(True, False)
    except ValueError:
        print ('negative time_order is caught')

    try:
        qtdmri.QtdmriModel(gtab_4d, laplacian_regularization='test')
        assert_equal(True, False)
    except ValueError:
        print ('non-bool laplacian_regularization is caught.')

    try:
        qtdmri.QtdmriModel(gtab_4d, laplacian_regularization=True,
                           laplacian_weighting='test')
        assert_equal(True, False)
    except ValueError:
        print ('non-"GCV" string for laplacian_weighting is caught.')

    try:
        qtdmri.QtdmriModel(gtab_4d, laplacian_regularization=True,
                           laplacian_weighting=-1.)
        assert_equal(True, False)
    except ValueError:
        print ('negative laplacian_weighting is caught.')

    try:
        qtdmri.QtdmriModel(gtab_4d, l1_regularization='test')
        assert_equal(True, False)
    except ValueError:
        print ('non-bool for l1_weighting is caught.')

    try:
        qtdmri.QtdmriModel(gtab_4d, l1_regularization=True,
                           l1_weighting='test')
        assert_equal(True, False)
    except ValueError:
        print ('non-"CV" string for laplacian_weighting is caught.')

    try:
        qtdmri.QtdmriModel(gtab_4d, l1_regularization=True,
                           l1_weighting=-1.)
        assert_equal(True, False)
    except ValueError:
        print ('negative l1_weighting is caught.')

    try:
        qtdmri.QtdmriModel(gtab_4d, cartesian='test')
        assert_equal(True, False)
    except ValueError:
        print ('non-bool cartesian is caught.')

    try:
        qtdmri.QtdmriModel(gtab_4d, anisotropic_scaling='test')
        assert_equal(True, False)
    except ValueError:
        print ('non-bool anisotropic_scaling is caught.')

    try:
        qtdmri.QtdmriModel(gtab_4d, constrain_q0='test')
        assert_equal(True, False)
    except ValueError:
        print ('non-bool constrain_q0 is caught.')

    try:
        qtdmri.QtdmriModel(gtab_4d, bval_threshold=-1)
        assert_equal(True, False)
    except ValueError:
        print ('negative bval_threshold is caught.')

    try:
        qtdmri.QtdmriModel(gtab_4d, eigenvalue_threshold=-1)
        assert_equal(True, False)
    except ValueError:
        print ('negative eigenvalue_threshold is caught.')

    try:
        qtdmri.QtdmriModel(gtab_4d, cvxpy_solver='test')
        assert_equal(True, False)
    except ValueError:
        print ('unavailable cvxpy solver is caught.')


def test_orthogonality_temporal_basis_functions():
    # numerical integration parameters
    ut = 10
    tmin = 0
    tmax = 100

    int1 = integrate.quad(lambda t:
                          qtdmri.temporal_basis(1, ut, t) *
                          qtdmri.temporal_basis(2, ut, t), tmin, tmax)
    int2 = integrate.quad(lambda t:
                          qtdmri.temporal_basis(2, ut, t) *
                          qtdmri.temporal_basis(3, ut, t), tmin, tmax)
    int3 = integrate.quad(lambda t:
                          qtdmri.temporal_basis(3, ut, t) *
                          qtdmri.temporal_basis(4, ut, t), tmin, tmax)
    int4 = integrate.quad(lambda t:
                          qtdmri.temporal_basis(4, ut, t) *
                          qtdmri.temporal_basis(5, ut, t), tmin, tmax)

    assert_almost_equal(int1, 0.)
    assert_almost_equal(int2, 0.)
    assert_almost_equal(int3, 0.)
    assert_almost_equal(int4, 0.)


def test_normalization_time():
    ut = 10
    tmin = 0
    tmax = 100

    int0 = integrate.quad(lambda t:
                          qtdmri.qtdmri_temporal_normalization(ut) ** 2 *
                          qtdmri.temporal_basis(0, ut, t) *
                          qtdmri.temporal_basis(0, ut, t), tmin, tmax)[0]
    int1 = integrate.quad(lambda t:
                          qtdmri.qtdmri_temporal_normalization(ut) ** 2 *
                          qtdmri.temporal_basis(1, ut, t) *
                          qtdmri.temporal_basis(1, ut, t), tmin, tmax)[0]
    int2 = integrate.quad(lambda t:
                          qtdmri.qtdmri_temporal_normalization(ut) ** 2 *
                          qtdmri.temporal_basis(2, ut, t) *
                          qtdmri.temporal_basis(2, ut, t), tmin, tmax)[0]

    assert_almost_equal(int0, 1.)
    assert_almost_equal(int1, 1.)
    assert_almost_equal(int2, 1.)


def test_anisotropic_isotropic_equivalence(radial_order=4, time_order=2):
    # generate qt-scheme and arbitary synthetic crossing data.
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)

    # initialize both cartesian and spherical models without any kind of
    # regularization
    qtdmri_mod_aniso = qtdmri.QtdmriModel(gtab_4d, radial_order=radial_order,
                                          time_order=time_order,
                                          cartesian=True,
                                          anisotropic_scaling=False)
    qtdmri_mod_iso = qtdmri.QtdmriModel(gtab_4d, radial_order=radial_order,
                                        time_order=time_order,
                                        cartesian=False,
                                        anisotropic_scaling=False)

    # both implementations fit the same signal
    qtdmri_fit_cart = qtdmri_mod_aniso.fit(S)
    qtdmri_fit_sphere = qtdmri_mod_iso.fit(S)

    # same signal fit
    assert_array_almost_equal(qtdmri_fit_cart.fitted_signal(),
                              qtdmri_fit_sphere.fitted_signal())

    # same PDF reconstruction
    rt_grid = qtdmri.create_rt_space_grid(5, 20e-3, 5, 0.02, .05)
    pdf_aniso = qtdmri_fit_cart.pdf(rt_grid)
    pdf_iso = qtdmri_fit_sphere.pdf(rt_grid)
    assert_array_almost_equal(pdf_aniso / pdf_aniso.max(),
                              pdf_iso / pdf_aniso.max())

    # same norm of the laplacian
    norm_laplacian_aniso = qtdmri_fit_cart.norm_of_laplacian_signal()
    norm_laplacian_iso = qtdmri_fit_sphere.norm_of_laplacian_signal()
    assert_almost_equal(norm_laplacian_aniso / norm_laplacian_aniso,
                        norm_laplacian_iso / norm_laplacian_aniso)

    # all q-space index is the same for arbitrary tau
    tau = 0.02
    assert_almost_equal(qtdmri_fit_cart.rtop(tau), qtdmri_fit_sphere.rtop(tau))
    assert_almost_equal(qtdmri_fit_cart.rtap(tau), qtdmri_fit_sphere.rtap(tau))
    assert_almost_equal(qtdmri_fit_cart.rtpp(tau), qtdmri_fit_sphere.rtpp(tau))
    assert_almost_equal(qtdmri_fit_cart.msd(tau), qtdmri_fit_sphere.msd(tau))
    assert_almost_equal(qtdmri_fit_cart.qiv(tau), qtdmri_fit_sphere.qiv(tau))

    # ODF estimation is the same
    sphere = get_sphere()
    assert_array_almost_equal(qtdmri_fit_cart.odf(sphere, tau, s=0),
                              qtdmri_fit_sphere.odf(sphere, tau, s=0))


def test_cartesian_normalization(radial_order=4, time_order=2):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)

    qtdmri_mod_aniso = qtdmri.QtdmriModel(gtab_4d, radial_order=radial_order,
                                          time_order=time_order,
                                          cartesian=True,
                                          normalization=False)
    qtdmri_mod_aniso_norm = qtdmri.QtdmriModel(gtab_4d,
                                               radial_order=radial_order,
                                               time_order=time_order,
                                               cartesian=True,
                                               normalization=True)
    qtdmri_fit_aniso = qtdmri_mod_aniso.fit(S)
    qtdmri_fit_aniso_norm = qtdmri_mod_aniso_norm.fit(S)
    assert_array_almost_equal(qtdmri_fit_aniso.fitted_signal(),
                              qtdmri_fit_aniso_norm.fitted_signal())
    rt_grid = qtdmri.create_rt_space_grid(5, 20e-3, 5, 0.02, .05)
    pdf_aniso = qtdmri_fit_aniso.pdf(rt_grid)
    pdf_aniso_norm = qtdmri_fit_aniso_norm.pdf(rt_grid)
    assert_array_almost_equal(pdf_aniso / pdf_aniso.max(),
                              pdf_aniso_norm / pdf_aniso.max())
    norm_laplacian = qtdmri_fit_aniso.norm_of_laplacian_signal()
    norm_laplacian_norm = qtdmri_fit_aniso_norm.norm_of_laplacian_signal()
    assert_array_almost_equal(norm_laplacian / norm_laplacian,
                              norm_laplacian_norm / norm_laplacian)


def test_spherical_normalization(radial_order=4, time_order=2):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)

    qtdmri_mod_aniso = qtdmri.QtdmriModel(gtab_4d, radial_order=radial_order,
                                          time_order=time_order,
                                          cartesian=False,
                                          normalization=False)
    qtdmri_mod_aniso_norm = qtdmri.QtdmriModel(gtab_4d,
                                               radial_order=radial_order,
                                               time_order=time_order,
                                               cartesian=False,
                                               normalization=True)
    qtdmri_fit = qtdmri_mod_aniso.fit(S)
    qtdmri_fit_norm = qtdmri_mod_aniso_norm.fit(S)
    assert_array_almost_equal(qtdmri_fit.fitted_signal(),
                              qtdmri_fit_norm.fitted_signal())

    rt_grid = qtdmri.create_rt_space_grid(5, 20e-3, 5, 0.02, .05)
    pdf = qtdmri_fit.pdf(rt_grid)
    pdf_norm = qtdmri_fit_norm.pdf(rt_grid)
    assert_array_almost_equal(pdf / pdf.max(),
                              pdf_norm / pdf.max())

    norm_laplacian = qtdmri_fit.norm_of_laplacian_signal()
    norm_laplacian_norm = qtdmri_fit_norm.norm_of_laplacian_signal()
    assert_array_almost_equal(norm_laplacian / norm_laplacian,
                              norm_laplacian_norm / norm_laplacian)


def test_anisotropic_reduced_MSE(radial_order=0, time_order=0):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)
    qtdmri_mod_aniso = qtdmri.QtdmriModel(gtab_4d, radial_order=radial_order,
                                          time_order=time_order,
                                          cartesian=True,
                                          anisotropic_scaling=True)
    qtdmri_mod_iso = qtdmri.QtdmriModel(gtab_4d, radial_order=radial_order,
                                        time_order=time_order,
                                        cartesian=True,
                                        anisotropic_scaling=False)
    qtdmri_fit_aniso = qtdmri_mod_aniso.fit(S)
    qtdmri_fit_iso = qtdmri_mod_iso.fit(S)
    mse_aniso = np.mean((S - qtdmri_fit_aniso.fitted_signal()) ** 2)
    mse_iso = np.mean((S - qtdmri_fit_iso.fitted_signal()) ** 2)
    assert_equal(mse_aniso < mse_iso, True)


def test_number_of_coefficients(radial_order=4, time_order=2):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)
    qtdmri_mod = qtdmri.QtdmriModel(
        gtab_4d, radial_order=radial_order, time_order=time_order)
    qtdmri_fit = qtdmri_mod.fit(S)
    number_of_coef_model = qtdmri_fit._qtdmri_coef.shape[0]
    number_of_coef_analytic = qtdmri.qtdmri_number_of_coefficients(
        radial_order, time_order
    )
    assert_equal(number_of_coef_model, number_of_coef_analytic)


def test_calling_cartesian_laplacian_with_precomputed_matrices(
        radial_order=4, time_order=2, ut=2e-3, us=np.r_[1e-3, 2e-3, 3e-3]):
    ind_mat = qtdmri.qtdmri_index_matrix(radial_order, time_order)
    part4_reg_mat_tau = qtdmri.part4_reg_matrix_tau(ind_mat, 1.)
    part23_reg_mat_tau = qtdmri.part23_reg_matrix_tau(ind_mat, 1.)
    part1_reg_mat_tau = qtdmri.part1_reg_matrix_tau(ind_mat, 1.)
    S_mat, T_mat, U_mat = mapmri.mapmri_STU_reg_matrices(radial_order)

    laplacian_matrix_precomputed = qtdmri.qtdmri_laplacian_reg_matrix(
        ind_mat, us, ut, S_mat, T_mat, U_mat,
        part1_reg_mat_tau, part23_reg_mat_tau, part4_reg_mat_tau
    )
    laplacian_matrix_regular = qtdmri.qtdmri_laplacian_reg_matrix(
        ind_mat, us, ut)
    assert_array_almost_equal(laplacian_matrix_precomputed,
                              laplacian_matrix_regular)


def test_calling_spherical_laplacian_with_precomputed_matrices(
        radial_order=4, time_order=2, ut=2e-3, us=np.r_[2e-3, 2e-3, 2e-3]):
    ind_mat = qtdmri.qtdmri_isotropic_index_matrix(radial_order, time_order)
    part4_reg_mat_tau = qtdmri.part4_reg_matrix_tau(ind_mat, 1.)
    part23_reg_mat_tau = qtdmri.part23_reg_matrix_tau(ind_mat, 1.)
    part1_reg_mat_tau = qtdmri.part1_reg_matrix_tau(ind_mat, 1.)
    part1_uq_iso_precomp = (
        mapmri.mapmri_isotropic_laplacian_reg_matrix_from_index_matrix(
            ind_mat[:, :3], 1.
            )
        )
    laplacian_matrix_precomp = qtdmri.qtdmri_isotropic_laplacian_reg_matrix(
        ind_mat, us, ut,
        part1_uq_iso_precomp=part1_uq_iso_precomp,
        part1_ut_precomp=part1_reg_mat_tau,
        part23_ut_precomp=part23_reg_mat_tau,
        part4_ut_precomp=part4_reg_mat_tau)
    laplacian_matrix_regular = qtdmri.qtdmri_isotropic_laplacian_reg_matrix(
        ind_mat, us, ut)
    assert_array_almost_equal(laplacian_matrix_precomp,
                              laplacian_matrix_regular)


def test_laplacian_reduces_laplacian_norm(radial_order=4, time_order=2):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)

    qtdmri_mod_no_laplacian = qtdmri.QtdmriModel(
        gtab_4d, radial_order=radial_order, time_order=time_order,
        laplacian_regularization=True, laplacian_weighting=0.
    )
    qtdmri_mod_laplacian = qtdmri.QtdmriModel(
        gtab_4d, radial_order=radial_order, time_order=time_order,
        laplacian_regularization=True, laplacian_weighting=1e-4
    )

    qtdmri_fit_no_laplacian = qtdmri_mod_no_laplacian.fit(S)
    qtdmri_fit_laplacian = qtdmri_mod_laplacian.fit(S)

    laplacian_norm_no_reg = qtdmri_fit_no_laplacian.norm_of_laplacian_signal()
    laplacian_norm_reg = qtdmri_fit_laplacian.norm_of_laplacian_signal()

    assert_equal(laplacian_norm_no_reg > laplacian_norm_reg, True)


def test_spherical_laplacian_reduces_laplacian_norm(radial_order=4,
                                                    time_order=2):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)

    qtdmri_mod_no_laplacian = qtdmri.QtdmriModel(
        gtab_4d, radial_order=radial_order, time_order=time_order,
        cartesian=False, laplacian_regularization=True, laplacian_weighting=0.
    )
    qtdmri_mod_laplacian = qtdmri.QtdmriModel(
        gtab_4d, radial_order=radial_order, time_order=time_order,
        cartesian=False, laplacian_regularization=True,
        laplacian_weighting=1e-4
    )

    qtdmri_fit_no_laplacian = qtdmri_mod_no_laplacian.fit(S)
    qtdmri_fit_laplacian = qtdmri_mod_laplacian.fit(S)

    laplacian_norm_no_reg = qtdmri_fit_no_laplacian.norm_of_laplacian_signal()
    laplacian_norm_reg = qtdmri_fit_laplacian.norm_of_laplacian_signal()

    assert_equal(laplacian_norm_no_reg > laplacian_norm_reg, True)


def test_laplacian_GCV_higher_weight_with_noise(radial_order=4, time_order=2):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)
    S_noise = add_noise(S, S0=1., snr=10)

    qtdmri_mod_laplacian_GCV = qtdmri.QtdmriModel(
        gtab_4d, radial_order=radial_order, time_order=time_order,
        laplacian_regularization=True, laplacian_weighting="GCV"
    )

    qtdmri_fit_no_noise = qtdmri_mod_laplacian_GCV.fit(S)
    qtdmri_fit_noise = qtdmri_mod_laplacian_GCV.fit(S_noise)

    assert_equal(qtdmri_fit_noise.lopt > qtdmri_fit_no_noise.lopt, True)


def test_l1_increases_sparsity(radial_order=4, time_order=2):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)

    qtdmri_mod_no_l1 = qtdmri.QtdmriModel(
        gtab_4d, radial_order=radial_order, time_order=time_order,
        l1_regularization=True, l1_weighting=0.
    )
    qtdmri_mod_l1 = qtdmri.QtdmriModel(
        gtab_4d, radial_order=radial_order, time_order=time_order,
        l1_regularization=True, l1_weighting=.1
    )

    qtdmri_fit_no_l1 = qtdmri_mod_no_l1.fit(S)
    qtdmri_fit_l1 = qtdmri_mod_l1.fit(S)

    sparsity_abs_no_reg = qtdmri_fit_no_l1.sparsity_abs()
    sparsity_abs_reg = qtdmri_fit_l1.sparsity_abs()
    assert_equal(sparsity_abs_no_reg > sparsity_abs_reg, True)

    sparsity_density_no_reg = qtdmri_fit_no_l1.sparsity_density()
    sparsity_density_reg = qtdmri_fit_l1.sparsity_density()
    assert_equal(sparsity_density_no_reg > sparsity_density_reg, True)


def test_spherical_l1_increases_sparsity(radial_order=4, time_order=2):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)

    qtdmri_mod_no_l1 = qtdmri.QtdmriModel(
        gtab_4d, radial_order=radial_order, time_order=time_order,
        l1_regularization=True, cartesian=False, normalization=True,
        l1_weighting=0.
    )
    qtdmri_mod_l1 = qtdmri.QtdmriModel(
        gtab_4d, radial_order=radial_order, time_order=time_order,
        l1_regularization=True, cartesian=False, normalization=True,
        l1_weighting=.1
    )

    qtdmri_fit_no_l1 = qtdmri_mod_no_l1.fit(S)
    qtdmri_fit_l1 = qtdmri_mod_l1.fit(S)

    sparsity_abs_no_reg = qtdmri_fit_no_l1.sparsity_abs()
    sparsity_abs_reg = qtdmri_fit_l1.sparsity_abs()
    assert_equal(sparsity_abs_no_reg > sparsity_abs_reg, True)

    sparsity_density_no_reg = qtdmri_fit_no_l1.sparsity_density()
    sparsity_density_reg = qtdmri_fit_l1.sparsity_density()
    assert_equal(sparsity_density_no_reg > sparsity_density_reg, True)


def test_l1_CV_higher_weight_with_noise(radial_order=4, time_order=2):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)
    S_noise = add_noise(S, S0=1., snr=10)

    qtdmri_mod_l1_cv = qtdmri.QtdmriModel(
        gtab_4d, radial_order=radial_order, time_order=time_order,
        l1_regularization=True, l1_weighting="CV"
    )

    qtdmri_fit_no_noise = qtdmri_mod_l1_cv.fit(S)
    qtdmri_fit_noise = qtdmri_mod_l1_cv.fit(S_noise)
    assert_equal(qtdmri_fit_noise.alpha > qtdmri_fit_no_noise.alpha, True)


def test_elastic_GCV_CV_higher_weight_with_noise(radial_order=4, time_order=2):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)
    S_noise = add_noise(S, S0=1., snr=10)

    qtdmri_mod_elastic = qtdmri.QtdmriModel(
        gtab_4d, radial_order=radial_order, time_order=time_order,
        l1_regularization=True, l1_weighting="CV",
        laplacian_regularization=True, laplacian_weighting="GCV"
    )

    qtdmri_fit_no_noise = qtdmri_mod_elastic.fit(S)
    qtdmri_fit_noise = qtdmri_mod_elastic.fit(S_noise)

    assert_equal(qtdmri_fit_noise.lopt > qtdmri_fit_no_noise.lopt, True)
    assert_equal(qtdmri_fit_noise.alpha > qtdmri_fit_no_noise.alpha, True)

if __name__ == '__main__':
    run_module_suite()
