import numpy as np
from dipy.data import get_gtab_taiwan_dsi
from numpy.testing import (assert_almost_equal,
                           assert_array_almost_equal,
                           assert_equal,
                           run_module_suite)
from dipy.reconst import qtdmri
from dipy.sims.voxel import MultiTensor
from dipy.data import get_sphere
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
    S, sticks = MultiTensor(gtab, mevals, S0=100.0, angles=angl,
                            fractions=[50, 50], snr=None)
    return S


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


def test_anisotropic_normalization(radial_order=4, time_order=2):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)

    qtdmri_mod_aniso = qtdmri.QtdmriModel(gtab_4d, radial_order=radial_order,
                                          time_order=time_order,
                                          cartesian=False)
    qtdmri_mod_aniso_norm = qtdmri.QtdmriModel(gtab_4d,
                                               radial_order=radial_order,
                                               time_order=time_order,
                                               cartesian=False,
                                               anisotropic_scaling=False,
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


def test_anisotropic_reduced_MSE(radial_order=0, time_order=0):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3) / 100.
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

if __name__ == '__main__':
    run_module_suite()
