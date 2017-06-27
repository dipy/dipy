import numpy as np
from dipy.data import get_gtab_taiwan_dsi
from numpy.testing import (assert_almost_equal,
                           assert_array_almost_equal,
                           assert_equal,
                           run_module_suite)
from dipy.reconst.shore_time import ShoreTemporalModel
from dipy.reconst import maptime
from dipy.sims.voxel import (MultiTensor, all_tensor_evecs,  multi_tensor_pdf)
from scipy.special import gamma
from scipy.misc import factorial
from dipy.data import get_sphere
from dipy.sims.voxel import add_noise
import scipy.integrate as integrate
import scipy.special as special
from dipy.core.gradients import gradient_table


def generate_gtab4D(number_of_tau_shells=4):
    gtab = get_gtab_taiwan_dsi()
    qvals = np.tile(gtab.bvals / 100., number_of_tau_shells)
    bvecs = np.tile(gtab.bvecs, (number_of_tau_shells, 1))
    pulse_separation = []
    for ps in np.linspace(0.02, 0.05, number_of_tau_shells):
        pulse_separation = np.append(pulse_separation,
                                     np.tile(ps, gtab.bvals.shape[0]))
    pulse_duration = np.tile(0.01, qvals.shape[0])
    gtab_4d = gradient_table(qvals=qvals, bvecs=bvecs,
                             pulse_separation=pulse_separation,
                             pulse_duration=pulse_duration)
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
                          maptime.temporal_basis(1, ut, t) *
                          maptime.temporal_basis(2, ut, t), tmin, tmax)
    int2 = integrate.quad(lambda t:
                          maptime.temporal_basis(2, ut, t) *
                          maptime.temporal_basis(3, ut, t), tmin, tmax)
    int3 = integrate.quad(lambda t:
                          maptime.temporal_basis(3, ut, t) *
                          maptime.temporal_basis(4, ut, t), tmin, tmax)
    int4 = integrate.quad(lambda t:
                          maptime.temporal_basis(4, ut, t) *
                          maptime.temporal_basis(5, ut, t), tmin, tmax)

    assert_almost_equal(int1, 0.)
    assert_almost_equal(int2, 0.)
    assert_almost_equal(int3, 0.)
    assert_almost_equal(int4, 0.)


def test_normalization_time():
    ut = 10
    tmin = 0
    tmax = 100

    int0 = integrate.quad(lambda t:
                          maptime.maptime_temporal_normalization(ut) ** 2 *
                          maptime.temporal_basis(0, ut, t) *
                          maptime.temporal_basis(0, ut, t), tmin, tmax)[0]
    int1 = integrate.quad(lambda t:
                          maptime.maptime_temporal_normalization(ut) ** 2 *
                          maptime.temporal_basis(1, ut, t) *
                          maptime.temporal_basis(1, ut, t), tmin, tmax)[0]
    int2 = integrate.quad(lambda t:
                          maptime.maptime_temporal_normalization(ut) ** 2 *
                          maptime.temporal_basis(2, ut, t) *
                          maptime.temporal_basis(2, ut, t), tmin, tmax)[0]

    assert_almost_equal(int0, 1.)
    assert_almost_equal(int1, 1.)
    assert_almost_equal(int2, 1.)


def test_anisotropic_isotropic_equivalence(radial_order=4, time_order=2):
    gtab_4d = generate_gtab4D()

    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)

    mapmod_aniso = maptime.MaptimeModel(gtab_4d, radial_order=radial_order,
                                        time_order=time_order,
                                        cartesian=True,
                                        anisotropic_scaling=False)
    mapmod_iso = maptime.MaptimeModel(gtab_4d, radial_order=radial_order,
                                      time_order=time_order,
                                      anisotropic_scaling=False)

    mapfit_aniso = mapmod_aniso.fit(S)
    mapfit_iso = mapmod_iso.fit(S)

    assert_array_almost_equal(mapfit_aniso.fitted_signal(),
                              mapfit_iso.fitted_signal())

    rt_grid = maptime.create_rspace_tau(5, 20e-3, 5, 0.02, .05)

    pdf_aniso = mapfit_aniso.pdf(rt_grid)
    pdf_iso = mapfit_iso.pdf(rt_grid)

    assert_array_almost_equal(pdf_aniso / pdf_aniso.max(),
                              pdf_iso / pdf_aniso.max())

    norm_laplacian_aniso = mapfit_aniso.norm_of_laplacian_signal()
    norm_laplacian_iso = mapfit_iso.norm_of_laplacian_signal()

    assert_almost_equal(norm_laplacian_aniso / norm_laplacian_aniso,
                        norm_laplacian_iso / norm_laplacian_aniso)


def test_anisotropic_normalization(radial_order=4, time_order=2):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3)

    mapmod_aniso = maptime.MaptimeModel(gtab_4d, radial_order=radial_order,
                                        time_order=time_order,
                                        cartesian=False,
                                        fit_tau_inf=True)
    mapmod_aniso_norm = maptime.MaptimeModel(gtab_4d,
                                             radial_order=radial_order,
                                             time_order=time_order,
                                             cartesian=False,
                                             anisotropic_scaling=False,
                                             normalization=True)
    mapfit_aniso = mapmod_aniso.fit(S)
    mapfit_aniso_norm = mapmod_aniso_norm.fit(S)
    assert_array_almost_equal(mapfit_aniso.fitted_signal(),
                              mapfit_aniso_norm.fitted_signal())
    rt_grid = maptime.create_rspace_tau(5, 20e-3, 5, 0.02, .05)
    pdf_aniso = mapfit_aniso.pdf(rt_grid)
    pdf_aniso_norm = mapfit_aniso_norm.pdf(rt_grid)
    assert_array_almost_equal(pdf_aniso / pdf_aniso.max(),
                              pdf_aniso_norm / pdf_aniso.max())


def test_anisotropic_reduced_MSE(radial_order=0, time_order=0):
    gtab_4d = generate_gtab4D()
    l1, l2, l3 = [0.0015, 0.0003, 0.0003]
    S = generate_signal_crossing(gtab_4d, l1, l2, l3) / 100.
    mapmod_aniso = maptime.MaptimeModel(gtab_4d, radial_order=radial_order,
                                        time_order=time_order,
                                        cartesian=True,
                                        anisotropic_scaling=True)
    mapmod_iso = maptime.MaptimeModel(gtab_4d, radial_order=radial_order,
                                      time_order=time_order,
                                      cartesian=True,
                                      anisotropic_scaling=False)
    mapfit_aniso = mapmod_aniso.fit(S)
    mapfit_iso = mapmod_iso.fit(S)
    mse_aniso = np.mean((S - mapfit_aniso.fitted_signal()) ** 2)
    mse_iso = np.mean((S - mapfit_iso.fitted_signal()) ** 2)
    assert_equal(mse_aniso < mse_iso, True)

if __name__ == '__main__':
    run_module_suite()
