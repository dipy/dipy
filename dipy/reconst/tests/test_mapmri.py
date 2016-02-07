import numpy as np
from dipy.data import get_gtab_taiwan_dsi
from numpy.testing import (assert_almost_equal,
                           assert_equal,
                           run_module_suite)
from dipy.reconst.mapmri import MapmriModel, mapmri_index_matrix, mapmri_EAP
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


def test_mapmri_metrics(radial_order=6):
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

if __name__ == '__main__':
    run_module_suite()
