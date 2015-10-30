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
    f = np.sqrt(2) * factorial(n) / float(((gamma(1 + n / 2.0))
                                           * np.sqrt(2**(n + 1) * factorial(n))))
    return f


def test_mapmri_metrics():
    gtab = get_gtab_taiwan_dsi()
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))
    angl = [(0, 0), (60, 0)]
    S, sticks = MultiTensor(gtab, mevals, S0=100.0, angles=angl,
                            fractions=[50, 50], snr=None)

    # since we are testing without noise we can use higher order and lower
    # lambdas, with respect to the default.
    radial_order = 6
    lambd = 1e-8

    # test mapmri_indices
    indices = mapmri_index_matrix(radial_order)
    n_c = indices.shape[0]
    F = radial_order / 2
    n_gt = np.round(1 / 6.0 * (F + 1) * (F + 2) * (4 * F + 3))
    assert_equal(n_c, n_gt)

    # test MAPMRI fitting

    mapm = MapmriModel(gtab, radial_order=radial_order, lambd=lambd)
    mapfit = mapm.fit(S)
    c_map = mapfit.mapmri_coeff

    R = mapfit.mapmri_R
    mu = mapfit.mapmri_mu

    S_reconst = mapfit.predict(gtab, 1.0)

    # test the signal reconstruction
    S = S / S[0]
    nmse_signal = np.sqrt(np.sum((S - S_reconst) ** 2)) / (S.sum())
    assert_almost_equal(nmse_signal, 0.0, 3)

    # test if the analytical integral of the pdf is equal to one
    integral = 0
    for i in range(indices.shape[0]):
        n1, n2, n3 = indices[i]
        integral += c_map[i] * int_func(n1) * int_func(n2) * int_func(n3)

    assert_almost_equal(integral, 1.0, 3)

    # compare the shore pdf with the ground truth multi_tensor pdf

    sphere = get_sphere('symmetric724')
    v = sphere.vertices
    radius = 10e-3
    r_points = v * radius
    pdf_mt = multi_tensor_pdf(r_points, mevals=mevals,
                              angles=angl, fractions=[50, 50])

    pdf_map = mapmri_EAP(r_points, radial_order, c_map, mu, R)

    nmse_pdf = np.sqrt(np.sum((pdf_mt - pdf_map) ** 2)) / (pdf_mt.sum())
    assert_almost_equal(nmse_pdf, 0.0, 2)

    # test MAPMRI metrics
    tau = 1 / (4 * np.pi ** 2)
    angl = [(0, 0), (0, 0)]
    S, sticks = MultiTensor(gtab, mevals, S0=100.0, angles=angl,
                            fractions=[50, 50], snr=None)

    mapm = MapmriModel(gtab, radial_order=radial_order, lambd=lambd)
    mapfit = mapm.fit(S)

    # RTOP
    gt_rtop = 1.0 / np.sqrt((4 * np.pi * tau)**3 *
                            mevals[0, 0] * mevals[0, 1] * mevals[0, 2])
    rtop = mapfit.rtop()
    assert_almost_equal(rtop, gt_rtop, 4)

    # RTAP
    gt_rtap = 1.0 / np.sqrt((4 * np.pi * tau)**2 * mevals[0, 1] * mevals[0, 2])
    rtap = mapfit.rtap()
    assert_almost_equal(rtap, gt_rtap, 4)

    # RTPP
    gt_rtpp = 1.0 / np.sqrt((4 * np.pi * tau) * mevals[0, 0])
    rtpp = mapfit.rtpp()
    assert_almost_equal(rtpp, gt_rtpp, 4)


if __name__ == '__main__':
    run_module_suite()
