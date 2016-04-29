import numpy as np
from dipy.data import get_gtab_taiwan_dsi
from numpy.testing import (assert_almost_equal,
                           assert_equal,
                           run_module_suite)
from dipy.reconst.shore import (ShoreModel,
                                shore_matrix,
                                shore_indices,
                                shore_order)
from dipy.sims.voxel import (
    MultiTensor, all_tensor_evecs, multi_tensor_odf, single_tensor_odf,
    multi_tensor_rtop, multi_tensor_msd, multi_tensor_pdf)
from dipy.data import get_sphere
from scipy.special import genlaguerre


def test_shore_metrics():
    gtab = get_gtab_taiwan_dsi()
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))
    angl = [(0, 0), (60, 0)]
    S, sticks = MultiTensor(gtab, mevals, S0=100.0, angles=angl,
                            fractions=[50, 50], snr=None)

    # test shore_indices
    n = 7
    l = 6
    m = -4
    radial_order, c = shore_order(n, l, m)
    n2, l2, m2 = shore_indices(radial_order, c)
    assert_equal(n, n2)
    assert_equal(l, l2)
    assert_equal(m, m2)

    radial_order = 6
    c = 41
    n, l, m = shore_indices(radial_order, c)
    radial_order2, c2 = shore_order(n, l, m)
    assert_equal(radial_order, radial_order2)
    assert_equal(c, c2)

    # since we are testing without noise we can use higher order and lower
    # lambdas, with respect to the default.
    radial_order = 8
    zeta = 700
    lambdaN = 1e-12
    lambdaL = 1e-12
    asm = ShoreModel(gtab, radial_order=radial_order,
                     zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
    asmfit = asm.fit(S)
    c_shore = asmfit.shore_coeff

    cmat = shore_matrix(radial_order, zeta, gtab)
    S_reconst = np.dot(cmat, c_shore)

    # test the signal reconstruction
    S = S / S[0]
    nmse_signal = np.sqrt(np.sum((S - S_reconst) ** 2)) / (S.sum())
    assert_almost_equal(nmse_signal, 0.0, 4)

    # test if the analytical integral of the pdf is equal to one
    integral = 0
    for n in range(int((radial_order)/2 + 1)):
        integral += c_shore[n] * (np.pi**(-1.5) * zeta ** (-1.5) *
                                  genlaguerre(n, 0.5)(0)) ** 0.5

    assert_almost_equal(integral, 1.0, 10)

    # test if the integral of the pdf calculated on a discrete grid is equal to
    # one
    pdf_discrete = asmfit.pdf_grid(17, 40e-3)
    integral = pdf_discrete.sum()
    assert_almost_equal(integral, 1.0, 1)

    # compare the shore pdf with the ground truth multi_tensor pdf

    sphere = get_sphere('symmetric724')
    v = sphere.vertices
    radius = 10e-3
    pdf_shore = asmfit.pdf(v * radius)
    pdf_mt = multi_tensor_pdf(v * radius, mevals=mevals,
                              angles=angl, fractions=[50, 50])

    nmse_pdf = np.sqrt(np.sum((pdf_mt - pdf_shore) ** 2)) / (pdf_mt.sum())
    assert_almost_equal(nmse_pdf, 0.0, 2)

    # compare the shore rtop with the ground truth multi_tensor rtop
    rtop_shore_signal = asmfit.rtop_signal()
    rtop_shore_pdf = asmfit.rtop_pdf()
    assert_almost_equal(rtop_shore_signal, rtop_shore_pdf, 9)
    rtop_mt = multi_tensor_rtop([.5, .5], mevals=mevals)
    assert_equal(rtop_mt / rtop_shore_signal < 1.10 and
                 rtop_mt / rtop_shore_signal > 0.95, True)

    # compare the shore msd with the ground truth multi_tensor msd
    msd_mt = multi_tensor_msd([.5, .5], mevals=mevals)
    msd_shore = asmfit.msd()
    assert_equal(msd_mt / msd_shore < 1.05 and msd_mt / msd_shore > 0.95, True)

if __name__ == '__main__':
    run_module_suite()
