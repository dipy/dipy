import numpy as np
from dipy.data import get_gtab_taiwan_dsi
from numpy.testing import (assert_almost_equal,
                           assert_equal,
                           run_module_suite)
from dipy.reconst.shore import ShoreModel, shore_matrix, shore_indices, shore_order
from dipy.sims.voxel import (
    MultiTensor, all_tensor_evecs, multi_tensor_odf, single_tensor_odf,
    multi_tensor_rtop, multi_tensor_msd, multi_tensor_pdf)
from dipy.data import get_sphere
from scipy.special import genlaguerre, gamma
from math import factorial


def test_shore_fitting_e0():
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

    radial_order = 8
    zeta = 700
    lambdaN = 1e-12
    lambdaL = 1e-12

    asm = ShoreModel(gtab, radial_order=radial_order,
                     zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
    asmfit = asm.fit(S)

    assert_almost_equal(compute_e0(asmfit), 1)

    asm = ShoreModel(gtab, radial_order=radial_order,
                     zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL,
                     constrain_e0=True)
    asmfit = asm.fit(S)

    assert_almost_equal(compute_e0(asmfit), 1.)


def compute_e0(shorefit):
    signal_0 = 0

    for n in range(int(shorefit.model.radial_order / 2) + 1):
        signal_0 += shorefit.shore_coeff[n] * (genlaguerre(n, 0.5)(0) *
        ((factorial(n)) / (2 * np.pi * (shorefit.model.zeta ** 1.5) * gamma(n + 1.5))) ** 0.5)

    return signal_0
