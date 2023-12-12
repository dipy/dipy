# Tests for shore fitting
import warnings
from math import factorial

import numpy as np
import numpy.testing as npt

from scipy.special import genlaguerre, gamma
import pytest

from dipy.data import get_gtab_taiwan_dsi
from dipy.reconst.shm import descoteaux07_legacy_msg
from dipy.reconst.shore import ShoreModel
from dipy.sims.voxel import multi_tensor
from dipy.utils.optpkg import optional_package

cvxpy, have_cvxpy, _ = optional_package("cvxpy", min_version="1.4.1")
needs_cvxpy = pytest.mark.skipif(not have_cvxpy, reason="Requires CVXPY")


# Object to hold module global data
class _C:
    pass
data = _C()


def setup():
    data.gtab = get_gtab_taiwan_dsi()
    data.mevals = np.array(([0.0015, 0.0003, 0.0003],
                            [0.0015, 0.0003, 0.0003]))
    data.angl = [(0, 0), (60, 0)]
    data.S, sticks = multi_tensor(data.gtab, data.mevals, S0=100.0,
                                  angles=data.angl, fractions=[50, 50],
                                  snr=None)
    data.radial_order = 6
    data.zeta = 700
    data.lambdaN = 1e-12
    data.lambdaL = 1e-12


def test_shore_error():
    data.gtab = get_gtab_taiwan_dsi()
    npt.assert_raises(ValueError, ShoreModel, data.gtab, radial_order=-4)
    npt.assert_raises(ValueError, ShoreModel, data.gtab, radial_order=7)
    npt.assert_raises(ValueError, ShoreModel, data.gtab, constrain_e0=False,
                      positive_constraint=True)


@needs_cvxpy
def test_shore_positive_constrain():
    asm = ShoreModel(data.gtab,
                     radial_order=data.radial_order,
                     zeta=data.zeta,
                     lambdaN=data.lambdaN,
                     lambdaL=data.lambdaL,
                     constrain_e0=True,
                     positive_constraint=True,
                     pos_grid=11,
                     pos_radius=20e-03)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        asmfit = asm.fit(data.S)
        eap = asmfit.pdf_grid(11, 20e-03)
    npt.assert_almost_equal(eap[eap < 0].sum(), 0, 3)


def test_shore_fitting_no_constrain_e0():
    asm = ShoreModel(data.gtab, radial_order=data.radial_order,
                     zeta=data.zeta, lambdaN=data.lambdaN,
                     lambdaL=data.lambdaL)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        asmfit = asm.fit(data.S)
    npt.assert_almost_equal(compute_e0(asmfit), 1)


@needs_cvxpy
def test_shore_fitting_constrain_e0():
    asm = ShoreModel(data.gtab, radial_order=data.radial_order,
                     zeta=data.zeta, lambdaN=data.lambdaN,
                     lambdaL=data.lambdaL,
                     constrain_e0=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        asmfit = asm.fit(data.S)
    npt.assert_almost_equal(compute_e0(asmfit), 1)


def compute_e0(shorefit):
    signal_0 = 0

    for n in range(int(shorefit.model.radial_order / 2) + 1):
        signal_0 += (shorefit.shore_coeff[n] * (genlaguerre(n, 0.5)(0) *
                     ((factorial(n)) / (2 * np.pi *
                      (shorefit.model.zeta ** 1.5) *
                      gamma(n + 1.5))) ** 0.5))

    return signal_0
