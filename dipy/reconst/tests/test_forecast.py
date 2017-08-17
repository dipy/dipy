# Tests for FORECAST fitting and metrics

import numpy as np

from dipy.data import get_sphere, get_3shell_gtab
from dipy.reconst.forecast import ForecastModel
from dipy.sims.voxel import MultiTensor

from numpy.testing import (assert_almost_equal,
                           assert_equal,
                           run_module_suite,
                           dec)
from dipy.direction.peaks import peak_directions
from dipy.core.sphere_stats import angular_similarity
from dipy.utils.optpkg import optional_package
cvxpy, have_cvxpy, _ = optional_package("cvxpy")

needs_cvxpy = dec.skipif(not have_cvxpy)


# Object to hold module global data
class _C(object):
    pass
data = _C()


def setup():
    data.gtab = get_3shell_gtab()
    data.mevals = np.array(([0.0017, 0.0003, 0.0003],
                            [0.0017, 0.0003, 0.0003]))
    data.angl = [(0, 0), (60, 0)]
    data.S, data.sticks = MultiTensor(
        data.gtab, data.mevals, S0=100.0, angles=data.angl,
        fractions=[50, 50], snr=None)
    data.sh_order = 6
    data.lambda_LB = 1e-8
    data.lambda_csd = 1.0
    sphere = get_sphere('repulsion100')
    data.sphere = sphere.vertices[0:int(sphere.vertices.shape[0]/2), :]


@needs_cvxpy
def test_forecast_positive_constrain():
    fm = ForecastModel(data.gtab,
                       sh_order=data.sh_order,
                       lambda_LB=data.lambda_LB,
                       optimizer='pos',
                       sphere=data.sphere)
    f_fit = fm.fit(data.S)

    sphere = get_sphere('repulsion100')
    fodf = f_fit.odf(sphere)
    assert_equal(fodf[fodf < 0].sum(), 0)

    coeff = f_fit.sh_coeff
    c0 = np.sqrt(1.0/(4*np.pi))
    assert_almost_equal(coeff[0], c0, 10)


def test_forecast_csd():
    sphere = get_sphere('repulsion100')
    fm = ForecastModel(data.gtab, optimizer='csd',
                       sphere=data.sphere, lambda_csd=data.lambda_csd)
    f_fit = fm.fit(data.S)
    fodf_csd = f_fit.odf(sphere)

    fm = ForecastModel(data.gtab, sh_order=data.sh_order,
                       lambda_LB=data.lambda_LB, optimizer='wls')
    f_fit = fm.fit(data.S)
    fodf_wls = f_fit.odf(sphere)

    value = fodf_wls[fodf_wls < 0].sum() < fodf_csd[fodf_csd < 0].sum()
    assert_equal(value, 1)


def test_forecast_odf():
    # check FORECAST fODF at different SH order
    fm = ForecastModel(data.gtab, sh_order=4,
                       optimizer='csd', sphere=data.sphere)
    f_fit = fm.fit(data.S)
    sphere = get_sphere('repulsion724')
    fodf = f_fit.odf(sphere)
    directions, _, _ = peak_directions(fodf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, data.sticks), 2, 1)

    fm = ForecastModel(data.gtab, sh_order=6,
                       optimizer='csd', sphere=data.sphere)
    f_fit = fm.fit(data.S)
    fodf = f_fit.odf(sphere)
    directions, _, _ = peak_directions(fodf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, data.sticks), 2, 1)

    fm = ForecastModel(data.gtab, sh_order=8,
                       optimizer='csd', sphere=data.sphere)
    f_fit = fm.fit(data.S)
    fodf = f_fit.odf(sphere)
    directions, _, _ = peak_directions(fodf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, data.sticks), 2, 1)

    # stronger regularization is required for high order SH
    fm = ForecastModel(data.gtab, sh_order=10,
                       optimizer='csd', sphere=sphere.vertices)
    f_fit = fm.fit(data.S)
    fodf = f_fit.odf(sphere)
    directions, _, _ = peak_directions(fodf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, data.sticks), 2, 1)

    fm = ForecastModel(data.gtab, sh_order=12,
                       optimizer='csd', sphere=sphere.vertices)
    f_fit = fm.fit(data.S)
    fodf = f_fit.odf(sphere)
    directions, _, _ = peak_directions(fodf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, data.sticks), 2, 1)


def test_forecast_indices():
    # check anisotropic tensor
    fm = ForecastModel(data.gtab, sh_order=2,
                       lambda_LB=data.lambda_LB, optimizer='wls')
    f_fit = fm.fit(data.S)

    d_par = f_fit.dpar
    d_perp = f_fit.dperp

    assert_almost_equal(d_par, data.mevals[0, 0], 5)
    assert_almost_equal(d_perp, data.mevals[0, 1], 5)

    gt_fa = np.sqrt(0.5 * (2*(data.mevals[0, 0] - data.mevals[0, 1])**2) / (
        data.mevals[0, 0]**2 + 2*data.mevals[0, 1]**2))
    gt_md = (data.mevals[0, 0] + 2*data.mevals[0, 1])/3.0

    assert_almost_equal(f_fit.fractional_anisotropy(), gt_fa, 2)
    assert_almost_equal(f_fit.mean_diffusivity(), gt_md, 5)

    # check isotropic tensor
    mevals = np.array(([0.003, 0.003, 0.003],
                       [0.003, 0.003, 0.003]))
    data.angl = [(0, 0), (60, 0)]
    S, sticks = MultiTensor(
        data.gtab, mevals, S0=100.0, angles=data.angl,
        fractions=[50, 50], snr=None)

    fm = ForecastModel(data.gtab, sh_order=data.sh_order,
                       lambda_LB=data.lambda_LB, optimizer='wls')
    f_fit = fm.fit(S)

    d_par = f_fit.dpar
    d_perp = f_fit.dperp

    assert_almost_equal(d_par, 3e-03, 5)
    assert_almost_equal(d_perp, 3e-03, 5)
    assert_almost_equal(f_fit.fractional_anisotropy(), 0.0, 5)
    assert_almost_equal(f_fit.mean_diffusivity(), 3e-03, 10)


if __name__ == '__main__':
    run_module_suite()
