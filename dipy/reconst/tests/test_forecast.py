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
    data.lambda_lb = 1e-8
    data.lambda_csd = 1.0
    sphere = get_sphere('repulsion100')
    data.sphere = sphere.vertices[0:int(sphere.vertices.shape[0]/2), :]


@needs_cvxpy
def test_forecast_positive_constrain():
    fm = ForecastModel(data.gtab,
                       sh_order=data.sh_order,
                       lambda_lb=data.lambda_lb,
                       dec_alg='POS',
                       sphere=data.sphere)
    f_fit = fm.fit(data.S)

    sphere = get_sphere('repulsion100')
    fodf = f_fit.odf(sphere, clip_negative=False)
    assert_almost_equal(fodf[fodf < 0].sum(), 0, 2)

    coeff = f_fit.sh_coeff
    c0 = np.sqrt(1.0/(4*np.pi))
    assert_almost_equal(coeff[0], c0, 5)


def test_forecast_csd():
    sphere = get_sphere('repulsion100')
    fm = ForecastModel(data.gtab, dec_alg='CSD',
                       sphere=data.sphere, lambda_csd=data.lambda_csd)
    f_fit = fm.fit(data.S)
    fodf_csd = f_fit.odf(sphere, clip_negative=False)

    fm = ForecastModel(data.gtab, sh_order=data.sh_order,
                       lambda_lb=data.lambda_lb, dec_alg='WLS')
    f_fit = fm.fit(data.S)
    fodf_wls = f_fit.odf(sphere, clip_negative=False)

    value = fodf_wls[fodf_wls < 0].sum() < fodf_csd[fodf_csd < 0].sum()
    assert_equal(value, 1)


def test_forecast_odf():
    # check FORECAST fODF at different SH order
    fm = ForecastModel(data.gtab, sh_order=4,
                       dec_alg='CSD', sphere=data.sphere)
    f_fit = fm.fit(data.S)
    sphere = get_sphere('repulsion724')
    fodf = f_fit.odf(sphere)
    directions, _, _ = peak_directions(fodf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, data.sticks), 2, 1)

    fm = ForecastModel(data.gtab, sh_order=6,
                       dec_alg='CSD', sphere=data.sphere)
    f_fit = fm.fit(data.S)
    fodf = f_fit.odf(sphere)
    directions, _, _ = peak_directions(fodf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, data.sticks), 2, 1)

    fm = ForecastModel(data.gtab, sh_order=8,
                       dec_alg='CSD', sphere=data.sphere)
    f_fit = fm.fit(data.S)
    fodf = f_fit.odf(sphere)
    directions, _, _ = peak_directions(fodf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, data.sticks), 2, 1)

    # stronger regularization is required for high order SH
    fm = ForecastModel(data.gtab, sh_order=10,
                       dec_alg='CSD', sphere=sphere.vertices)
    f_fit = fm.fit(data.S)
    fodf = f_fit.odf(sphere)
    directions, _, _ = peak_directions(fodf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, data.sticks), 2, 1)

    fm = ForecastModel(data.gtab, sh_order=12,
                       dec_alg='CSD', sphere=sphere.vertices)
    f_fit = fm.fit(data.S)
    fodf = f_fit.odf(sphere)
    directions, _, _ = peak_directions(fodf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, data.sticks), 2, 1)


def test_forecast_indices():
    # check anisotropic tensor
    fm = ForecastModel(data.gtab, sh_order=2,
                       lambda_lb=data.lambda_lb, dec_alg='WLS')
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
                       lambda_lb=data.lambda_lb, dec_alg='WLS')
    f_fit = fm.fit(S)

    d_par = f_fit.dpar
    d_perp = f_fit.dperp

    assert_almost_equal(d_par, 3e-03, 5)
    assert_almost_equal(d_perp, 3e-03, 5)
    assert_almost_equal(f_fit.fractional_anisotropy(), 0.0, 5)
    assert_almost_equal(f_fit.mean_diffusivity(), 3e-03, 10)


def test_forecast_predict():
    # check anisotropic tensor
    fm = ForecastModel(data.gtab, sh_order=8,
                       dec_alg='CSD', sphere=data.sphere)
    f_fit = fm.fit(data.S)

    S = f_fit.predict(S0=1.0)

    mse = np.sum((S-data.S/100.0)**2) / len(S)

    assert_almost_equal(mse, 0.0, 3)


def test_multivox_forecast():
    gtab = get_3shell_gtab()
    mevals = np.array(([0.0017, 0.0003, 0.0003],
                       [0.0017, 0.0003, 0.0003]))

    angl1 = [(0, 0), (60, 0)]
    angl2 = [(90, 0), (45, 90)]
    angl3 = [(0, 0), (90, 0)]

    S = np.zeros((3, 1, 1, len(gtab.bvals)))
    S[0, 0, 0], sticks = MultiTensor(
        gtab, mevals, S0=1.0, angles=angl1,
        fractions=[50, 50], snr=None)
    S[1, 0, 0], sticks = MultiTensor(
        gtab, mevals, S0=1.0, angles=angl2,
        fractions=[50, 50], snr=None)
    S[2, 0, 0], sticks = MultiTensor(
        gtab, mevals, S0=1.0, angles=angl3,
        fractions=[50, 50], snr=None)

    fm = ForecastModel(gtab, sh_order=8,
                       dec_alg='CSD')
    f_fit = fm.fit(S)

    S_predict = f_fit.predict()

    assert_equal(S_predict.shape, S.shape)

    mse1 = np.sum((S_predict[0, 0, 0]-S[0, 0, 0])**2) / len(gtab.bvals)
    assert_almost_equal(mse1, 0.0, 3)

    mse2 = np.sum((S_predict[1, 0, 0]-S[1, 0, 0])**2) / len(gtab.bvals)
    assert_almost_equal(mse2, 0.0, 3)

    mse3 = np.sum((S_predict[2, 0, 0]-S[2, 0, 0])**2) / len(gtab.bvals)
    assert_almost_equal(mse3, 0.0, 3)

if __name__ == '__main__':
    run_module_suite()
