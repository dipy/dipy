from dipy.reconst.mcsd import MultiShellDeconvModel, MultiShellResponse
from dipy.reconst import mcsd
import numpy as np
import numpy.testing as npt

from dipy.sims.voxel import (multi_tensor, single_tensor)
from dipy.reconst import shm
from dipy.data import default_sphere, get_3shell_gtab
from dipy.core.gradients import GradientTable

from dipy.utils.optpkg import optional_package
cvx, have_cvxpy, _ = optional_package("cvxpy")

needs_cvxpy = npt.dec.skipif(not have_cvxpy)


csf_md = 3e-3
gm_md = .76e-3
evals_d = np.array([.992, .254, .254]) * 1e-3


def sim_response(sh_order, bvals, evals=evals_d, csf_md=csf_md, gm_md=gm_md):
    bvals = np.array(bvals, copy=True)
    evecs = np.zeros((3, 3))
    z = np.array([0, 0, 1.])
    evecs[:, 0] = z
    evecs[:2, 1:] = np.eye(2)

    n = np.arange(0, sh_order + 1, 2)
    m = np.zeros_like(n)

    big_sphere = default_sphere.subdivide()
    theta, phi = big_sphere.theta, big_sphere.phi

    B = shm.real_sph_harm(m, n, theta[:, None], phi[:, None])
    A = shm.real_sph_harm(0, 0, 0, 0)

    response = np.empty([len(bvals), len(n) + 2])
    for i, bvalue in enumerate(bvals):
        gtab = GradientTable(big_sphere.vertices * bvalue)
        wm_response = single_tensor(gtab, 1., evals, evecs, snr=None)
        response[i, 2:] = np.linalg.lstsq(B, wm_response)[0]

        response[i, 0] = np.exp(-bvalue * csf_md) / A
        response[i, 1] = np.exp(-bvalue * gm_md) / A

    return MultiShellResponse(response, sh_order, bvals)


def _expand(m, iso, coeff):
    params = np.zeros(len(m))
    params[m == 0] = coeff[iso:]
    params = np.concatenate([coeff[:iso], params])
    return params


@npt.dec.skipif(not mcsd.have_cvxpy)
def test_mcsd_model_delta():
    sh_order = 8
    gtab = get_3shell_gtab()
    shells = np.unique(gtab.bvals // 100.) * 100.
    response = sim_response(sh_order, shells, evals_d)
    model = MultiShellDeconvModel(gtab, response)
    iso = response.iso

    theta, phi = default_sphere.theta, default_sphere.phi
    B = shm.real_sph_harm(response.m, response.n, theta[:, None], phi[:, None])

    wm_delta = model.delta.copy()
    # set isotropic components to zero
    wm_delta[:iso] = 0.
    wm_delta = _expand(model.m, iso, wm_delta)

    for i, s in enumerate(shells):
        g = GradientTable(default_sphere.vertices * s)
        signal = model.predict(wm_delta, g)
        expected = np.dot(response.response[i, iso:], B.T)
        npt.assert_array_almost_equal(signal, expected)

    signal = model.predict(wm_delta, gtab)
    fit = model.fit(signal)
    m = model.m
    npt.assert_array_almost_equal(fit.shm_coeff[m != 0], 0., 2)


@npt.dec.skipif(not mcsd.have_cvxpy)
def test_compartments():
    # test for failure if no. of compartments less than 2
    gtab = get_3shell_gtab()
    sh_order = 8
    response = sim_response(sh_order, [0, 1000, 2000, 3500])
    npt.assert_raises(ValueError, MultiShellDeconvModel, gtab, response, iso=1)


@npt.dec.skipif(not mcsd.have_cvxpy)
def test_MultiShellDeconvModel():

    gtab = get_3shell_gtab()

    S0 = 1.
    evals = np.array([.992, .254, .254]) * 1e-3
    mevals = np.array([evals, evals])
    angles = [(0, 0), (60, 0)]

    S_wm, sticks = multi_tensor(gtab, mevals, S0, angles=angles,
                                fractions=[30., 70.], snr=None)
    S_gm = np.exp(-gtab.bvals * gm_md)
    S_csf = np.exp(-gtab.bvals * csf_md)

    sh_order = 8
    response = sim_response(sh_order, [0, 1000, 2000, 3500])
    model = MultiShellDeconvModel(gtab, response)
    vf = [1.3, .8, 1.9]
    signal = sum(i * j for i, j in zip(vf, [S_csf, S_gm, S_wm]))
    fit = model.fit(signal)

    npt.assert_array_almost_equal(fit.volume_fractions, vf, 0)

    S_pred = fit.predict()
    npt.assert_array_almost_equal(S_pred, signal, 0)


if __name__ == "__main__":
    npt.run_module_suite()
