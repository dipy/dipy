from dipy.reconst.mcsd import MultiShellDeconvModel
from dipy.reconst import mcsd
import numpy as np
import numpy.testing as npt
import pytest

from dipy.sims.voxel import multi_shell_fiber_response, multi_tensor
from dipy.reconst import shm
from dipy.data import default_sphere, get_3shell_gtab
from dipy.core.gradients import GradientTable

from dipy.utils.optpkg import optional_package
cvx, have_cvxpy, _ = optional_package("cvxpy")

needs_cvxpy = pytest.mark.skipif(not have_cvxpy)


csf_md = 3e-3
gm_md = .76e-3
evals_d = np.array([.992, .254, .254]) * 1e-3


def _expand(m, iso, coeff):
    params = np.zeros(len(m))
    params[m == 0] = coeff[iso:]
    params = np.concatenate([coeff[:iso], params])
    return params


@pytest.mark.skipif(not mcsd.have_cvxpy, reason="Requires CVXPY")
def test_mcsd_model_delta():
    sh_order = 8
    gtab = get_3shell_gtab()
    shells = np.unique(gtab.bvals // 100.) * 100.
    response = multi_shell_fiber_response(sh_order, shells, evals_d, csf_md,
                                          gm_md)
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


@pytest.mark.skipif(not mcsd.have_cvxpy, reason="Requires CVXPY")
def test_compartments():
    # test for failure if no. of compartments less than 2
    gtab = get_3shell_gtab()
    sh_order = 8
    response = multi_shell_fiber_response(sh_order, [0, 1000, 2000, 3500],
                                          evals_d, csf_md, gm_md)
    npt.assert_raises(ValueError, MultiShellDeconvModel, gtab, response, iso=1)


@pytest.mark.skipif(not mcsd.have_cvxpy, reason="Requires CVXPY")
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
    response = multi_shell_fiber_response(sh_order, [0, 1000, 2000, 3500],
                                          evals_d, csf_md, gm_md)
    model = MultiShellDeconvModel(gtab, response)
    vf = [1.3, .8, 1.9]
    signal = sum(i * j for i, j in zip(vf, [S_csf, S_gm, S_wm]))
    fit = model.fit(signal)

    npt.assert_array_almost_equal(fit.volume_fractions, vf, 0)

    S_pred = fit.predict()
    npt.assert_array_almost_equal(S_pred, signal, 0)


if __name__ == "__main__":
    npt.run_module_suite()
