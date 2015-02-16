from dipy.reconst.msd import MultiShellDeconvModel, MultiShellResponse
import numpy as np
import numpy.testing as npt

from dipy.sims.voxel import (multi_tensor, single_tensor)
from dipy.reconst import shm
from dipy.sims import voxel as voxSim
from dipy.data import default_sphere, get_3shell_gtab
from dipy.core.gradients import GradientTable

import dipy.viz.fvtk as fvtk

def show_response(r, m, n):
    sphere = default_sphere.mirror().subdivide()
    theta, phi = sphere.theta, sphere.phi
    B = shm.real_sph_harm(m, n, theta[:, None], phi[:, None])
    sp_func = np.dot(r, B.T)
    print(sp_func[:, :5])

    act = fvtk.sphere_funcs(sp_func, sphere, norm=False)
    ren = fvtk.ren()
    fvtk.add(ren, act)
    fvtk.show(ren)

csf_md=3e-3
gm_md=.76e-3
evals_d = np.array([.992, .254, .254]) * 1e-3

def sim_response(sh_order, bvals, evals=evals_d, csf_md=3e-3, gm_md=.76e-3):
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


def test_MultiShellDeconvModel():

    gtab = get_3shell_gtab()

    S0 = 1.
    evals = np.array([.992, .254, .254]) * 1e-3
    mevals = np.array([evals, evals])
    angles = [(0, 0), (60, 0)]

    S_wm, sticks = multi_tensor(gtab, mevals, S0, angles=angles,
                                fractions=[50., 50.], snr=None)
    S_gm = np.exp(-gtab.bvals * gm_md)
    S_csf = np.exp(-gtab.bvals * csf_md)

    sh_order = 8
    response = sim_response(sh_order, [0, 1000, 2000, 3500])
    model = MultiShellDeconvModel(gtab, response)
    signal = S_wm + 2 * S_gm + .5 * S_csf
    fit = model.fit(signal)
    q = model.predict(fit._shm_coef)

    S = fit.predict()
    refit = model.fit(S)
    npt.assert_array_almost_equal(refit._shm_coef, fit._shm_coef)


test_MultiShellDeconvModel()
