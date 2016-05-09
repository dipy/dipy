from __future__ import division

import numpy as np

from numpy.testing import (assert_, assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_almost_equal,
                           run_module_suite)

from dipy.data import get_data
from dipy.reconst.dti import TensorModel
from dipy.sims.phantom import orbital_phantom
from dipy.core.gradients import gradient_table


fimg, fbvals, fbvecs = get_data('small_64D')
bvals = np.load(fbvals)
bvecs = np.load(fbvecs)
bvecs[np.isnan(bvecs)] = 0

gtab = gradient_table(bvals, bvecs)


def f(t):
    """
    Helper function used to define a mapping time => xyz
    """
    x = np.linspace(-1, 1, len(t))
    y = np.linspace(-1, 1, len(t))
    z = np.linspace(-1, 1, len(t))
    return x, y, z


def test_phantom():
    N = 50

    vol = orbital_phantom(gtab,
                          func=f,
                          t=np.linspace(0, 2 * np.pi, N),
                          datashape=(10, 10, 10, len(bvals)),
                          origin=(5, 5, 5),
                          scale=(3, 3, 3),
                          angles=np.linspace(0, 2 * np.pi, 16),
                          radii=np.linspace(0.2, 2, 6),
                          S0=100)

    m = TensorModel(gtab)
    t = m.fit(vol)
    FA = t.fa
    # print vol
    FA[np.isnan(FA)] = 0
    # 686 -> expected FA given diffusivities of [1500, 400, 400]
    l1, l2, l3 = 1500e-6, 400e-6, 400e-6
    expected_fa = (np.sqrt(0.5) *
                   np.sqrt((l1 - l2)**2 + (l2-l3)**2 + (l3-l1)**2) /
                   np.sqrt(l1**2 + l2**2 + l3**2))

    assert_array_almost_equal(FA.max(), expected_fa, decimal=2)


def test_add_noise():
    np.random.seed(1980)

    N = 50
    S0 = 100

    options = dict(func=f,
                   t=np.linspace(0, 2 * np.pi, N),
                   datashape=(10, 10, 10, len(bvals)),
                   origin=(5, 5, 5),
                   scale=(3, 3, 3),
                   angles=np.linspace(0, 2 * np.pi, 16),
                   radii=np.linspace(0.2, 2, 6),
                   S0=S0)

    vol = orbital_phantom(gtab, **options)

    for snr in [10, 20, 30, 50]:
        vol_noise = orbital_phantom(gtab, snr=snr, **options)

        sigma = S0 / snr

        assert_(np.abs(np.var(vol_noise - vol) - sigma ** 2) < 1)


if __name__ == "__main__":
    run_module_suite()
