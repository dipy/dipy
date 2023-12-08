import numpy as np
from dipy.viz import regtools
import numpy.testing as npt
import pytest
from dipy.align.metrics import SSDMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.testing.decorators import set_random_number_generator

# Conditional import machinery for matplotlib
from dipy.utils.optpkg import optional_package

_, have_matplotlib, _ = optional_package('matplotlib')


@pytest.mark.skipif(not have_matplotlib, reason='Requires Matplotlib')
@set_random_number_generator()
def test_plot_2d_diffeomorphic_map(rng):
    # Test the regtools plotting interface (lightly).
    mv_shape = (11, 12)
    moving = rng.random(mv_shape)
    st_shape = (13, 14)
    static = rng.random(st_shape)
    dim = static.ndim
    metric = SSDMetric(dim)
    level_iters = [200, 100, 50, 25]
    sdr = SymmetricDiffeomorphicRegistration(metric,
                                             level_iters,
                                             inv_iter=50)
    mapping = sdr.optimize(static, moving)
    # Smoke testing of plots
    ff = regtools.plot_2d_diffeomorphic_map(mapping, 10)
    # Default shape is static shape, moving shape
    npt.assert_equal(ff[0].shape, st_shape)
    npt.assert_equal(ff[1].shape, mv_shape)
    # Can specify shape
    ff = regtools.plot_2d_diffeomorphic_map(mapping,
                                            delta = 10,
                                            direct_grid_shape=(7, 8),
                                            inverse_grid_shape=(9, 10))
    npt.assert_equal(ff[0].shape, (7, 8))
    npt.assert_equal(ff[1].shape, (9, 10))
