import numpy as np
from dipy.viz import regtools
import numpy.testing as npt
from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration


def test_plot_2d_diffeomorphic_map():
    """
    Test the regtools plotting interface (lightly).
    """
    nn = 12 
    moving = np.random.rand(nn, nn)
    static = np.random.rand(nn, nn)
    dim = static.ndim
    metric = SSDMetric(dim) 
    level_iters = [200, 100, 50, 25]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter = 50)
    mapping = sdr.optimize(static, moving)
    
    # Smoke testing:
    ff = regtools.plot_2d_diffeomorphic_map(mapping, 10)
    npt.assert_equal(ff[0].shape, (nn, nn))
    ff = regtools.plot_2d_diffeomorphic_map(mapping, delta = 10,
                                            direct_grid_shape=(10, 10),
                                            inverse_grid_shape=(10, 10))
    npt.assert_equal(ff[0].shape, (nn, nn))
