import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           run_module_suite,
                           assert_array_equal,
                           assert_raises)
from dipy.data import get_data, dsi_deconv_voxels
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.reconst.odf import gfa
from dipy.direction.peaks import peak_directions
from dipy.sims.voxel import SticksAndBall
from dipy.core.sphere import Sphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from numpy.testing import assert_equal
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.tests.test_dsi import sticks_and_ball_dummies


def test_dsi():
    # load symmetric 724 sphere
    sphere = get_sphere('symmetric724')
    # load icosahedron sphere
    sphere2 = create_unit_sphere(5)
    btable = np.loadtxt(get_data('dsi515btable'))
    gtab = gradient_table(btable[:, 0], btable[:, 1:])
    data, golden_directions = SticksAndBall(gtab, d=0.0015,
                                            S0=100, angles=[(0, 0), (90, 0)],
                                            fractions=[50, 50], snr=None)

    ds = DiffusionSpectrumDeconvModel(gtab)

    # symmetric724
    dsfit = ds.fit(data)
    odf = dsfit.odf(sphere)
    directions, _, _ = peak_directions(odf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions),
                        2, 1)

    # 5 subdivisions
    dsfit = ds.fit(data)
    odf2 = dsfit.odf(sphere2)
    directions, _, _ = peak_directions(odf2, sphere2, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions),
                        2, 1)

    assert_equal(dsfit.pdf().shape, 3 * (ds.qgrid_size, ))
    sb_dummies = sticks_and_ball_dummies(gtab)
    for sbd in sb_dummies:
        data, golden_directions = sb_dummies[sbd]
        odf = ds.fit(data).odf(sphere2)
        directions, _, _ = peak_directions(odf, sphere2, .35, 25)
        if len(directions) <= 3:
            assert_equal(len(directions), len(golden_directions))
        if len(directions) > 3:
            assert_equal(gfa(odf) < 0.1, True)

    assert_raises(ValueError, DiffusionSpectrumDeconvModel, gtab, qgrid_size=16)


def test_multivox_dsi():
    data, gtab = dsi_deconv_voxels()
    DS = DiffusionSpectrumDeconvModel(gtab)
    sphere = get_sphere('symmetric724')

    DSfit = DS.fit(data)
    PDF = DSfit.pdf()
    assert_equal(data.shape[:-1] + (35, 35, 35), PDF.shape)
    assert_equal(np.alltrue(np.isreal(PDF)), True)


if __name__ == '__main__':
    run_module_suite()
