import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_
from dipy.reconst.odf import (OdfFit, OdfModel, minmax_normalize, gfa)

from dipy.core.subdivide_octahedron import create_unit_hemisphere
from dipy.sims.voxel import multi_tensor, multi_tensor_odf
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table, GradientTable


_sphere = create_unit_hemisphere(4)
_odf = (_sphere.vertices * [1, 2, 3]).sum(-1)
_gtab = GradientTable(np.ones((64, 3)))


class SimpleOdfModel(OdfModel):
    sphere = _sphere

    def fit(self, data):
        fit = SimpleOdfFit(self, data)
        return fit


class SimpleOdfFit(OdfFit):

    def odf(self, sphere=None):
        if sphere is None:
            sphere = self.model.sphere

        # Use ascontiguousarray to work around a bug in NumPy
        return np.ascontiguousarray((sphere.vertices * [1, 2, 3]).sum(-1))


def test_OdfFit():
    m = SimpleOdfModel(_gtab)
    f = m.fit(None)
    odf = f.odf(_sphere)
    assert_equal(len(odf), len(_sphere.theta))


def test_minmax_normalize():

    bvalue = 3000
    S0 = 1
    SNR = 100

    sphere = get_sphere('symmetric362')
    bvecs = np.concatenate(([[0, 0, 0]], sphere.vertices))
    bvals = np.zeros(len(bvecs)) + bvalue
    bvals[0] = 0
    gtab = gradient_table(bvals, bvecs)

    evals = np.array(([0.0017, 0.0003, 0.0003], [0.0017, 0.0003, 0.0003]))

    multi_tensor(gtab, evals, S0, angles=[(0, 0), (90, 0)],
                 fractions=[50, 50], snr=SNR)
    odf = multi_tensor_odf(sphere.vertices, evals, angles=[(0, 0), (90, 0)],
                           fractions=[50, 50])

    odf2 = minmax_normalize(odf)
    assert_equal(odf2.max(), 1)
    assert_equal(odf2.min(), 0)

    odf3 = np.empty(odf.shape)
    odf3 = minmax_normalize(odf, odf3)
    assert_equal(odf3.max(), 1)
    assert_equal(odf3.min(), 0)


def test_gfa():
    g = gfa(np.ones(100))
    assert_equal(g, 0)

    g = gfa(np.ones((2, 100)))
    assert_equal(g, np.array([0, 0]))

    # The following series follows the rule (sqrt(n-1)/((n-1)^2))
    g = gfa(np.hstack([np.ones(9), [0]]))
    assert_almost_equal(g, np.sqrt(9./81))
    g = gfa(np.hstack([np.ones(99), [0]]))
    assert_almost_equal(g, np.sqrt(99./(99.**2)))

    # All-zeros returns a nan with no warning:
    g = gfa(np.zeros(10))
    assert_(np.isnan(g))
