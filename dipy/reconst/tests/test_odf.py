import numpy as np
from numpy.testing import assert_array_equal
from ..odf import OdfFit, OdfModel, gfa
from dipy.core.triangle_subdivide import (create_half_unit_sphere,
    disperse_charges)
from nose.tools import (assert_almost_equal, assert_equal, assert_raises,
                        assert_true)

class SimpleOdfModel(OdfModel):
    def __init__(self):
        v, e, f = create_half_unit_sphere(4)
        self.set_odf_vertices(v, e)
        self.odf = (v * [1, 2, 3]).sum(-1)

    def evaluate_odf(self, signal):
        return self.odf

def test_OdfModelfit():
    data = np.zeros((10,2))

    # Test basic case
    model = SimpleOdfModel()
    odffit = model.fit(data, normalize_peaks=1)
    assert_array_equal(odffit.gfa, gfa(model.odf))
    assert_array_equal(odffit.peak_values[:, 0], 1.)
    assert_array_equal(odffit.peak_values[:, 1:], 0.)
    mn, mx = model.odf.min(), model.odf.max()
    assert_array_equal(odffit.qa[:, 0], (mx - mn) / mx)
    assert_array_equal(odffit.qa[:, 1:], 0.)
    assert_array_equal(odffit.peak_indices[:, 0], 53)
    assert_array_equal(odffit.peak_indices[:, 1:], -1)

    # Test that odf array matches and is right shape
    odffit = model.fit(data, return_odf=True)
    expected_shape = (len(data), len(model.odf))
    assert_equal(odffit.odf.shape, expected_shape)
    assert_true((model.odf == odffit.odf).all())
    assert_array_equal(odffit.peak_values[:, 0], model.odf.max())
   
    # Test mask
    mask = (np.arange(10) % 2) == 1
    odffit = model.fit(data, mask=mask, normalize_peaks=True)
    assert_array_equal(odffit.gfa[~mask], 0)
    assert_array_equal(odffit.qa[~mask], 0)
    assert_array_equal(odffit.peak_values[~mask], 0)
    assert_array_equal(odffit.peak_indices[~mask], -1)
    
    assert_array_equal(odffit.gfa[mask], gfa(model.odf))
    assert_array_equal(odffit.peak_values[mask, 0], 1.)
    assert_array_equal(odffit.peak_values[mask, 1:], 0.)
    mn, mx = model.odf.min(), model.odf.max()
    assert_array_equal(odffit.qa[mask, 0], (mx - mn) / mx)
    assert_array_equal(odffit.qa[mask, 1:], 0.)
    assert_array_equal(odffit.peak_indices[mask, 0], 53)
    assert_array_equal(odffit.peak_indices[mask, 1:], -1)


def test_OdfModelgetpeaks():
    model = SimpleOdfModel()
    peaks = model.get_peaks(None)
    assert_array_equal(peaks, model.odf_vertices[[53]])


class TestOdfModel():

    def test_angular_distance(self):
        model = SimpleOdfModel()
        assert_almost_equal(model.angular_distance_threshold, 45)
        model.angular_distance_threshold = 60
        assert_almost_equal(model.angular_distance_threshold, 60)
        assert_almost_equal(model._cos_distance_threshold, .5)

    def test_set_odf_vertices(self):
        model = OdfModel()
        v, e, f = create_half_unit_sphere(4)
        model.set_odf_vertices(v, e)
        assert_array_equal(v, model.odf_vertices)
        assert_array_equal(e, model.odf_edges)
        assert_array_equal(abs(v.dot(v.T)), model._distance_matrix)
        assert_raises(ValueError, model.set_odf_vertices, v[:, :2], e)
        assert_raises(ValueError, model.set_odf_vertices, v/2, e)

