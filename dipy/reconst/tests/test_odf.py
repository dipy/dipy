import numpy as np
from numpy.testing import assert_array_equal
from ..odf import OdfFit, OdfModel, gfa, peaksFromModel
from dipy.core.triangle_subdivide import (create_half_unit_sphere,
    disperse_charges)
from nose.tools import (assert_almost_equal, assert_equal, assert_raises,
                        assert_true)

_sphere = create_half_unit_sphere(4)
_odf = (_sphere.vertices * [1, 2, 3]).sum(-1)
class SimpleOdfModel(object):
    sphere = _sphere
    def fit(self, data):
        fit = SimpleOdfFit()
        fit.model = self
        return fit

class SimpleOdfFit(object):
    def odf(self, sphere=None):
        if sphere is None:
            sphere = self.model.sphere
        return (sphere.vertices * [1, 2, 3]).sum(-1)

def test_peaksFromModel():
    data = np.zeros((10,2))

    # Test basic case
    model = SimpleOdfModel()
    pam = peaksFromModel(model, data, normalize_peaks=True)

    assert_array_equal(pam.gfa, gfa(_odf))
    assert_array_equal(pam.peak_values[:, 0], 1.)
    assert_array_equal(pam.peak_values[:, 1:], 0.)
    mn, mx = _odf.min(), _odf.max()
    assert_array_equal(pam.qa[:, 0], (mx - mn) / mx)
    assert_array_equal(pam.qa[:, 1:], 0.)
    assert_array_equal(pam.peak_indices[:, 0], 53)
    assert_array_equal(pam.peak_indices[:, 1:], -1)

    # Test that odf array matches and is right shape
    pam = peaksFromModel(model, data, return_odf=True)
    expected_shape = (len(data), len(_odf))
    assert_equal(pam.odf.shape, expected_shape)
    assert_true((_odf == pam.odf).all())
    assert_array_equal(pam.peak_values[:, 0], _odf.max())
   
    # Test mask
    mask = (np.arange(10) % 2) == 1

    pam = peaksFromModel(model, data, mask=mask, normalize_peaks=True)
    assert_array_equal(pam.gfa[~mask], 0)
    assert_array_equal(pam.qa[~mask], 0)
    assert_array_equal(pam.peak_values[~mask], 0)
    assert_array_equal(pam.peak_indices[~mask], -1)
    
    assert_array_equal(pam.gfa[mask], gfa(_odf))
    assert_array_equal(pam.peak_values[mask, 0], 1.)
    assert_array_equal(pam.peak_values[mask, 1:], 0.)
    mn, mx = _odf.min(), _odf.max()
    assert_array_equal(pam.qa[mask, 0], (mx - mn) / mx)
    assert_array_equal(pam.qa[mask, 1:], 0.)
    assert_array_equal(pam.peak_indices[mask, 0], 53)
    assert_array_equal(pam.peak_indices[mask, 1:], -1)

