import numpy as np
from numpy.testing import assert_array_equal
from ..odf import OdfFit, OdfModel, gfa, peaks_from_model, peak_directions
from dipy.core.subdivide_octahedron import create_unit_hemisphere
from nose.tools import (assert_almost_equal, assert_equal, assert_raises,
                        assert_true)

_sphere = create_unit_hemisphere(4)
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

def test_peak_directions():
    model = SimpleOdfModel()
    fit = model.fit(None)
    odf = fit.odf()

    argmax = odf.argmax()
    mx = odf.max()
    sphere = fit.model.sphere

    # Only one peak
    dir = peak_directions(odf, sphere, .5, 45)
    dir_e = sphere.vertices[[argmax]]
    assert_array_equal(dir, dir_e)

    odf[0] = mx * .9
    # Two peaks, relative_threshold
    dir = peak_directions(odf, sphere, 1., 0)
    dir_e = sphere.vertices[[argmax]]
    assert_array_equal(dir, dir_e)
    dir = peak_directions(odf, sphere, .8, 0)
    dir_e = sphere.vertices[[argmax, 0]]
    assert_array_equal(dir, dir_e)

    # Two peaks, angle_sep
    dir = peak_directions(odf, sphere, 0., 90)
    dir_e = sphere.vertices[[argmax]]
    assert_array_equal(dir, dir_e)
    dir = peak_directions(odf, sphere, 0., 0)
    dir_e = sphere.vertices[[argmax, 0]]
    assert_array_equal(dir, dir_e)

def test_peaksFromModel():
    data = np.zeros((10,2))

    # Test basic case
    model = SimpleOdfModel()
    odf_argmax = _odf.argmax()
    pam = peaks_from_model(model, data, normalize_peaks=True)

    assert_array_equal(pam.gfa, gfa(_odf))
    assert_array_equal(pam.peak_values[:, 0], 1.)
    assert_array_equal(pam.peak_values[:, 1:], 0.)
    mn, mx = _odf.min(), _odf.max()
    assert_array_equal(pam.qa[:, 0], (mx - mn) / mx)
    assert_array_equal(pam.qa[:, 1:], 0.)
    assert_array_equal(pam.peak_indices[:, 0], odf_argmax)
    assert_array_equal(pam.peak_indices[:, 1:], -1)

    # Test that odf array matches and is right shape
    pam = peaks_from_model(model, data, return_odf=True)
    expected_shape = (len(data), len(_odf))
    assert_equal(pam.odf.shape, expected_shape)
    assert_true((_odf == pam.odf).all())
    assert_array_equal(pam.peak_values[:, 0], _odf.max())
   
    # Test mask
    mask = (np.arange(10) % 2) == 1

    pam = peaks_from_model(model, data, mask=mask, normalize_peaks=True)
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
    assert_array_equal(pam.peak_indices[mask, 0], odf_argmax)
    assert_array_equal(pam.peak_indices[mask, 1:], -1)

