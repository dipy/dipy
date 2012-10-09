import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from ..odf import (DiscreteDirectionFinder, OdfFit, OdfModel, gfa,
                   peaks_from_model, peak_directions)
from dipy.core.subdivide_octahedron import create_unit_hemisphere
from nose.tools import (assert_almost_equal, assert_equal, assert_raises,
                        assert_true)

def test_DiscreteDirectionFinder():
    def discrete_eval(sphere):
        X = np.zeros(len(sphere.phi))
        X[0] = 1.
        X[1] = .3
        return X

    sphere = create_unit_hemisphere(3)

    ddf = DiscreteDirectionFinder()
    ddf.config(sphere=sphere, relative_peak_threshold=.5, min_separation_angle=45)
    direction = ddf(discrete_eval)
    assert_array_almost_equal(direction, sphere.vertices[:1])

    ddf.config(relative_peak_threshold=.2)
    direction = ddf(discrete_eval)
    assert_array_almost_equal(direction, sphere.vertices[:2])

_sphere = create_unit_hemisphere(4)
_odf = (_sphere.vertices * [1, 2, 3]).sum(-1)
class SimpleOdfModel(OdfModel):
    sphere = _sphere
    def fit(self, data):
        fit = SimpleOdfFit()
        fit.model = self
        return fit

class SimpleOdfFit(OdfFit):
    def odf(self, sphere=None):
        if sphere is None:
            sphere = self.model.sphere

        # Use ascontiguousarray to work around a bug in NumPy
        return np.ascontiguousarray((sphere.vertices * [1, 2, 3]).sum(-1))

def test_OdfFit():
    m = SimpleOdfModel()
    m.direction_finder.config(sphere=_sphere)
    f = m.fit(None)
    argmax = _odf.argmax()
    assert_array_almost_equal(f.directions, _sphere.vertices[argmax:argmax+1])

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
    pam = peaks_from_model(model, data, _sphere, .5, 45, normalize_peaks=True)

    assert_array_equal(pam.gfa, gfa(_odf))
    assert_array_equal(pam.peak_values[:, 0], 1.)
    assert_array_equal(pam.peak_values[:, 1:], 0.)
    mn, mx = _odf.min(), _odf.max()
    assert_array_equal(pam.qa[:, 0], (mx - mn) / mx)
    assert_array_equal(pam.qa[:, 1:], 0.)
    assert_array_equal(pam.peak_indices[:, 0], odf_argmax)
    assert_array_equal(pam.peak_indices[:, 1:], -1)

    # Test that odf array matches and is right shape
    pam = peaks_from_model(model, data, _sphere, .5, 45, return_odf=True)
    expected_shape = (len(data), len(_odf))
    assert_equal(pam.odf.shape, expected_shape)
    assert_true((_odf == pam.odf).all())
    assert_array_equal(pam.peak_values[:, 0], _odf.max())

    # Test mask
    mask = (np.arange(10) % 2) == 1

    pam = peaks_from_model(model, data, _sphere, .5, 45, mask=mask,
                           normalize_peaks=True)
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

