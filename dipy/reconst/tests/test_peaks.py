import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
                           assert_array_almost_equal)

from ..peaks import (DiscreteDirectionFinder, NonLinearDirectionFinder,
                     peaks_from_model, peak_directions)
from ..odf import OdfFit, OdfModel, gfa
from dipy.core.subdivide_octahedron import create_unit_hemisphere


_sphere = create_unit_hemisphere(4)
_direction = np.array([1, 2, 3]) / np.sqrt(14)
_odf = (_sphere.vertices * [1, 2, 3]).sum(-1)


def test_DiscreteDirectionFinder():
    def discrete_eval(sphere):
        X = np.zeros(len(sphere.phi))
        X[0] = 1.
        X[1] = .3
        return X

    ddf = DiscreteDirectionFinder(
        sphere=_sphere,
        relative_peak_threshold=0.5,
        min_separation_angle=45)

    direction = ddf._directions_from_sphere(discrete_eval)
    assert_array_almost_equal(direction, _sphere.vertices[:1])

    ddf = DiscreteDirectionFinder(
        sphere=_sphere,
        relative_peak_threshold=0.2,
        min_separation_angle=45)

    direction = ddf._directions_from_sphere(discrete_eval)
    assert_array_almost_equal(direction, _sphere.vertices[:2])


def test_NonLinearDirectionFinder():
    def discrete_eval(sphere):
        return abs(sphere.vertices).sum(-1)

    df = NonLinearDirectionFinder()
    directions = df._directions_from_sphere(discrete_eval)
    assert_equal(directions.shape, (4, 3))
    assert_array_almost_equal(abs(directions), 1 / np.sqrt(3))


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


def test_odffit_peak():
    m = SimpleOdfModel()
    f = m.fit(None)

    ddf = DiscreteDirectionFinder(sphere=_sphere)
    nldf = NonLinearDirectionFinder(sphere=create_unit_hemisphere(1))

    argmax = _odf.argmax()
    assert_array_almost_equal(ddf(f),
                              _sphere.vertices[argmax:argmax + 1])

    assert_array_almost_equal(nldf(f), _direction[None, :])


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
    data = np.zeros((10, 2))

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
    assert_((_odf == pam.odf).all())
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
