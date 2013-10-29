import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, run_module_suite, assert_)
from dipy.reconst.odf import (OdfFit, OdfModel, gfa, peaks_from_model,
                              peak_directions, peak_directions_nl,
                              minmax_normalize, PeaksAndMetrics,
                              reshape_peaks_for_visualisation)
from dipy.core.subdivide_octahedron import create_unit_hemisphere
from dipy.core.sphere import unit_icosahedron
from dipy.sims.voxel import multi_tensor, multi_tensor_odf
from dipy.data import get_sphere, get_data
from dipy.core.gradients import gradient_table
from nose.tools import assert_equal, assert_true


def test_peak_directions_nl():
    def discrete_eval(sphere):
        return abs(sphere.vertices).sum(-1)

    directions, values = peak_directions_nl(discrete_eval)
    assert_equal(directions.shape, (4, 3))
    assert_array_almost_equal(abs(directions), 1 / np.sqrt(3))
    assert_array_equal(values, abs(directions).sum(-1))

    # Test using a different sphere
    sphere = unit_icosahedron.subdivide(4)
    directions, values = peak_directions_nl(discrete_eval, sphere=sphere)
    assert_equal(directions.shape, (4, 3))
    assert_array_almost_equal(abs(directions), 1 / np.sqrt(3))
    assert_array_equal(values, abs(directions).sum(-1))

    # Test the relative_peak_threshold
    def discrete_eval(sphere):
        A = abs(sphere.vertices).sum(-1)
        x, y, z = sphere.vertices.T
        B = 1 + (x * z > 0) + 2 * (y * z > 0)
        return A * B

    directions, values = peak_directions_nl(discrete_eval, .01)
    assert_equal(directions.shape, (4, 3))

    directions, values = peak_directions_nl(discrete_eval, .3)
    assert_equal(directions.shape, (3, 3))

    directions, values = peak_directions_nl(discrete_eval, .6)
    assert_equal(directions.shape, (2, 3))

    directions, values = peak_directions_nl(discrete_eval, .8)
    assert_equal(directions.shape, (1, 3))
    assert_almost_equal(values, 4 * 3 / np.sqrt(3))

    # Test odfs with large areas of zero
    def discrete_eval(sphere):
        A = abs(sphere.vertices).sum(-1)
        x, y, z = sphere.vertices.T
        B = (x * z > 0) + 2 * (y * z > 0)
        return A * B

    directions, values = peak_directions_nl(discrete_eval, 0.)
    assert_equal(directions.shape, (3, 3))

    directions, values = peak_directions_nl(discrete_eval, .6)
    assert_equal(directions.shape, (2, 3))

    directions, values = peak_directions_nl(discrete_eval, .8)
    assert_equal(directions.shape, (1, 3))
    assert_almost_equal(values, 3 * 3 / np.sqrt(3))

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
    f = m.fit(None)
    odf = f.odf(_sphere)
    assert_equal(len(odf), len(_sphere.theta))


def test_peak_directions():
    model = SimpleOdfModel()
    fit = model.fit(None)
    odf = fit.odf()

    argmax = odf.argmax()
    mx = odf.max()
    sphere = fit.model.sphere

    # Only one peak
    dir, val, ind = peak_directions(odf, sphere, .5, 45)
    dir_e = sphere.vertices[[argmax]]
    assert_array_equal(ind, [argmax])
    assert_array_equal(val, odf[ind])
    assert_array_equal(dir, dir_e)

    odf[0] = mx * .9
    # Two peaks, relative_threshold
    dir, val, ind = peak_directions(odf, sphere, 1., 0)
    dir_e = sphere.vertices[[argmax]]
    assert_array_equal(dir, dir_e)
    assert_array_equal(ind, [argmax])
    assert_array_equal(val, odf[ind])
    dir, val, ind = peak_directions(odf, sphere, .8, 0)
    dir_e = sphere.vertices[[argmax, 0]]
    assert_array_equal(dir, dir_e)
    assert_array_equal(ind, [argmax, 0])
    assert_array_equal(val, odf[ind])

    # Two peaks, angle_sep
    dir, val, ind = peak_directions(odf, sphere, 0., 90)
    dir_e = sphere.vertices[[argmax]]
    assert_array_equal(dir, dir_e)
    assert_array_equal(ind, [argmax])
    assert_array_equal(val, odf[ind])
    dir, val, ind = peak_directions(odf, sphere, 0., 0)
    dir_e = sphere.vertices[[argmax, 0]]
    assert_array_equal(dir, dir_e)
    assert_array_equal(ind, [argmax, 0])
    assert_array_equal(val, odf[ind])


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


def test_peaksFromModelParallel():
    SNR = 100
    S0 = 100

    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    data, _ = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (60, 0)],
                           fractions=[50, 50], snr=SNR)

    # test equality with/without multiprocessing
    model = SimpleOdfModel()
    pam_multi = peaks_from_model(model, data, _sphere, .5, 45,
                                 normalize_peaks=True, return_odf=True,
                                 return_sh=True, parallel=True)

    pam_single = peaks_from_model(model, data, _sphere, .5, 45,
                                  normalize_peaks=True, return_odf=True,
                                  return_sh=True, parallel=False)

    assert_array_almost_equal(pam_multi.gfa, pam_single.gfa)
    assert_array_almost_equal(pam_multi.qa, pam_single.qa)
    assert_array_almost_equal(pam_multi.peak_values, pam_single.peak_values)
    assert_array_equal(pam_multi.peak_indices, pam_single.peak_indices)
    assert_array_almost_equal(pam_multi.peak_dirs, pam_single.peak_dirs)
    assert_array_almost_equal(pam_multi.shm_coeff, pam_single.shm_coeff)
    assert_array_almost_equal(pam_multi.odf, pam_single.odf)


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
    evecs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
             np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])]

    S, sticks = multi_tensor(gtab, evals, S0, angles=[(0, 0), (90, 0)],
                             fractions=[50, 50], snr=SNR)
    odf = multi_tensor_odf(sphere.vertices, [0.5, 0.5], evals, evecs)

    odf2 = minmax_normalize(odf)
    assert_equal(odf2.max(), 1)
    assert_equal(odf2.min(), 0)

    odf3 = np.empty(odf.shape)
    odf3 = minmax_normalize(odf, odf3)
    assert_equal(odf3.max(), 1)
    assert_equal(odf3.min(), 0)


def test_peaks_shm_coeff():

    SNR = 100
    S0 = 100

    _, fbvals, fbvecs = get_data('small_64D')

    from dipy.data import get_sphere

    sphere = get_sphere('symmetric724')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    data, _ = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (60, 0)],
                           fractions=[50, 50], snr=SNR)

    from dipy.reconst.shm import CsaOdfModel

    model = CsaOdfModel(gtab, 4)

    pam = peaks_from_model(model, data[None, :], sphere, .5, 45,
                           return_odf=True, return_sh=True)
    # Test that spherical harmonic coefficients return back correctly
    B = np.linalg.pinv(pam.invB)
    odf2 = np.dot(pam.shm_coeff, B)
    assert_array_almost_equal(pam.odf, odf2)
    assert_equal(pam.shm_coeff.shape[-1], 45)

    pam = peaks_from_model(model, data[None, :], sphere, .5, 45,
                           return_odf=True, return_sh=False)
    assert_equal(pam.shm_coeff, None)

    pam = peaks_from_model(model, data[None, :], sphere, .5, 45,
                           return_odf=True, return_sh=True,
                           sh_basis_type='mrtrix')

    B = np.linalg.pinv(pam.invB)
    odf2 = np.dot(pam.shm_coeff, B)
    assert_array_almost_equal(pam.odf, odf2)


def test_reshape_peaks_for_visualisation():

    data1 = np.random.randn(10, 5, 3).astype('float32')
    data2 = np.random.randn(10, 2, 5, 3).astype('float32')
    data3 = np.random.randn(10, 2, 12, 5, 3).astype('float32')

    data1_reshape = reshape_peaks_for_visualisation(data1)
    data2_reshape = reshape_peaks_for_visualisation(data2)
    data3_reshape = reshape_peaks_for_visualisation(data3)

    assert_array_equal(data1_reshape.shape, (10, 15))
    assert_array_equal(data2_reshape.shape, (10, 2, 15))
    assert_array_equal(data3_reshape.shape, (10, 2, 12, 15))

    assert_array_equal(data1_reshape.reshape(10, 5, 3), data1)
    assert_array_equal(data2_reshape.reshape(10, 2, 5, 3), data2)
    assert_array_equal(data3_reshape.reshape(10, 2, 12, 5, 3), data3)


if __name__ == '__main__':

    run_module_suite()
