import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, run_module_suite,
                           assert_equal, assert_)
from dipy.reconst.odf import (OdfFit, OdfModel, gfa)

from dipy.reconst.peaks import (peaks_from_model,
                                peak_directions,
                                peak_directions_nl,
                                reshape_peaks_for_visualization)
from dipy.core.subdivide_octahedron import create_unit_hemisphere
from dipy.core.sphere import unit_icosahedron
from dipy.sims.voxel import multi_tensor, all_tensor_evecs, multi_tensor_odf
from dipy.data import get_data, get_sphere
from dipy.core.gradients import gradient_table, GradientTable
from dipy.core.sphere_stats import angular_similarity
from dipy.core.sphere import HemiSphere


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
_gtab = GradientTable(np.ones((64, 3)))


class SimpleOdfModel(OdfModel):
    sphere = _sphere

    def fit(self, data):
        fit = SimpleOdfFit(self, data)
        fit.model = self
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


def test_peak_directions():
    model = SimpleOdfModel(_gtab)
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


def _create_mt_sim(mevals, angles, fractions, S0, SNR, half_sphere=False):

    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)

    S, sticks = multi_tensor(gtab, mevals, S0, angles=angles,
                             fractions=fractions, snr=SNR)

    sphere = get_sphere('symmetric724').subdivide(2)

    if half_sphere:

        sphere = HemiSphere.from_sphere(sphere)

    odf_gt = multi_tensor_odf(sphere.vertices, mevals,
                              angles=angles, fractions=fractions)

    return odf_gt, sticks, sphere


def test_peak_directions_thorough():

    # two equal fibers (creating a very sharp odf)
    mevals = np.array([[0.0025, 0.0003, 0.0003],
                       [0.0025, 0.0003, 0.0003]])
    angles = [(0, 0), (45, 0)]
    fractions = [50, 50]
    odf_gt, sticks, sphere = _create_mt_sim(mevals, angles, fractions,
                                            100, None)

    directions, values, indices = peak_directions(odf_gt, sphere, .5, 25.)
    assert_almost_equal(angular_similarity(directions, sticks), 2, 2)

    # two unequal fibers
    fractions = [75, 25]
    odf_gt, sticks, sphere = _create_mt_sim(mevals, angles, fractions,
                                            100, None)

    directions, values, indices = peak_directions(odf_gt, sphere, .5, 25.)
    assert_almost_equal(angular_similarity(directions, sticks), 1, 2)

    directions, values, indices = peak_directions(odf_gt, sphere, .20, 25.)
    assert_almost_equal(angular_similarity(directions, sticks), 2, 2)

    # two equal fibers short angle (simulating very sharp ODF)
    mevals = np.array(([0.0045, 0.0003, 0.0003],
                       [0.0045, 0.0003, 0.0003]))
    fractions = [50, 50]
    angles = [(0, 0), (20, 0)]
    odf_gt, sticks, sphere = _create_mt_sim(mevals, angles,
                                            fractions, 100, None)

    directions, values, indices = peak_directions(odf_gt, sphere, .5, 25.)
    assert_almost_equal(angular_similarity(directions, sticks), 1, 2)

    directions, values, indices = peak_directions(odf_gt, sphere, .5, 15.)

    assert_almost_equal(angular_similarity(directions, sticks), 2, 2)

    # 1 fiber
    mevals = np.array([[0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]])
    fractions = [50, 50]
    angles = [(15, 0), (15, 0)]
    odf_gt, sticks, sphere = _create_mt_sim(mevals, angles,
                                            fractions, 100, None)

    directions, values, indices = peak_directions(odf_gt, sphere, .5, 15.)
    assert_almost_equal(angular_similarity(directions, sticks), 1, 2)

    AE = np.rad2deg(np.arccos(np.dot(directions[0], sticks[0])))
    assert_(abs(AE) < 2. or abs(AE - 180) < 2.)

    # two equal fibers and one small noisy one
    mevals = np.array([[0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (45, 0), (90, 0)]
    fractions = [45, 45, 10]
    odf_gt, sticks, sphere = _create_mt_sim(mevals, angles, fractions, 100, None)

    directions, values, indices = peak_directions(odf_gt, sphere, .5, 25.)
    assert_almost_equal(angular_similarity(directions, sticks), 2, 2)

    # two equal fibers and one faulty
    mevals = np.array([[0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (45, 0), (60, 0)]
    fractions = [45, 45, 10]
    odf_gt, sticks, sphere = _create_mt_sim(mevals, angles, fractions, 100, None)

    directions, values, indices = peak_directions(odf_gt, sphere, .5, 25.)
    assert_almost_equal(angular_similarity(directions, sticks), 2, 2)

    # two equal fibers and one very very annoying one
    mevals = np.array([[0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (45, 0), (60, 0)]
    fractions = [40, 40, 20]
    odf_gt, sticks, sphere = _create_mt_sim(mevals, angles, fractions, 100, None)

    directions, values, indices = peak_directions(odf_gt, sphere, .5, 25.)
    assert_almost_equal(angular_similarity(directions, sticks), 2, 2)

    # three peaks and one faulty
    mevals = np.array([[0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (45, 0), (90, 0), (90, 45)]
    fractions = [35, 35, 20, 10]
    odf_gt, sticks, sphere = _create_mt_sim(mevals, angles, fractions, 100, None)

    directions, values, indices = peak_directions(odf_gt, sphere, .5, 25.)
    assert_almost_equal(angular_similarity(directions, sticks), 3, 2)

    # four peaks
    mevals = np.array([[0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (45, 0), (90, 0), (90, 45)]
    fractions = [25, 25, 25, 25]
    odf_gt, sticks, sphere = _create_mt_sim(mevals, angles, fractions, 100, None)

    directions, values, indices = peak_directions(odf_gt, sphere, .15, 5.)
    assert_almost_equal(angular_similarity(directions, sticks), 4, 2)

    # four difficult peaks
    mevals = np.array([[0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (45, 0), (90, 0), (90, 45)]
    fractions = [30, 30, 20, 20]
    odf_gt, sticks, sphere = _create_mt_sim(mevals, angles, fractions, 100, None)

    directions, values, indices = peak_directions(odf_gt, sphere, 0, 0)
    assert_almost_equal(angular_similarity(directions, sticks), 4, 1)

    odf_gt, sticks, hsphere = _create_mt_sim(mevals, angles, fractions,
                                             100, None, half_sphere=True)

    directions, values, indices = peak_directions(odf_gt, hsphere, 0, 0)
    assert_equal(angular_similarity(directions, sticks) < 4, True)

    # four peaks and one them quite small
    fractions = [35, 35, 20, 10]

    odf_gt, sticks, sphere = _create_mt_sim(mevals, angles, fractions, 100, None)

    directions, values, indices = peak_directions(odf_gt, sphere, 0, 0)
    assert_equal(angular_similarity(directions, sticks) < 4, True)

    odf_gt, sticks, hsphere = _create_mt_sim(mevals, angles, fractions,
                                             100, None, half_sphere=True)

    directions, values, indices = peak_directions(odf_gt, hsphere, 0, 0)
    assert_equal(angular_similarity(directions, sticks) < 4, True)

    # isotropic case
    mevals = np.array([[0.0015, 0.0015, 0.0015]])
    angles = [(0, 0)]
    fractions = [100.]
    odf_gt, sticks, sphere = _create_mt_sim(mevals, angles, fractions,
                                            100, None)

    directions, values, indices = peak_directions(odf_gt, sphere, .5, 25.)
    assert_equal(len(values) > 10, True)


def test_difference_with_minmax():

    # Show difference with and without minmax normalization
    # we create an odf here with 3 main peaks, 1 small sharp unwanted peak
    # (noise) and an isotropic compartment.
    mevals = np.array([[0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003],
                       [0.0015, 0.00005, 0.00005],
                       [0.0015, 0.0015, 0.0015]])
    angles = [(0, 0), (45, 0), (90, 0), (90, 90), (0, 0)]
    fractions = [20, 20, 10, 1, 100 - 20 - 20 - 10 - 1]
    odf_gt, sticks, sphere = _create_mt_sim(mevals, angles, fractions,
                                            100, None)

    # We will show that when the minmax normalization is used we can remove
    # the noisy peak using a lower threshold.

    odf_gt_minmax = (odf_gt - odf_gt.min()) / (odf_gt.max() - odf_gt.min())

    _, values_1, _ = peak_directions(odf_gt, sphere, .30, 25.)

    assert_equal(len(values_1), 3)

    _, values_2, _ = peak_directions(odf_gt_minmax, sphere, .30, 25.)

    assert_equal(len(values_2), 3)

    # Setting the smallest value of the odf to zero is like running
    # peak_directions without the odf_min correction.
    odf_gt[odf_gt.argmin()] = 0.
    _, values_3, _ = peak_directions(odf_gt, sphere, .30, 25.,)

    assert_equal(len(values_3), 4)

    # we show here that to actually get that noisy peak out we need to
    # increase the peak threshold considerably
    directions, values_4, indices = peak_directions(odf_gt, sphere, .60, 25.,)

    assert_equal(len(values_4), 3)
    assert_almost_equal(values_1, values_4)


def test_degenerative_cases():

    sphere = get_sphere('symmetric724')

    # completely isotropic and degencase
    odf = np.zeros(sphere.vertices.shape[0])
    directions, values, indices = peak_directions(odf, sphere, .5, 25)
    print(directions, values, indices)

    assert_equal(len(values), 0)
    assert_equal(len(directions), 0)
    assert_equal(len(indices), 0)

    odf = np.zeros(sphere.vertices.shape[0])
    odf[0] = 0.020
    odf[1] = 0.018

    directions, values, indices = peak_directions(odf, sphere, .5, 25)
    print(directions, values, indices)

    assert_equal(values[0], 0.02)

    odf = - np.ones(sphere.vertices.shape[0])
    directions, values, indices = peak_directions(odf, sphere, .5, 25)
    print(directions, values, indices)

    assert_equal(len(values), 0)

    odf = np.zeros(sphere.vertices.shape[0])
    odf[0] = 0.020
    odf[1] = 0.018
    odf[2] = - 0.018

    directions, values, indices = peak_directions(odf, sphere, .5, 25)
    assert_equal(values[0], 0.02)

    odf = np.ones(sphere.vertices.shape[0])
    odf += 0.1 * np.random.rand(odf.shape[0])
    directions, values, indices = peak_directions(odf, sphere, .5, 25)
    assert_(all(values > values[0] * .5))
    assert_array_equal(values, odf[indices])

    odf = np.ones(sphere.vertices.shape[0])
    odf[1:] = np.finfo(np.float).eps * np.random.rand(odf.shape[0] - 1)
    directions, values, indices = peak_directions(odf, sphere, .5, 25)

    assert_equal(values[0], 1)
    assert_equal(len(values), 1)


def test_peaksFromModel():
    data = np.zeros((10, 2))

    # Test basic case
    model = SimpleOdfModel(_gtab)
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
    model = SimpleOdfModel(gtab)
    pam_multi = peaks_from_model(model, data, _sphere, .5, 45,
                                 normalize_peaks=True, return_odf=True,
                                 return_sh=True, parallel=True)

    pam_single = peaks_from_model(model, data, _sphere, .5, 45,
                                  normalize_peaks=True, return_odf=True,
                                  return_sh=True, parallel=False)

    assert_equal(pam_multi.gfa.dtype, pam_single.gfa.dtype)
    assert_equal(pam_multi.gfa.shape, pam_single.gfa.shape)
    assert_array_almost_equal(pam_multi.gfa, pam_single.gfa)

    assert_equal(pam_multi.qa.dtype, pam_single.qa.dtype)
    assert_equal(pam_multi.qa.shape, pam_single.qa.shape)
    assert_array_almost_equal(pam_multi.qa, pam_single.qa)

    assert_equal(pam_multi.peak_values.dtype, pam_single.peak_values.dtype)
    assert_equal(pam_multi.peak_values.shape, pam_single.peak_values.shape)
    assert_array_almost_equal(pam_multi.peak_values, pam_single.peak_values)

    assert_equal(pam_multi.peak_indices.dtype, pam_single.peak_indices.dtype)
    assert_equal(pam_multi.peak_indices.shape, pam_single.peak_indices.shape)
    assert_array_equal(pam_multi.peak_indices, pam_single.peak_indices)

    assert_equal(pam_multi.peak_dirs.dtype, pam_single.peak_dirs.dtype)
    assert_equal(pam_multi.peak_dirs.shape, pam_single.peak_dirs.shape)
    assert_array_almost_equal(pam_multi.peak_dirs, pam_single.peak_dirs)

    assert_equal(pam_multi.shm_coeff.dtype, pam_single.shm_coeff.dtype)
    assert_equal(pam_multi.shm_coeff.shape, pam_single.shm_coeff.shape)
    assert_array_almost_equal(pam_multi.shm_coeff, pam_single.shm_coeff)

    assert_equal(pam_multi.odf.dtype, pam_single.odf.dtype)
    assert_equal(pam_multi.odf.shape, pam_single.odf.shape)
    assert_array_almost_equal(pam_multi.odf, pam_single.odf)


def test_peaks_shm_coeff():

    SNR = 100
    S0 = 100

    _, fbvals, fbvecs = get_data('small_64D')

    from dipy.data import get_sphere

    sphere = get_sphere('repulsion724')

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
    odf2 = np.dot(pam.shm_coeff, pam.B)
    assert_array_almost_equal(pam.odf, odf2)
    assert_equal(pam.shm_coeff.shape[-1], 45)

    pam = peaks_from_model(model, data[None, :], sphere, .5, 45,
                           return_odf=True, return_sh=False)
    assert_equal(pam.shm_coeff, None)

    pam = peaks_from_model(model, data[None, :], sphere, .5, 45,
                           return_odf=True, return_sh=True,
                           sh_basis_type='mrtrix')

    odf2 = np.dot(pam.shm_coeff, pam.B)
    assert_array_almost_equal(pam.odf, odf2)


def test_reshape_peaks_for_visualization():

    data1 = np.random.randn(10, 5, 3).astype('float32')
    data2 = np.random.randn(10, 2, 5, 3).astype('float32')
    data3 = np.random.randn(10, 2, 12, 5, 3).astype('float32')

    data1_reshape = reshape_peaks_for_visualization(data1)
    data2_reshape = reshape_peaks_for_visualization(data2)
    data3_reshape = reshape_peaks_for_visualization(data3)

    assert_array_equal(data1_reshape.shape, (10, 15))
    assert_array_equal(data2_reshape.shape, (10, 2, 15))
    assert_array_equal(data3_reshape.shape, (10, 2, 12, 15))

    assert_array_equal(data1_reshape.reshape(10, 5, 3), data1)
    assert_array_equal(data2_reshape.reshape(10, 2, 5, 3), data2)
    assert_array_equal(data3_reshape.reshape(10, 2, 12, 5, 3), data3)


if __name__ == '__main__':

    run_module_suite()

