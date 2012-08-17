
import numpy as np
import numpy.linalg as npl
from dipy.core.subdivide_octahedron import create_unit_hemisphere
from dipy.reconst.dti import design_matrix, lower_triangular

from nose.tools import assert_equal, assert_raises, assert_true, assert_false
from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import (real_sph_harm, sph_harm_ind_list, _closest_peak,
                              SlowAdcOpdfModel, normalize_data,
                              ClosestPeakSelector, QballOdfModel, hat,
                              lcr_matrix, smooth_pinv, bootstrap_data_array,
                              bootstrap_data_voxel, ResidualBootstrapWrapper)

def test_sph_harm_ind_list():
    m_list, n_list = sph_harm_ind_list(8)
    assert_equal(m_list.shape, n_list.shape)
    assert_equal(m_list.shape, (45,))
    assert_true(np.all(np.abs(m_list) <= n_list))
    assert_array_equal(n_list % 2, 0)
    assert_raises(ValueError, sph_harm_ind_list, 1)

def test_real_sph_harm():
    # Tests derived from tables in
    # http://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    # where real spherical harmonic $Y^m_n$ is defined to be:
    #    Real($Y^m_n$) * sqrt(2) if m > 0
    #    $Y^m_n$                 if m == 0
    #    Imag($Y^m_n$) * sqrt(2) if m < 0

    rsh = real_sph_harm
    pi = np.pi
    exp = np.exp
    sqrt = np.sqrt
    sin = np.sin
    cos = np.cos
    assert_array_almost_equal(rsh(0,0,0,0),
           0.5/sqrt(pi))
    assert_array_almost_equal(rsh(2,2,pi/3,pi/5),
           0.25*sqrt(15./(2.*pi))*
           (sin(pi/5.))**2.*cos(0+2.*pi/3)*sqrt(2))
    assert_array_almost_equal(rsh(-2,2,pi/3,pi/5),
           0.25*sqrt(15./(2.*pi))*
           (sin(pi/5.))**2.*sin(0-2.*pi/3)*sqrt(2))
    assert_array_almost_equal(rsh(2,2,pi,pi/2),
           0.25*sqrt(15/(2.*pi))*
           cos(2.*pi)*sin(pi/2.)**2.*sqrt(2))
    assert_array_almost_equal(rsh(-2,4,pi/4.,pi/3.),
           (3./8.)*sqrt(5./(2.*pi))*
           sin(0-2.*pi/4.)*
           sin(pi/3.)**2.*
           (7.*cos(pi/3.)**2.-1)*sqrt(2))
    assert_array_almost_equal(rsh(4,4,pi/8.,pi/6.),
           (3./16.)*sqrt(35./(2.*pi))*
           cos(0+4.*pi/8.)*sin(pi/6.)**4.*sqrt(2))
    assert_array_almost_equal(rsh(-4,4,pi/8.,pi/6.),
           (3./16.)*sqrt(35./(2.*pi))*
           sin(0-4.*pi/8.)*sin(pi/6.)**4.*sqrt(2))
    aa = np.ones((3,1,1,1))
    bb = np.ones((1,4,1,1))
    cc = np.ones((1,1,5,1))
    dd = np.ones((1,1,1,6))
    assert_equal(rsh(aa, bb, cc, dd).shape, (3, 4, 5, 6))

def test_closest_peak():
    peak_values = np.array([1, .9, .8, .7, .6, .2, .1])
    peak_points = np.array([[1., 0., 0.],
                            [0., .9, .1],
                            [0., 1., 0.],
                            [.9, .1, 0.],
                            [0., 0., 1.],
                            [1., 1., 0.],
                            [0., 1., 1.]])
    norms = np.sqrt((peak_points*peak_points).sum(-1))
    peak_points = peak_points/norms[:, None]

    prev = np.array([1, -.9, 0])
    prev = prev/np.sqrt(np.dot(prev, prev))
    cp = _closest_peak(peak_points, prev)
    assert_array_equal(cp, peak_points[0])
    cp = _closest_peak(peak_points, -prev)
    assert_array_equal(cp, -peak_points[0])


def test_smooth_pinv():
    hemi = create_unit_hemisphere(3)
    m, n = sph_harm_ind_list(4)
    B = real_sph_harm(m, n, hemi.phi[:, None], hemi.theta[:, None])

    L = np.zeros(len(m))
    C = smooth_pinv(B, L)
    D = np.dot(npl.inv(np.dot(B.T, B)), B.T)
    assert_array_almost_equal(C, D)

    L = n*(n+1)*.05
    C = smooth_pinv(B, L)
    L = np.diag(L)
    D = np.dot(npl.inv(np.dot(B.T, B) + L*L), B.T)

    assert_array_almost_equal(C, D)

    L = np.arange(len(n))*.05
    C = smooth_pinv(B, L)
    L = np.diag(L)
    D = np.dot(npl.inv(np.dot(B.T, B) + L*L), B.T)
    assert_array_almost_equal(C, D)

def test_normalize_data():

    sig = np.arange(1, 66)[::-1]

    bval = np.repeat([0, 1000], [2, 20])
    assert_raises(ValueError, normalize_data, sig, bval)
    bval = np.ones(65)*1000
    assert_raises(ValueError, normalize_data, sig, bval)
    bval = np.repeat([0, 1], [1, 64])
    d = normalize_data(sig, bval, 1)
    assert_raises(ValueError, normalize_data, None, bval, 0)

    bval[[0, 1]] = [0, 1]
    norm_sig = normalize_data(sig, bval, min_signal=1)
    assert_array_equal(norm_sig, sig/65.)
    norm_sig = normalize_data(sig, bval, min_signal=5)
    assert_array_equal(norm_sig[-5:], 5/65.)

    bval[[0, 1]] = [0, 0]
    norm_sig = normalize_data(sig, bval, min_signal=1)
    assert_array_equal(norm_sig, sig/64.5)
    norm_sig = normalize_data(sig, bval, min_signal=5)
    assert_array_equal(norm_sig[-5:], 5/64.5)

    sig = sig*np.ones((2,3,1))

    bval[[0, 1]] = [0, 1]
    norm_sig = normalize_data(sig, bval, min_signal=1)
    assert_array_equal(norm_sig, sig/65.)
    norm_sig = normalize_data(sig, bval, min_signal=5)
    assert_array_equal(norm_sig[..., -5:], 5/65.)

    bval[[0, 1]] = [0, 0]
    norm_sig = normalize_data(sig, bval, min_signal=1)
    assert_array_equal(norm_sig, sig/64.5)
    norm_sig = normalize_data(sig, bval, min_signal=5)
    assert_array_equal(norm_sig[..., -5:], 5/64.5)

def make_fake_signal():
    hemisphere = create_unit_hemisphere(4)
    v, e = hemisphere.vertices, hemisphere.edges
    vecs_xy = v[np.flatnonzero(v[:, 2] < .001)]
    evals = np.array([1.8, .2, .2])*10**-3*1.5
    evecs_moveing = np.empty((len(vecs_xy), 3, 3))
    evecs_moveing[:, :, 0] = vecs_xy
    evecs_moveing[:, :, 1] = [0, 0, 1]
    evecs_moveing[:, :, 2] = np.cross(evecs_moveing[:, :, 0],
                                      evecs_moveing[:, :, 1])
    assert ((evecs_moveing * evecs_moveing).sum(1) - 1 < .001).all()
    assert ((evecs_moveing * evecs_moveing).sum(2) - 1 < .001).all()

    gtab = np.empty((len(v) + 1, 3))
    bval = np.empty(len(v) + 1)
    bval[0] = 0
    bval[1:] = 2000
    gtab[0] = [0, 0, 0]
    gtab[1:] = v
    bvec = gtab.T
    B = design_matrix(bvec, bval)

    tensor_moveing = np.empty_like(evecs_moveing)
    for ii in xrange(len(vecs_xy)):
        tensor_moveing[ii] = np.dot(evecs_moveing[ii]*evals,
                                    evecs_moveing[ii].T)
    D_moveing = lower_triangular(tensor_moveing, 1)
    tensor_fixed = np.diag(evals)
    D_fixed = lower_triangular(tensor_fixed, 1)

    sig = .45*np.exp(np.dot(D_moveing, B.T)) + .55*np.exp(np.dot(B, D_fixed))
    assert sig.max() <= 1
    assert sig.min() > 0
    return hemisphere, vecs_xy, bval, bvec, sig

class SimpleInterpolator(object):
    data = np.ones((10, 10, 10))
    voxel_size = np.ones(3)
    def __getitem__(index):
        return 1.

class SimpleModel(object):
    def fit(data):
        return SimpleFit(object)

class SimpleFit(object):
    directions = np.array([[ 1.,  0.,  0.],
                           [ 0.,  1.,  0.],
                           [ 0.,  0.,  1.],
                          ])

def test_ClosestPeakSelector():
    sphere, vecs_xy, bval, bvec, sig = make_fake_signal()
    opdf_fitter = SlowAdcOpdfModel(bval, bvec.T, 6)
    opdf_fitter.angular_distance_threshold = 0.
    norm_sig = sig
    # angle_limit=49
    stepper = ClosestPeakSelector(opdf_fitter, norm_sig)
    stepper.angular_spacing_threshould = 0.
    S = opdf_fitter.fit(norm_sig).odf(sphere)
    for ii in xrange(len(vecs_xy)):
        """
        if np.dot(vecs_xy[ii], [0, 1., 0]) < .56:
            assert_raises(StopIteration, stepper.next_step, ii, [0, 1., 0])
        else:
        """
        step = stepper.next_step(ii, [0, 1., 0])
        s2 = stepper.next_step(ii, vecs_xy[ii])
        assert_array_equal(vecs_xy[ii], step)
        step = stepper.next_step(ii, [1., 0, 0.])
        assert_array_almost_equal([1., 0, 0.], step)
        print vecs_xy[ii]
        print step
        print s2

    norm_sig.shape = (2, 2, 4, -1)
    # angle_limit=49
    stepper = ClosestPeakSelector(opdf_fitter, norm_sig)
    step = stepper.next_step((0, 0, 0), [1, 0, 0])
    assert_array_almost_equal(step, [1, 0, 0])

def testQballOdfModel():
    sphere, vecs_xy, bval, bvec, sig = make_fake_signal()
    qball_fitter = QballOdfModel(bval, bvec.T, 6)
    qball_fitter.angular_spacing_threshould = 0.

    norm_sig = sig
    S = qball_fitter.fit(norm_sig).odf(sphere)
    # angle_limit=39
    stepper = ClosestPeakSelector(qball_fitter, norm_sig)
    for ii in xrange(len(vecs_xy)):
        """
        if np.dot(vecs_xy[ii], [0, 1., 0]) < .84:
            assert_raises(StopIteration, stepper.next_step, ii, [0, 1., 0])
        else:
        """
        step = stepper.next_step(ii, [0, 1., 0])
        s2 = stepper.next_step(ii, vecs_xy[ii])
        assert step is not None
        assert np.dot(vecs_xy[ii], step) > .98
        print ii
        step = stepper.next_step(ii, [1., 0, 0.])
        assert_array_almost_equal([1., 0, 0.], step)

def test_hat_and_lcr():
    hemi = create_unit_hemisphere(6)
    m, n = sph_harm_ind_list(8)
    B = real_sph_harm(m, n, hemi.phi[:, None], hemi.theta[:, None])
    H = hat(B)
    B_hat = np.dot(H, B)
    assert_array_almost_equal(B, B_hat)

    R = lcr_matrix(H)
    d = np.arange(len(hemi.theta))
    r = d - np.dot(H, d)
    lev = np.sqrt(1-H.diagonal())
    r /= lev
    r -= r.mean()

    r2 = np.dot(R, d)
    assert_array_almost_equal(r, r2)

    r3 = np.dot(d, R.T)
    assert_array_almost_equal(r, r3)

def test_bootstrap_array():
    B = np.array([[4, 5, 7, 4, 2.],
                  [4, 6, 2, 3, 6.]])
    H = hat(B.T)

    R = np.zeros((5,5))
    d = np.arange(1, 6)
    dhat = np.dot(H, d)

    assert_array_almost_equal(bootstrap_data_voxel(dhat, H, R), dhat)
    assert_array_almost_equal(bootstrap_data_array(dhat, H, R), dhat)

    H = np.zeros((5,5))

def test_ResidualBootstrapWrapper():
    B = np.array([[4, 5, 7, 4, 2.],
                  [4, 6, 2, 3, 6.]])
    B = B.T
    H = hat(B)
    d = np.arange(10)/8.
    d.shape = (2,5)
    dhat = np.dot(d, H)
    ms = .2

    boot_obj = ResidualBootstrapWrapper(dhat, B, ms)
    assert_array_almost_equal(boot_obj[0], dhat[0].clip(ms, 1))
    assert_array_almost_equal(boot_obj[1], dhat[1].clip(ms, 1))
